import argparse
import json
import os
import time
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import SFGModel


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=32, type=int)  # 减少从64到32
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=48, type=int)  # 减少从64到48
    parser.add_argument('--num_blocks', default=1, type=int)  # 减少从2到1
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--num_heads', default=4, type=int)  # 减少从8到4
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--l2_emb', default=0.0001, type=float)

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--cross_layers', default=2, type=int, help='Cross Network layers')

    # SFG Model construction
    parser.add_argument('--use_feature_generator', action='store_true')
    parser.add_argument('--num_generated_features', default=1, type=int)
    parser.add_argument('--reconstruction_loss_weight', default=0.1, type=float, help='Weight for reconstruction loss')
    parser.add_argument('--generation_loss_weight', default=0.1, type=float, help='Weight for generation loss')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81', '82'], type=str, choices=[str(s) for s in range(81, 87)])
    parser.add_argument('--num_embedding_sets', default=2, type=int, help='Number of embedding sets for Multi-Embedding')  # 减少从4到2

    args = parser.parse_args()
    return args

def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'a', buffering=1)
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    # 支持SFG创新点参数
    args.aggregation_type = os.environ.get('AGGREGATION_TYPE', 'mean') # 可选: mean, weighted, adaptive
    args.multi_head = bool(int(os.environ.get('MULTI_HEAD', '0')))
    args.num_heads = int(os.environ.get('NUM_HEADS', args.num_heads))
    dataset = MyDataset(data_path, args)
    total_len = len(dataset)
    train_len = int(total_len * 0.9)
    valid_len = total_len - train_len
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_len, valid_len])

    workers = int(os.environ.get('DATALOADER_WORKERS', '2'))
    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
            prefetch_factor=2 if workers>0 else None,
            persistent_workers=(workers>0),
            collate_fn=dataset.collate_fn
        )

    train_loader = make_loader(train_dataset, shuffle=True)
    valid_loader = make_loader(valid_dataset, shuffle=False)

    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = SFGModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    # 自动传递聚合方式、多头输出等参数
    if hasattr(args, 'aggregation_type'):
        model.aggregation_type = args.aggregation_type
    if hasattr(args, 'multi_head'):
        model.multi_head = args.multi_head
    if hasattr(args, 'num_heads'):
        model.num_heads = args.num_heads
    # 可插拔自适应融合MLP示例
    if getattr(model, 'aggregation_type', None) == 'adaptive':
        model.fusion_mlp = torch.nn.Sequential(
            torch.nn.Linear(model.hidden_units * args.num_embedding_sets, model.hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(model.hidden_units, model.hidden_units)
        )

    for name, param in model.named_parameters():
        if param.dim() >= 2:
            try:
                torch.nn.init.xavier_normal_(param.data)
            except Exception:
                pass

    if hasattr(model, 'pos_emb') and hasattr(model.pos_emb, 'weight'):
        with torch.no_grad():
            model.pos_emb.weight.data[0, :] = 0
    if hasattr(model, 'item_emb') and hasattr(model, 'user_emb'):
        with torch.no_grad():
            for i in range(args.num_embedding_sets):
                if i < len(model.item_emb):
                    model.item_emb[i].weight.data[0, :] = 0
                if i < len(model.user_emb):
                    model.user_emb[i].weight.data[0, :] = 0
    if hasattr(model, 'sparse_emb'):
        with torch.no_grad():
            for k in model.sparse_emb:
                model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except Exception:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), mininterval=1.0):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device, non_blocking=True)
            pos = pos.to(args.device, non_blocking=True)
            neg = neg.to(args.device, non_blocking=True)
            # 将特征字典搬到device，避免后续生成式损失计算设备不一致
            for feat_dict in [seq_feat, pos_feat, neg_feat]:
                for key in feat_dict:
                    feat_dict[key] = feat_dict[key].to(args.device, non_blocking=True)
            pos_logits, neg_logits, reconstructed_features, generated_features, encoder_output = model(
                seq,
                pos,
                neg,
                token_type,
                next_token_type,
                next_action_type,
                seq_feature=seq_feat,
                 pos_feature=pos_feat,
                 neg_feature=neg_feat
            )
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            optimizer.zero_grad(set_to_none=True)
            indices = np.where(next_token_type == 1)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            # 计算生成式损失
            if args.use_feature_generator and reconstructed_features is not None:
                reconstruction_loss = 0.0
                for feat_name, recon_tensor in reconstructed_features.items():
                    original_feat_name = feat_name.replace('_reconstructed', '')
                    if original_feat_name in seq_feat:
                        if len(seq_feat[original_feat_name].shape) == 3:
                            target_tensor = seq_feat[original_feat_name][indices, -1, :]
                        else:
                            target_tensor = seq_feat[original_feat_name][indices]
                        
                        # 确保recon_tensor和target_tensor在计算MSE之前没有NaN
                        recon_tensor = torch.nan_to_num(recon_tensor, nan=0.0)
                        target_tensor = torch.nan_to_num(target_tensor, nan=0.0)

                        reconstruction_loss += F.mse_loss(recon_tensor[indices], target_tensor.float()[indices])
                loss += args.reconstruction_loss_weight * reconstruction_loss

            if args.use_feature_generator and generated_features is not None:
                generation_loss = 0.0
                for feat_name, gen_tensor in generated_features.items():
                    if feat_name in seq_feat:
                        if len(seq_feat[feat_name].shape) == 3:
                            target_tensor = seq_feat[feat_name][indices, -1, :]
                        else:
                            target_tensor = seq_feat[feat_name][indices]
                        
                        # 确保gen_tensor和target_tensor在计算MSE之前没有NaN
                        gen_tensor = torch.nan_to_num(gen_tensor, nan=0.0)
                        target_tensor = torch.nan_to_num(target_tensor, nan=0.0)

                        generation_loss += F.mse_loss(gen_tensor[indices], target_tensor.float()[indices])
                loss += args.generation_loss_weight * generation_loss

            log_json = json.dumps(
                {'global_step': global_step, 'loss': float(loss.item()), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            print(log_json)

            writer.add_scalar('Loss/train', float(loss.item()), global_step)

            global_step += 1

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            # 【新增】梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 设置一个合适的max_norm值，例如5.0
            optimizer.step()

            if step % 50 == 0:
                _cleanup()

        model.eval()
        valid_loss_sum = 0.0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader), mininterval=1.0):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                seq = seq.to(args.device, non_blocking=True)
                pos = pos.to(args.device, non_blocking=True)
                neg = neg.to(args.device, non_blocking=True)

                # Move feature dictionaries to device
                for feat_dict in [seq_feat, pos_feat, neg_feat]:
                    for key in feat_dict:
                        feat_dict[key] = feat_dict[key].to(args.device, non_blocking=True)
                pos_logits, neg_logits, reconstructed_features, generated_features, encoder_output = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                )
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                    neg_logits.shape, device=args.device
                )
                indices = np.where(next_token_type == 1)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])

                if args.use_feature_generator and reconstructed_features is not None:
                    reconstruction_loss = 0.0
                    for feat_id, recon_tensor in reconstructed_features.items():
                        if feat_id in seq_feat:
                            if len(seq_feat[feat_id].shape) == 3:
                                target_tensor = seq_feat[feat_id][indices, -1, :]
                            else:
                                target_tensor = seq_feat[feat_id][indices]
                            reconstruction_loss += F.mse_loss(recon_tensor[indices], target_tensor.float()[indices])
                    loss += args.reconstruction_loss_weight * reconstruction_loss

                if args.use_feature_generator and generated_features is not None:
                    generation_loss = 0.0
                    for feat_id, gen_tensor in generated_features.items():
                        if feat_id in seq_feat:
                            if len(seq_feat[feat_id].shape) == 3:
                                target_tensor = seq_feat[feat_id][indices, -1, :]
                            else:
                                target_tensor = seq_feat[feat_id][indices]
                            
                            # 确保gen_tensor和target_tensor在计算MSE之前没有NaN
                            gen_tensor = torch.nan_to_num(gen_tensor, nan=0.0)
                            target_tensor = torch.nan_to_num(target_tensor, nan=0.0)

                            # 简化索引逻辑，确保gen_tensor和target_tensor形状一致
                            if gen_tensor.shape == target_tensor.shape:
                                generation_loss += F.mse_loss(gen_tensor, target_tensor.float())
                            else:
                                # 可以选择记录警告或跳过此特征的损失计算
                                logger.warning(f"Shape mismatch for feat_id {feat_id} in validation: gen_tensor {gen_tensor.shape} vs target_tensor {target_tensor.shape}")
                    loss += args.generation_loss_weight * generation_loss
                valid_loss_sum += loss.item()

                if step % 50 == 0:
                    _cleanup()

        valid_loss_avg = valid_loss_sum / max(1, len(valid_loader))
        writer.add_scalar('Loss/valid', valid_loss_avg, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_avg:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

        _cleanup()

    print("Done")
    writer.close()
    log_file.close()
