import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel
# 导入新的RQVAEEnhancedModel
from model_rqvae import RQVAEEnhancedModel


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81','82'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    # 在main.py中添加以下代码，位置在rqvae_config定义之后，model初始化之前
    
    # RQ-VAE配置参数
    rqvae_config = {
        'input_dim': 1024,  # 根据多模态嵌入维度调整
        'hidden_channels': [512, 256],
        'latent_dim': 128,
        'num_codebooks': 4,
        'codebook_size': [64, 64, 64, 64],
        'shared_codebook': False,
        'kmeans_method': 'kmeans',
        'kmeans_iters': 100,
        'distances_method': 'l2',
        'loss_beta': 0.25
    }
    
    # 动态设置input_dim，根据实际使用的特征ID来确定维度
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    # 确定实际使用的特征ID
    actual_feat_id = args.mm_emb_id[0] if args.mm_emb_id else '82'
    rqvae_config['input_dim'] = SHAPE_DICT.get(actual_feat_id, 1024)
    print(f"Using feature ID: {actual_feat_id}, input_dim: {rqvae_config['input_dim']}")
    
    # 初始化模型 - 替换为RQVAEEnhancedModel
    model = RQVAEEnhancedModel(usernum, itemnum, feat_statistics, feat_types, args, rqvae_config).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
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
        for step, (
            seq,
            pos,
            neg,
            token_type,
            next_token_type,
            next_action_type,
            seq_feat,
            pos_feat,
            neg_feat
        ) in enumerate(tqdm(train_loader, desc="Training")):
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            
            # 获取多模态嵌入
            multimodal_emb_list = []
            for i in range(len(seq_feat)):
                # 从seq_feat中提取多模态嵌入
                # 这里需要根据实际数据结构调整
                multimodal_emb = None
                for feat_dict in seq_feat[i]:
                    # 优先使用指定的多模态特征ID
                    for emb_id in args.mm_emb_id:
                        if emb_id in feat_dict and isinstance(feat_dict[emb_id], np.ndarray):
                            multimodal_emb = feat_dict[emb_id]
                            break
                    # 如果没有找到指定的特征ID，尝试使用任何可用的多模态嵌入
                    if multimodal_emb is None:
                        for feat_id, feat_value in feat_dict.items():
                            if feat_id in args.mm_emb_id and isinstance(feat_value, np.ndarray):
                                multimodal_emb = feat_value
                                break
                    if multimodal_emb is not None:
                        break
                
                if multimodal_emb is not None:
                    multimodal_emb_list.append(multimodal_emb)
                else:
                    # 如果没有多模态嵌入，使用零向量，维度与rqvae_config['input_dim']一致
                    multimodal_emb_list.append(np.zeros(rqvae_config['input_dim']))
            
            # 在获取多模态嵌入后，增加轻微的噪声以减少重复样本
            # 这可以帮助缓解k-means聚类中的收敛问题
            noise_factor = 1e-6
            multimodal_emb = torch.tensor(np.array(multimodal_emb_list), dtype=torch.float32).to(args.device)
            multimodal_emb += torch.randn_like(multimodal_emb) * noise_factor
            
            # 添加调试信息
            print(f"Batch size: {multimodal_emb.shape[0]}, Feature dimension: {multimodal_emb.shape[1]}")
            
            # 使用新的前向传播方法
            pos_logits, neg_logits, recon_loss, rqvae_loss, total_loss, semantic_embeddings = model.forward_with_rqvae(
                seq,
                pos,
                neg,
                token_type,
                next_token_type,
                next_action_type,
                seq_feat,
                pos_feat,
                neg_feat,
                multimodal_emb
            )
            
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            optimizer.zero_grad()
            indices = np.where(next_token_type.cpu().numpy() == 1)
            
            # 修改损失计算，包含RQ-VAE损失
            bce_loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            bce_loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            loss = bce_loss + total_loss  # 添加RQ-VAE总损失
            
            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'bce_loss': bce_loss.item(), 
                 'recon_loss': recon_loss.item(), 'rqvae_loss': rqvae_loss.item(), 
                 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Loss/BCE', bce_loss.item(), global_step)
            writer.add_scalar('Loss/Recon', recon_loss.item(), global_step)
            writer.add_scalar('Loss/RQVAE', rqvae_loss.item(), global_step)

            global_step += 1

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss_sum = 0
        with torch.no_grad():
            for step, (
                seq,
                pos,
                neg,
                token_type,
                next_token_type,
                next_action_type,
                seq_feat,
                pos_feat,
                neg_feat
            ) in enumerate(tqdm(valid_loader, desc="Validation")):
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
                
                # 获取多模态嵌入
                multimodal_emb_list = []
                for i in range(len(seq_feat)):
                    multimodal_emb = None
                    for feat_dict in seq_feat[i]:
                        # 优先使用指定的多模态特征ID
                        for emb_id in args.mm_emb_id:
                            if emb_id in feat_dict and isinstance(feat_dict[emb_id], np.ndarray):
                                multimodal_emb = feat_dict[emb_id]
                                break
                            # 如果没有找到指定的特征ID，尝试使用任何可用的多模态嵌入
                            if multimodal_emb is None:
                                for feat_id, feat_value in feat_dict.items():
                                    if feat_id in args.mm_emb_id and isinstance(feat_value, np.ndarray):
                                        multimodal_emb = feat_value
                                        break
                            if multimodal_emb is not None:
                                break
                        
                        if multimodal_emb is not None:
                            multimodal_emb_list.append(multimodal_emb)
                        else:
                            # 如果没有多模态嵌入，使用零向量，维度与rqvae_config['input_dim']一致
                            multimodal_emb_list.append(np.zeros(rqvae_config['input_dim']))
                    
                multimodal_emb = torch.tensor(np.array(multimodal_emb_list), dtype=torch.float32).to(args.device)
                
                # 使用新的前向传播方法
                pos_logits, neg_logits, recon_loss, rqvae_loss, total_loss, semantic_embeddings = model.forward_with_rqvae(
                    seq,
                    pos,
                    neg,
                    token_type,
                    next_token_type,
                    next_action_type,
                    seq_feat,
                    pos_feat,
                    neg_feat,
                    multimodal_emb
                )
                
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                    neg_logits.shape, device=args.device
                )
                indices = np.where(next_token_type.cpu().numpy() == 1)
                
                # 修改损失计算，包含RQ-VAE损失
                bce_loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                bce_loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                loss = bce_loss + total_loss  # 添加RQ-VAE总损失
                valid_loss_sum += loss.item()
                
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
