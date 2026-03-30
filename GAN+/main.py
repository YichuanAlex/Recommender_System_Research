import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel
from model_rqvae import GANQuantizerWrapper, MmEmbDataset
import torch.nn as nn
from model import NaryEncoder
import traceback


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=30000, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    try:
        print('==== [DEBUG] Start main.py ====')
        print('[DEBUG] Loading dataset...')
        Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
        Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
        log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
        writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
        # global dataset
        data_path = os.environ.get('TRAIN_DATA_PATH')

        args = get_args()
        print(f'[DEBUG] Args loaded: batch_size={args.batch_size}, maxlen={args.maxlen}, mm_emb_id={args.mm_emb_id}')
        
        print('[DEBUG] Creating MyDataset...')
        dataset = MyDataset(data_path, args)
        print(f'[DEBUG] MyDataset created: usernum={dataset.usernum}, itemnum={dataset.itemnum}')
        
        print('[DEBUG] Splitting dataset...')
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
        print(f'[DEBUG] Dataset split: train={len(train_dataset)}, valid={len(valid_dataset)}')
        
        print('[DEBUG] Creating train_loader...')
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
        )
        print(f'[DEBUG] Train_loader created: {len(train_loader)} batches')
        
        print('[DEBUG] Creating valid_loader...')
        valid_loader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
        )
        print(f'[DEBUG] Valid_loader created: {len(valid_loader)} batches')
        
        usernum, itemnum = dataset.usernum, dataset.itemnum
        feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types
        print(f'[DEBUG] Feature info: feat_statistics keys={list(feat_statistics.keys())}, feat_types keys={list(feat_types.keys())}')

        # 假设只用第一个多模态特征ID
        mm_emb_id = args.mm_emb_id[0]
        print('[DEBUG] Creating MmEmbDataset for mm_emb_id=81')
        
        # 只创建emb_81_32的dataset，避免内存问题
        try:
            mm_dataset = MmEmbDataset(data_path, '81')
            if len(mm_dataset) > 0:
                mm_loader = DataLoader(mm_dataset, batch_size=128, shuffle=True, collate_fn=MmEmbDataset.collate_fn)
                print(f'[DEBUG] mm_loader created successfully, dataset size: {len(mm_dataset)}')
            else:
                print('[WARNING] mm_dataset is empty')
                mm_loader = None
        except Exception as e:
            print(f'[WARNING] Failed to create mm_loader: {e}')
            mm_loader = None

        print('[DEBUG] Initializing GANQuantizerWrapper...')
        # 为emb_81_32创建GAN量化器
        gan_quantizer = None
        semantic_emb = None
        nary_encoder = None
        
        if mm_loader is not None:
            mm_emb_dim = 32  # emb_81_32的维度
            num_classes = 256  # 可以根据embedding维度调整
            gan_quantizer = GANQuantizerWrapper(
                input_dim=mm_emb_dim,
                hidden_channels=[128, 64],
                latent_dim=32,
                num_classes=num_classes,
                device=args.device
            )
            semantic_emb = nn.Embedding(num_classes, args.hidden_units).to(args.device)
            
            # 为emb_81_32创建N-ary编码器
            nary_bases = [2, 4, 8]
            nary_encoder = NaryEncoder(nary_bases, args.hidden_units).to(args.device)
            
            print(f'[DEBUG] Created quantizer for emb_81_32: dim={mm_emb_dim}, classes={num_classes}')
        else:
            print('[WARNING] mm_loader is None, skipping GAN quantizer creation')

        print('[DEBUG] Initializing BaselineModel...')
        model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args, gan_quantizer=gan_quantizer, semantic_emb=semantic_emb, nary_encoder=nary_encoder).to(args.device)
        print('[DEBUG] BaselineModel initialized')
        print('[DEBUG] Start training loop...')

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

        # 训练GAN量化器（在主模型训练之前）
        print('[DEBUG] Training GAN quantizer...')
        if mm_loader is not None and gan_quantizer is not None:
            optimizer_g = torch.optim.Adam(gan_quantizer.gan.generator.parameters(), lr=1e-3)
            optimizer_d = torch.optim.Adam(gan_quantizer.gan.discriminator.parameters(), lr=1e-3)
            for epoch in range(3):  # 训练3个epoch
                print(f'[DEBUG] GAN training epoch {epoch}')
                for batch_idx, (tid_batch, emb_batch) in enumerate(mm_loader):
                    print(f'[DEBUG] GAN fit batch {batch_idx}, emb_batch.shape={emb_batch.shape}')
                    gan_quantizer.fit(emb_batch, optimizer_g, optimizer_d, n_steps=1)
                    if batch_idx >= 10:  # 训练前10个batch
                        break
            print('[DEBUG] GAN quantizer training done')
        else:
            print('[WARNING] mm_loader or gan_quantizer is None, skipping GAN quantizer training')

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
                
            epoch_loss_sum = 0
            epoch_steps = 0
            
            for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                try:
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                    seq = seq.to(args.device)
                    pos = pos.to(args.device)
                    neg = neg.to(args.device)
                    pos_logits, neg_logits = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                    )
                    pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                        neg_logits.shape, device=args.device
                    )
                    optimizer.zero_grad()
                    
                    # 修复索引问题 - next_token_type是[batch_size, seq_len]，我们需要找到item token的位置
                    # 只对最后一个位置（target）计算损失
                    target_mask = (next_token_type[:, -1] == 1)  # [batch_size]
                    
                    if target_mask.sum() > 0:  # 确保有有效的target
                        loss = bce_criterion(pos_logits[target_mask], pos_labels[target_mask])
                        loss += bce_criterion(neg_logits[target_mask], neg_labels[target_mask])
                        
                        # 添加调试信息
                        if global_step % 100 == 0:
                            print(f'[DEBUG] Step {global_step}: target_mask.sum()={target_mask.sum()}, pos_logits.mean()={pos_logits.mean():.4f}, neg_logits.mean()={neg_logits.mean():.4f}, loss={loss.item():.4f}')
                    else:
                        # 如果没有有效的target，使用所有样本
                        loss = bce_criterion(pos_logits, pos_labels)
                        loss += bce_criterion(neg_logits, neg_labels)
                        
                        # 添加调试信息
                        if global_step % 100 == 0:
                            print(f'[DEBUG] Step {global_step}: No valid targets, using all samples, loss={loss.item():.4f}')

                    # 添加L2正则化
                    for param in model.item_emb.parameters():
                        loss += args.l2_emb * torch.norm(param)
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss_sum += loss.item()
                    epoch_steps += 1

                    log_json = json.dumps(
                        {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
                    )
                    log_file.write(log_json + '\n')
                    log_file.flush()
                    print(log_json)

                    writer.add_scalar('Loss/train_step', loss.item(), global_step)
                    writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    global_step += 1
                    
                except Exception as e:
                    print(f'[ERROR] Error in training step {step}: {e}')
                    continue

            # 记录epoch级别的指标
            avg_epoch_loss = epoch_loss_sum / epoch_steps if epoch_steps > 0 else 0
            writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
            print(f'[INFO] Epoch {epoch} average loss: {avg_epoch_loss:.4f}')

            # 验证
            model.eval()
            valid_loss_sum = 0
            valid_steps = 0
            with torch.no_grad():
                for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    try:
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                        seq = seq.to(args.device)
                        pos = pos.to(args.device)
                        neg = neg.to(args.device)
                        pos_logits, neg_logits = model(
                            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                        )
                        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                            neg_logits.shape, device=args.device
                        )
                        
                        # 修复索引问题 - 与训练循环保持一致
                        target_mask = (next_token_type[:, -1] == 1)  # [batch_size]
                        
                        if target_mask.sum() > 0:  # 确保有有效的target
                            loss = bce_criterion(pos_logits[target_mask], pos_labels[target_mask])
                            loss += bce_criterion(neg_logits[target_mask], neg_labels[target_mask])
                        else:
                            # 如果没有有效的target，使用所有样本
                            loss = bce_criterion(pos_logits, pos_labels)
                            loss += bce_criterion(neg_logits, neg_labels)
                        valid_loss_sum += loss.item()
                        valid_steps += 1
                    except Exception as e:
                        print(f'[ERROR] Error in validation step {step}: {e}')
                        continue
                        
            valid_loss_avg = valid_loss_sum / valid_steps if valid_steps > 0 else 0
            writer.add_scalar('Loss/valid', valid_loss_avg, global_step)
            print(f'[INFO] Epoch {epoch} validation loss: {valid_loss_avg:.4f}')

            # 按照平台规范保存checkpoint
            try:
                save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), 
                              f"global_step{global_step}.epoch={epoch}.lr={args.lr}.hidden={args.hidden_units}.maxlen={args.maxlen}.valid_loss={valid_loss_avg:.4f}")
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_dir / "model.pt")
                
                # 同时保存GAN量化器状态
                if gan_quantizer is not None:
                    torch.save(gan_quantizer.state_dict(), save_dir / "gan_quantizer.pt")
                    
                print(f'[INFO] Checkpoint saved to {save_dir}')
            except Exception as e:
                print(f'[ERROR] Failed to save checkpoint: {e}')

        print("Done")
        writer.close()
        log_file.close()
        print('==== [DEBUG] End main.py ====')
    except Exception as e:
        print('[ERROR] Exception occurred:')
        traceback.print_exc()
        exit(1)
