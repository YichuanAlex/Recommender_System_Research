import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel


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
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args


def print_mm_emb_info(dataset):
    """
    打印所有特征数据的详细信息
    """ 
    print("==================== 数据特征详细信息 ====================")
    print(f"用户数量: {dataset.usernum}")
    print(f"物品数量: {dataset.itemnum}")
    
    # 检查特征ID是否有重复
    if len(dataset.mm_emb_ids) != len(set(dataset.mm_emb_ids)):
        print("警告：发现重复的特征ID")
    
    # 特征统计信息
    feat_statistics = dataset.feat_statistics
    feature_types = dataset.feature_types
    
    print("\n用户特征:")
    print(f"  稀疏特征: {feature_types['user_sparse']}")
    for feat_id in feature_types['user_sparse']:
        print(f"    特征{feat_id}的数量: {feat_statistics[feat_id]}")
        # 打印用户特征样例
        if 'f' in dataset.indexer and feat_id in dataset.indexer['f']:
            print(f"      样例值: {list(dataset.indexer['f'][feat_id].keys())[:5]} (显示前5个)")
        
    print(f"  数组特征: {feature_types['user_array']}")
    for feat_id in feature_types['user_array']:
        print(f"    特征{feat_id}的数量: {feat_statistics[feat_id]}")
        # 打印用户数组特征样例
        if 'f' in dataset.indexer and feat_id in dataset.indexer['f']:
            print(f"      样例值: {list(dataset.indexer['f'][feat_id].keys())[:5]} (显示前5个)")
        
    print(f"  连续特征: {feature_types['user_continual']}")
    for feat_id in feature_types['user_continual']:
        print(f"    特征{feat_id}: 连续特征")
        
    print("\n物品特征:")
    print(f"  稀疏特征: {feature_types['item_sparse']}")
    for feat_id in feature_types['item_sparse']:
        print(f"    特征{feat_id}的数量: {feat_statistics[feat_id]}")
        # 打印物品特征样例
        if 'f' in dataset.indexer and feat_id in dataset.indexer['f']:
            print(f"      样例值: {list(dataset.indexer['f'][feat_id].keys())[:5]} (显示前5个)")
        
    print(f"  数组特征: {feature_types['item_array']}")
    for feat_id in feature_types['item_array']:
        print(f"    特征{feat_id}的数量: {feat_statistics[feat_id]}")
        # 打印物品数组特征样例
        if 'f' in dataset.indexer and feat_id in dataset.indexer['f']:
            print(f"      样例值: {list(dataset.indexer['f'][feat_id].keys())[:5]} (显示前5个)")
        
    print(f"  连续特征: {feature_types['item_continual']}")
    for feat_id in feature_types['item_continual']:
        print(f"    特征{feat_id}: 连续特征")
        
    print(f"\n多模态特征ID: {dataset.mm_emb_ids}")
    
    # 多模态特征维度
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    for feat_id in dataset.mm_emb_ids:
        print(f"  特征{feat_id}的维度: {SHAPE_DICT[feat_id]}")
        
    # 多模态特征数据情况
    for feat_id in dataset.mm_emb_ids:
        emb_dict = dataset.mm_emb_dict[feat_id]
        print(f"\n特征{feat_id}的数据情况:")
        print(f"  物品数量: {len(emb_dict)}")
        
        # 打印多模态特征样例
        if emb_dict:
            print(f"  样例键值对 (显示前3个):")
            for i, (key, value) in enumerate(emb_dict.items()):
                if i >= 3:
                    break
                print(f"    {key}: {value[:10]}... (显示前10个维度)")
        
        # 检查是否有缺失的物品
        total_items = set(dataset.indexer_i_rev.keys())
        items_with_emb = set(emb_dict.keys())
        missing_items = total_items - items_with_emb
        print(f"  缺失物品数量: {len(missing_items)}")
        
        # 计算emb文件大小
        data_path = os.environ.get('TRAIN_DATA_PATH')
        if data_path:
            emb_dir = os.path.join(data_path, 'creative_emb', f'emb_{feat_id}_{SHAPE_DICT[feat_id]}')
            total_size = 0
            if os.path.exists(emb_dir):
                print(f"  正在计算特征{feat_id}的文件大小...")
                for file in os.listdir(emb_dir):
                    if file.startswith('part'):
                        file_path = os.path.join(emb_dir, file)
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        print(f"    文件 {file} 大小: {file_size} bytes")
            # 转换为MB
            size_mb = total_size / (1024 * 1024)
            print(f"  emb文件大小: {size_mb:.2f} MB")
        
        if len(total_items) > 0:
            coverage = len(items_with_emb) / len(total_items) * 100
            print(f"  特征覆盖率: {coverage:.2f}%")
        else:
            print("  特征覆盖率: 0.00%")
    
    # 在print_mm_emb_info函数中添加以下代码
    
    # 打印用户特征键值对样例
    print("\n用户特征键值对样例:")
    # 遍历用户特征类型
    for feat_type, feat_ids in feature_types.items():
        if feat_type.startswith('user'):
            print(f"  {feat_type}:")
            for feat_id in feat_ids[:2]:  # 每种类型只打印前2个特征ID
                if 'f' in dataset.indexer and feat_id in dataset.indexer['f']:
                    print(f"    特征{feat_id}:")
                    # 打印前3个键值对
                    for i, (key, value) in enumerate(dataset.indexer['f'][feat_id].items()):
                        if i >= 3:
                            break
                        print(f"      {key} -> {value}")
    
    # 打印物品特征键值对样例
    print("\n物品特征键值对样例:")
    # 遍历物品特征类型
    for feat_type, feat_ids in feature_types.items():
        if feat_type.startswith('item') and not feat_type.endswith('emb'):
            print(f"  {feat_type}:")
            for feat_id in feat_ids[:2]:  # 每种类型只打印前2个特征ID
                if 'f' in dataset.indexer and feat_id in dataset.indexer['f']:
                    print(f"    特征{feat_id}:")
                    # 打印前3个键值对
                    for i, (key, value) in enumerate(dataset.indexer['f'][feat_id].items()):
                        if i >= 3:
                            break
                        print(f"      {key} -> {value}")
    
    # 打印物品特征字典样例（来自item_feat_dict）
    print("\n物品特征字典样例 (来自item_feat_dict):")
    for i, (item_id, feat_dict) in enumerate(dataset.item_feat_dict.items()):
        if i >= 3:  # 只打印前3个物品的特征
            break
        print(f"  物品 {item_id}:")
        # 每个物品只打印前3个特征
        printed_features = 0
        for feat_id, feat_value in feat_dict.items():
            if printed_features >= 3:
                break
            print(f"    特征 {feat_id}: {feat_value}")
            printed_features += 1
    
    # 打印索引映射样例
    print("\n索引映射样例:")
    print("  用户索引 (显示前5个):")
    for i, (reid, user_id) in enumerate(dataset.indexer_u_rev.items()):
        if i >= 5:
            break
        print(f"    {reid} -> {user_id}")
        
    print("  物品索引 (显示前5个):")
    for i, (reid, item_id) in enumerate(dataset.indexer_i_rev.items()):
        if i >= 5:
            break
        print(f"    {reid} -> {item_id}")
        
    print("=============================================================")


if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    dataset = MyDataset(data_path, args)
    
    # 打印所有特征数据的详细信息
    print_mm_emb_info(dataset)
    sys.exit(0)
    
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

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
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
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
            indices = np.where(next_token_type == 1)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss_sum = 0
        for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
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
            indices = np.where(next_token_type == 1)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            valid_loss_sum += loss.item()
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
