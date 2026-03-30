import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset, save_emb
from model import SFGModel


def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=48, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--l2_emb', default=0.0001, type=float)

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--cross_layers', default=2, type=int, help='Cross Network layers')
    parser.add_argument('--use_feature_generator', action='store_true', help='Whether to use feature generator')
    parser.add_argument('--num_generated_features', default=10, type=int, help='Number of generated features')
    parser.add_argument('--num_embedding_sets', default=2, type=int, help='Number of embedding sets for Multi-Embedding')

    # SFG Model construction parameters (matching main.py)
    parser.add_argument('--aggregation_type', default='mean', type=str, help='Aggregation type for multi-embedding: mean, weighted, adaptive')
    parser.add_argument('--multi_head', action='store_true', help='Whether to use multi-head attention for feature aggregation')
    parser.add_argument('--agg_num_heads', default=8, type=int, help='Number of heads for multi-head attention in feature aggregation')


    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81','82'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    # Align with main.py: allow env overrides
    args.aggregation_type = os.environ.get('AGGREGATION_TYPE', args.aggregation_type)
    args.multi_head = bool(int(os.environ.get('MULTI_HEAD', '0')))
    args.num_heads = int(os.environ.get('NUM_HEADS', args.num_heads))

    return args


def read_result_ids(file_path):
    with open(file_path, 'rb') as f:
        # Read the header (num_points_query and FLAGS_query_ann_top_k)
        num_points_query = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes
        query_ann_top_k = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes

        print(f"num_points_query: {num_points_query}, query_ann_top_k: {query_ann_top_k}")

        # Calculate how many result_ids there are (num_points_query * query_ann_top_k)
        num_result_ids = num_points_query * query_ann_top_k

        # Read result_ids (uint64_t, 8 bytes per value)
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)

        return result_ids.reshape((num_points_query, query_ann_top_k))


def process_cold_start_feat(feat):
    """
    处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
    """
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if type(feat_value) == list:
            value_list = []
            for v in feat_value:
                if type(v) == str:
                    value_list.append(0)
                else:
                    value_list.append(v)
            processed_feat[feat_id] = value_list
        elif type(feat_value) == str:
            processed_feat[feat_id] = 0
        else:
            processed_feat[feat_id] = feat_value
    return processed_feat


# ---------------------- Torch-based ANN helpers (batched matmul Top-K) ----------------------
def _load_fbin(path: Path) -> np.ndarray:
    with open(path, 'rb') as f:
        num_points = struct.unpack('I', f.read(4))[0]
        dim = struct.unpack('I', f.read(4))[0]
        data = np.fromfile(f, dtype=np.float32, count=num_points * dim)
    return data.reshape(num_points, dim)


def _load_u64bin(path: Path) -> np.ndarray:
    with open(path, 'rb') as f:
        num_points = struct.unpack('I', f.read(4))[0]
        dim = struct.unpack('I', f.read(4))[0]
        data = np.fromfile(f, dtype=np.uint64, count=num_points * dim)
    return data.reshape(num_points, dim)


def _write_u64_topk(path: Path, ids_2d: np.ndarray):
    ids_2d = ids_2d.astype(np.uint64, copy=False)
    with open(path, 'wb') as f:
        f.write(struct.pack('II', ids_2d.shape[0], ids_2d.shape[1]))
        ids_2d.tofile(f)


def torch_ann_search(query_np: np.ndarray, item_fbin_path: Path, item_id_path: Path, result_path: Path, topk: int = 10, device: str = 'cuda', qbatch: int = 256, item_chunk: int = 50000):
    """
    使用 Torch 实现的 batched ANN：按块对 item 向量做矩阵乘法，逐块维护每个 query 的 Top-K，避免显存爆。
    写出与原 faiss_demo 一致的 id100.u64bin 文件（uint32 header + uint64 ids）。
    """
    device = torch.device(device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')
    # 查询向量
    queries = torch.from_numpy(query_np.astype(np.float32, copy=False))
    # item 向量与其 ID
    items_np = _load_fbin(item_fbin_path)
    item_ids_np = _load_u64bin(item_id_path).reshape(-1)

    num_items = items_np.shape[0]
    num_queries = queries.shape[0]

    results_ids = []
    with torch.no_grad():
        # 将查询按批次送入设备
        for q_start in tqdm(range(0, num_queries, qbatch), desc='ANN[queries]'):
            q_end = min(q_start + qbatch, num_queries)
            q = queries[q_start:q_end].to(device, non_blocking=True)  # [B, D]

            # 初始化当前批次的全局Top-K
            best_scores = torch.full((q.shape[0], topk), -float('inf'), device=device)
            best_ids = torch.zeros((q.shape[0], topk), dtype=torch.long, device=device)

            # 逐块扫描 items
            for i_start in range(0, num_items, item_chunk):
                i_end = min(i_start + item_chunk, num_items)
                items_chunk = torch.from_numpy(items_np[i_start:i_end]).to(device, non_blocking=True)  # [C, D]

                # 相似度：内积（与训练时打分一致）
                # [B, D] @ [D, C] -> [B, C]
                scores = q @ items_chunk.t()

                k_local = min(topk, scores.shape[1])
                local_vals, local_idx = torch.topk(scores, k=k_local, dim=1)
                # 将局部 idx 映射为全局 item 索引
                global_idx = local_idx + i_start
                # 获取对应的检索ID（注意uint64），先转为long存储，最终写文件时再转uint64
                # 我们只在 CPU 上持久化，先在 device 上维护，再一次性搬回
                # 需要从 numpy 的 item_ids_np 获取值
                ids_chunk = torch.from_numpy(item_ids_np[global_idx.detach().cpu().numpy()])  # [B, k_local]
                ids_chunk = ids_chunk.to(device)

                # 合并进全局 Top-K
                comb_scores = torch.cat([best_scores, local_vals], dim=1)  # [B, 2K]
                comb_ids = torch.cat([best_ids, ids_chunk.long()], dim=1)   # [B, 2K]
                sel = torch.topk(comb_scores, k=topk, dim=1).indices        # [B, K]
                best_scores = torch.gather(comb_scores, 1, sel)
                best_ids = torch.gather(comb_ids, 1, sel)
                # 释放临时张量
                del scores, local_vals, local_idx, global_idx, ids_chunk, comb_scores, comb_ids, sel

            # 将当前批次最佳ID保存
            results_ids.append(best_ids.detach().cpu().numpy().astype(np.uint64))
            del q, best_scores, best_ids
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    final_ids = np.concatenate(results_ids, axis=0)  # [Q, K]
    _write_u64_topk(result_path, final_ids)


def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model):
    """
    生产候选库item的id和embedding

    Args:
        indexer: 索引字典
        feat_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feature_default_value: 特征缺省值
        mm_emb_dict: 多模态特征字典
        model: 模型
    Returns:
        retrieve_id2creative_id: 索引id->creative_id的dict
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}

    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            # 读取item特征，并补充缺失值
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer[creative_id] if creative_id in indexer else 0
            missing_fields = set(
                feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
            ) - set(feature.keys())
            feature = process_cold_start_feat(feature)
            for feat_id in missing_fields:
                feature[feat_id] = feat_default_value[feat_id]
            for feat_id in feat_types['item_emb']:
                if creative_id in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][creative_id]
                else:
                    feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    # 保存候选库的embedding和sid
    model.save_item_emb(item_ids, retrieval_ids, features, os.environ.get('EVAL_RESULT_PATH'))
    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)
    return retrieve_id2creative_id


def infer():
    args = get_args()
    data_path = os.environ.get('EVAL_DATA_PATH')
    test_dataset = MyTestDataset(data_path, args)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=int(os.environ.get('DATALOADER_WORKERS','1')), pin_memory=True, prefetch_factor=2, persistent_workers=True, collate_fn=test_dataset.collate_fn)
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
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
    model.eval()

    ckpt_path = get_ckpt_path()
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
    all_embs = []
    user_list = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_loader), total=len(test_loader), mininterval=1.0):
            seq, token_type, seq_feat, user_id = batch
            seq = seq.to(args.device, non_blocking=True)

            # Create dummy pos_seqs and neg_seqs for inference
            # Use the last item in the sequence as a dummy positive
            pos_seqs_dummy = seq[:, -1]
            # Dummy negative, shape (batch_size, 1) - values don't matter for inference of user embedding
            neg_seqs_dummy = torch.zeros_like(seq[:, :1])

            # Dummy token_type, next_token_type, next_action_type for inference
            # These are not used in SFGModel's forward for feature processing, but are required by signature.
            dummy_token_type = torch.zeros_like(seq)
            dummy_next_token_type = torch.zeros_like(seq)
            dummy_next_action_type = torch.zeros_like(seq)

            # Call the forward method of SFGModel
            # The forward method expects (log_seqs, pos_seqs, neg_seqs, token_type, next_token_type, next_action_type, seq_feature, pos_feature, neg_feature)
            # In inference, pos_feature and neg_feature are not directly used for user embedding generation,
            dummy_pos_feature = {}
            dummy_neg_feature = {}
            _, _, _, _, encoder_output = model(seq, pos_seqs_dummy, neg_seqs_dummy, dummy_token_type, dummy_next_token_type, dummy_next_action_type, seq_feat, dummy_pos_feature, dummy_neg_feature)

            # In inference, we typically want the representation of the user or the last item in the sequence.
            # Assuming `logits` here refers to the user's embedding for retrieval.
            # `encoder_output` is (batch_size, seq_len, hidden_units).
            # A common approach is to take the embedding of the last item in the sequence.
            logits = encoder_output[:, -1, :]
            for i in range(logits.shape[0]):
                emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
                all_embs.append(emb)
            user_list += user_id
    
    # 生成候选库的embedding 以及 id文件
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
    )
    all_embs = np.concatenate(all_embs, axis=0)
    # 保存query文件
    save_emb(all_embs, Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin'))

    # 使用基于Torch的ANN检索，写出与faiss_demo一致的结果文件
    dataset_vector_file = Path(os.environ.get("EVAL_RESULT_PATH"), "embedding.fbin")
    dataset_id_file = Path(os.environ.get("EVAL_RESULT_PATH"), "id.u64bin")
    result_id_file = Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin")
    qbatch = int(os.environ.get('ANN_QBATCH', '256'))
    item_chunk = int(os.environ.get('ANN_ITEM_CHUNK', '50000'))
    torch_ann_search(
        all_embs,
        dataset_vector_file,
        dataset_id_file,
        result_id_file,
        topk=10,
        device=args.device,
        qbatch=qbatch,
        item_chunk=item_chunk,
    )

    # 取出top-k
    top10s_retrieved = read_result_ids(Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin"))
    top10s_untrimmed = []
    for top10 in tqdm(top10s_retrieved):
        for item in top10:
            top10s_untrimmed.append(retrieve_id2creative_id.get(int(item), 0))

    top10s = [top10s_untrimmed[i : i + 10] for i in range(0, len(top10s_untrimmed), 10)]

    return top10s, user_list
