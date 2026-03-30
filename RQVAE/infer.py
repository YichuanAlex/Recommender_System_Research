import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset, load_mm_emb, save_emb
from model_rqvae import RQVAEEnhancedModel


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
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
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
                if feat_id in mm_emb_dict and creative_id in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][creative_id]
                else:
                    # 改进缺失值填充
                    feature[feat_id] = np.zeros(EMB_SHAPE_DICT.get(feat_id, 1024), dtype=np.float32)

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    # 使用save_item_emb_with_rqvae方法生成候选库的embedding和sid
    model.save_item_emb_with_rqvae(item_ids, retrieval_ids, features, os.environ.get('EVAL_RESULT_PATH'), mm_emb_dict)
    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)
    return retrieve_id2creative_id


def infer(args=None, test_dataset=None, model_path=None, save_dir=None, feat_statistics=None, feat_types=None):
    # 添加调试信息
    print("Starting infer function")
    print(f"args: {args}")
    print(f"test_dataset: {test_dataset}")
    print(f"model_path: {model_path}")
    print(f"save_dir: {save_dir}")
    print(f"feat_statistics: {feat_statistics}")
    print(f"feat_types: {feat_types}")
    
    # 如果参数为None，则使用默认值或从环境变量中获取
    if args is None:
        args = get_args()
    
    if model_path is None:
        model_path = os.environ.get('MODEL_OUTPUT_PATH')
    
    if save_dir is None:
        save_dir = os.environ.get('EVAL_RESULT_PATH')
    
    # 创建test_dataset（如果未提供）
    if test_dataset is None:
        data_path = os.environ.get('EVAL_DATA_PATH')
        test_dataset = MyTestDataset(data_path, args)
    
    # 获取特征统计信息和类型（如果未提供）
    if feat_statistics is None or feat_types is None:
        feat_statistics = test_dataset.feat_statistics
        feat_types = test_dataset.feature_types
    
    # 定义rqvae_config
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    # 确定实际使用的特征ID
    actual_feat_id = args.mm_emb_id[0] if args.mm_emb_id else '82'
    input_dim = SHAPE_DICT.get(actual_feat_id, 1024)
    print(f"Using feature ID: {actual_feat_id}, input_dim: {input_dim}")
    
    rqvae_config = {
        'input_dim': input_dim,  # 根据多模态嵌入维度调整
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
    
    print(f"rqvae_config: {rqvae_config}")
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=test_dataset.collate_fn
    )
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    
    # 初始化模型 - 使用RQVAEEnhancedModel
    print("Initializing model")
    model = RQVAEEnhancedModel(usernum, itemnum, feat_statistics, feat_types, args, rqvae_config).to(args.device)
    
    # 处理model_path，如果它是一个目录，则在其中查找.pt文件
    if model_path is not None and os.path.isdir(model_path):
        print(f"model_path is a directory, searching for .pt file in {model_path}")
        for item in os.listdir(model_path):
            if item.endswith(".pt"):
                model_path = os.path.join(model_path, item)
                print(f"Found model file: {model_path}")
                break
        else:
            # 如果没有找到.pt文件，使用get_ckpt_path函数
            print("No .pt file found in directory, using get_ckpt_path function")
            model_path = get_ckpt_path()
    
    print(f"Loading model from {model_path}")
    
    # 加载模型状态字典
    state_dict = torch.load(model_path, map_location=args.device)
    
    # 获取模型的当前状态字典
    model_state_dict = model.state_dict()
    
    # 创建一个新的状态字典，只包含模型中存在的键
    new_state_dict = {}
    unexpected_keys = []
    mismatched_keys = []
    missing_keys = []
    
    # 检查模型中缺失的键
    for key in model_state_dict.keys():
        if key not in state_dict:
            missing_keys.append(key)
    
    # 如果itemdnn.weight在缺失的键中，但state_dict中有类似名称的键，尝试进行映射
    if 'itemdnn.weight' in missing_keys:
        # 查找可能的替代键
        for key in state_dict.keys():
            if 'itemdnn' in key and 'weight' in key:
                # 尝试使用这个键来替代缺失的itemdnn.weight
                if state_dict[key].shape == model_state_dict['itemdnn.weight'].shape:
                    new_state_dict['itemdnn.weight'] = state_dict[key]
                    missing_keys.remove('itemdnn.weight')
                    print(f"Mapped {key} to itemdnn.weight")
                    break
    
    for key, value in state_dict.items():
        if key in model_state_dict:
            # 检查维度是否匹配
            if model_state_dict[key].shape == value.shape:
                new_state_dict[key] = value
            else:
                print(f"Skipping {key} due to shape mismatch. Expected {model_state_dict[key].shape}, got {value.shape}")
                mismatched_keys.append(key)
        else:
            unexpected_keys.append(key)
    
    # 对于缺失的键，如果它们在state_dict中有类似名称的键，尝试进行映射
    for key in missing_keys[:]:  # 使用切片复制列表，避免在迭代时修改列表
        if key not in new_state_dict:
            # 查找可能的替代键
            for state_key in state_dict.keys():
                if key.split('.')[-1] in state_key:
                    # 检查维度是否匹配
                    if model_state_dict[key].shape == state_dict[state_key].shape:
                        new_state_dict[key] = state_dict[state_key]
                        missing_keys.remove(key)
                        print(f"Mapped {state_key} to {key}")
                        break
    
    # 打印意外的键
    if unexpected_keys:
        print(f"Unexpected keys in state_dict: {unexpected_keys}")
    
    # 打印不匹配的键
    if mismatched_keys:
        print(f"Mismatched keys in state_dict: {mismatched_keys}")
    
    # 打印缺失的键
    if missing_keys:
        print(f"Missing keys in state_dict: {missing_keys}")
    
    # 使用非严格模式加载修改后的状态字典
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print("Model loaded and set to eval mode")

    all_embs = []
    user_list = []
    print("Starting inference loop")
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        seq, token_type, seq_feat, user_id = batch
        seq = seq.to(args.device)
        
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
        
        # 使用RQVAEEnhancedModel的predict_with_rqvae方法
        final_feat, semantic_embeddings = model.predict_with_rqvae(seq, seq_feat, token_type, multimodal_emb)
        
        # 使用最终特征
        logits = final_feat
        
        for i in range(logits.shape[0]):
            emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
            all_embs.append(emb)
        user_list += user_id
    print("Inference loop completed")

    # 生成候选库的embedding 以及 id文件
    print("Generating candidate embeddings")
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
    )
    print("Candidate embeddings generated")
    
    all_embs = np.concatenate(all_embs, axis=0)
    # 保存query文件
    save_emb(all_embs, Path(save_dir, 'query.fbin'))
    # ANN 检索
    ann_cmd = (
        str(Path("/workspace", "faiss-based-ann", "faiss_demo"))
        + " --dataset_vector_file_path="
        + str(Path(save_dir, "embedding.fbin"))
        + " --dataset_id_file_path="
        + str(Path(save_dir, "id.u64bin"))
        + " --query_vector_file_path="
        + str(Path(save_dir, "query.fbin"))
        + " --result_id_file_path="
        + str(Path(save_dir, "id100.u64bin"))
        + " --query_ann_top_k=10 --faiss_M=64 --faiss_ef_construction=1280 --query_ef_search=640 --faiss_metric_type=0"
    )
    print(f"Running ANN command: {ann_cmd}")
    os.system(ann_cmd)

    # 取出top-k
    print("Reading result IDs")
    top10s_retrieved = read_result_ids(Path(save_dir, "id100.u64bin"))
    top10s_untrimmed = []
    for top10 in tqdm(top10s_retrieved):
        for item in top10:
            top10s_untrimmed.append(retrieve_id2creative_id.get(int(item), 0))

    top10s = [top10s_untrimmed[i : i + 10] for i in range(0, len(top10s_untrimmed), 10)]
    print("Infer function completed")
    return top10s, user_list


# Add this at the end of the file, after the infer function

if __name__ == "__main__":
    # Get arguments
    args = get_args()
    
    # Set up paths
    data_path = os.environ.get('EVAL_DATA_PATH')
    model_path = os.environ.get('MODEL_OUTPUT_PATH')
    save_dir = os.environ.get('EVAL_RESULT_PATH')
    
    # Create test dataset
    test_dataset = MyTestDataset(data_path, args)
    
    # Get feature statistics and types
    feat_statistics = test_dataset.feat_statistics
    feat_types = test_dataset.feature_types
    
    # Call infer function with all required arguments
    top10s, user_list = infer(args, test_dataset, model_path, save_dir, feat_statistics, feat_types)
    
    # Optionally save results
    # with open(Path(save_dir, "results.json"), "w") as f:
    #     json.dump({"top10s": top10s, "user_list": user_list}, f)
