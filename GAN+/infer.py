import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset, save_emb
from model import BaselineModel, NaryEncoder
from model_rqvae import GANQuantizerWrapper


def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")

    print(f"[DEBUG] MODEL_OUTPUT_PATH = {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise ValueError(f"MODEL_OUTPUT_PATH does not exist: {ckpt_path}")

    items = os.listdir(ckpt_path)
    print(f"[DEBUG] Contents of MODEL_OUTPUT_PATH: {items}")

    ckpt_dirs = [item for item in items if item.startswith("global_step") and os.path.isdir(os.path.join(ckpt_path, item))]
    if not ckpt_dirs:
        print("[WARNING] No global_step* directories found. Trying fallback...")
        fallback_path = os.path.join(ckpt_path, "model.pt")
        if os.path.exists(fallback_path):
            print(f"[DEBUG] Using fallback model: {fallback_path}")
            return fallback_path
        else:
            raise ValueError("No checkpoint directories found and no fallback model.pt available.")

    ckpt_dirs.sort(key=lambda x: int(x.split('.')[0].replace('global_step', '')))
    latest_ckpt_dir = ckpt_dirs[-1]
    model_path = os.path.join(ckpt_path, latest_ckpt_dir, "model.pt")
    if not os.path.exists(model_path):
        raise ValueError(f"Model checkpoint not found: {model_path}")

    print(f"[DEBUG] Using latest checkpoint: {model_path}")
    return model_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
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
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])
    args = parser.parse_args()
    return args


def read_result_ids(file_path):
    with open(file_path, 'rb') as f:
        num_points_query = struct.unpack('I', f.read(4))[0]
        query_ann_top_k = struct.unpack('I', f.read(4))[0]
        print(f"num_points_query: {num_points_query}, query_ann_top_k: {query_ann_top_k}")
        num_result_ids = num_points_query * query_ann_top_k
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)
        return result_ids.reshape((num_points_query, query_ann_top_k))


def process_cold_start_feat(feat):
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if type(feat_value) == list:
            value_list = []
            for v in feat_value:
                value_list.append(0 if type(v) == str else v)
            processed_feat[feat_id] = value_list
        elif type(feat_value) == str:
            processed_feat[feat_id] = 0
        else:
            processed_feat[feat_id] = feat_value
    return processed_feat


def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model):
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}

    print(f'[DEBUG] Loading candidate items from: {candidate_path}')
    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
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
                    feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    print(f'[DEBUG] Loaded {len(item_ids)} candidate items')
    model.save_item_emb(item_ids, retrieval_ids, features, os.environ.get('EVAL_RESULT_PATH'))
    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)
    return retrieve_id2creative_id


def infer():
    args = get_args()
    data_path = os.environ.get('EVAL_DATA_PATH')
    test_dataset = MyTestDataset(data_path, args)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=test_dataset.collate_fn
    )
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types

    print('[DEBUG] Initializing GANQuantizerWrapper for inference...')
    gan_quantizer = None
    semantic_emb = None
    nary_encoder = None

    emb_81_path = Path(data_path, "creative_emb", "emb_81")
    print(f'[DEBUG] Checking emb_81 path: {emb_81_path}')
    if emb_81_path.exists():
        try:
            from model_rqvae import MmEmbDataset
            mm_dataset = MmEmbDataset(data_path, '81')
            if len(mm_dataset) > 0:
                mm_emb_dim = 32
                num_classes = 256
                gan_quantizer = GANQuantizerWrapper(
                    input_dim=mm_emb_dim,
                    hidden_channels=[128, 64],
                    latent_dim=32,
                    num_classes=num_classes,
                    device=args.device
                )
                semantic_emb = nn.Embedding(num_classes, args.hidden_units).to(args.device)
                nary_bases = [2, 4, 8]
                nary_encoder = NaryEncoder(nary_bases, args.hidden_units).to(args.device)
                print(f'[DEBUG] Created quantizer for emb_81: dim={mm_emb_dim}, classes={num_classes}')
            else:
                print('[WARNING] mm_dataset is empty')
        except Exception as e:
            print(f'[WARNING] Failed to create GAN quantizer: {e}')
    else:
        print('[WARNING] emb_81 path does not exist, skipping GAN quantizer')

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args,
                         gan_quantizer=gan_quantizer, semantic_emb=semantic_emb, nary_encoder=nary_encoder).to(args.device)
    model.eval()

    ckpt_path = get_ckpt_path()
    print(f'[DEBUG] Loading model from: {ckpt_path}')
    state_dict = torch.load(ckpt_path, map_location=torch.device(args.device))
    model.load_state_dict(state_dict, strict=False)

    ckpt_dir = os.path.dirname(ckpt_path)
    gan_ckpt_path = os.path.join(ckpt_dir, "gan_quantizer.pt")
    if gan_quantizer is not None and os.path.exists(gan_ckpt_path):
        print(f'[DEBUG] Loading GAN quantizer from: {gan_ckpt_path}')
        gan_quantizer.load_state_dict(torch.load(gan_ckpt_path, map_location=torch.device(args.device)), strict=False)

    all_embs = []
    user_list = []
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        seq, token_type, seq_feat, user_id = batch
        seq = seq.to(args.device)
        logits = model.predict(seq, seq_feat, token_type)
        for i in range(logits.shape[0]):
            emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
            all_embs.append(emb)
        user_list += user_id

    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
    )
    all_embs = np.concatenate(all_embs, axis=0)
    save_emb(all_embs, Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin'))

    ann_cmd = (
        str(Path("/workspace", "faiss-based-ann", "faiss_demo"))
        + " --dataset_vector_file_path="
        + str(Path(os.environ.get("EVAL_RESULT_PATH"), "embedding.fbin"))
        + " --dataset_id_file_path="
        + str(Path(os.environ.get("EVAL_RESULT_PATH"), "id.u64bin"))
        + " --query_vector_file_path="
        + str(Path(os.environ.get('EVAL_RESULT_PATH'), "query.fbin"))
        + " --result_id_file_path="
        + str(Path(os.environ.get('EVAL_RESULT_PATH'), "id100.u64bin"))
        + " --query_ann_top_k=10 --faiss_M=64 --faiss_ef_construction=1280 --query_ef_search=640 --faiss_metric_type=0"
    )
    os.system(ann_cmd)

    top10s_retrieved = read_result_ids(Path(os.environ.get('EVAL_RESULT_PATH'), "id100.u64bin"))
    top10s_untrimmed = []
    for top10 in tqdm(top10s_retrieved):
        for item in top10:
            top10s_untrimmed.append(retrieve_id2creative_id.get(int(item), 0))

    top10s = [top10s_untrimmed[i:i + 10] for i in range(0, len(top10s_untrimmed), 10)]
    return top10s, user_list


if __name__ == "__main__":
    top10s, user_list = infer()
    print(f"[INFO] Inference completed. Total users: {len(user_list)}")