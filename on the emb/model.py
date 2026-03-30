from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import save_emb


class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 使用内置的Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask.unsqueeze(1)
            )
        else:
            # 降级到标准注意力机制
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)

        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)

        # 最终的线性变换
        output = self.out_linear(attn_output)

        return output, None


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs


class CrossNetwork(torch.nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.cross_layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.cross_layers.append(torch.nn.Linear(input_dim, input_dim))

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x0 = x
        for layer in self.cross_layers:
            x = x0 * layer(x) + x
        return x


class SFGModel(torch.nn.Module):
    def __init__(self, user_num, item_num, feat_statistics, feat_types, args, use_modal_feat=False):
        super(SFGModel, self).__init__()
        self.args = args
        self.user_num = user_num
        self.item_num = item_num
        self.feat_statistics = feat_statistics
        self.feat_types = feat_types
        self.use_modal_feat = use_modal_feat
        # ---------- Added core configs and aliases ----------
        self.dev = torch.device(args.device)
        self.hidden_units = args.hidden_units
        self.num_embedding_sets = getattr(args, 'num_embedding_sets', 1)
        self.norm_first = getattr(args, 'norm_first', False)
        # Alias lists to match downstream code
        self.USER_SPARSE_FEAT = self.feat_types.get('user_sparse', [])
        self.ITEM_SPARSE_FEAT = self.feat_types.get('item_sparse', [])
        self.USER_ARRAY_FEAT = self.feat_types.get('user_array', [])
        self.ITEM_ARRAY_FEAT = self.feat_types.get('item_array', [])
        self.USER_CONTINUAL_FEAT = self.feat_types.get('user_continual', [])
        self.ITEM_CONTINUAL_FEAT = self.feat_types.get('item_continual', [])
        self.ITEM_EMB_FEAT = self.feat_types.get('item_emb', [])
        # Known multimodal embedding dims (keep consistent with dataset.load_mm_emb)
        _EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT_DIMS = {
            fid: _EMB_SHAPE_DICT[str(fid)] if str(fid) in _EMB_SHAPE_DICT else self.feat_statistics.get(fid, self.hidden_units)
            for fid in self.ITEM_EMB_FEAT
        }
        # Unified sparse embeddings (include array features for pooling)
        self.sparse_emb = torch.nn.ModuleDict()
        for feat_id in self.USER_SPARSE_FEAT + self.ITEM_SPARSE_FEAT:
            self.sparse_emb[feat_id] = torch.nn.Embedding(self.feat_statistics[feat_id], self.hidden_units, padding_idx=0)
        for feat_id in self.USER_ARRAY_FEAT + self.ITEM_ARRAY_FEAT:
            if feat_id in self.feat_statistics:
                self.sparse_emb[feat_id] = torch.nn.Embedding(self.feat_statistics[feat_id], self.hidden_units, padding_idx=0)
        # Multi-embedding sets for user/item ids
        self.user_emb = torch.nn.ModuleList([
            torch.nn.Embedding(self.user_num + 1, self.hidden_units, padding_idx=0)
            for _ in range(self.num_embedding_sets)
        ])
        self.item_emb = torch.nn.ModuleList([
            torch.nn.Embedding(self.item_num + 1, self.hidden_units, padding_idx=0)
            for _ in range(self.num_embedding_sets)
        ])
        # Positional embedding & dropout
        self.pos_emb = torch.nn.Embedding(args.maxlen + 2, self.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        # Attention stacks per embedding set (used in log2feats/feat2emb)
        self.attention_layers = torch.nn.ModuleList()
        self.attention_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        for _ in range(self.num_embedding_sets):
            attn_list = torch.nn.ModuleList()
            attn_ln_list = torch.nn.ModuleList()
            ff_list = torch.nn.ModuleList()
            ff_ln_list = torch.nn.ModuleList()
            for _b in range(args.num_blocks):
                attn_list.append(FlashMultiHeadAttention(self.hidden_units, args.num_heads, args.dropout_rate))
                attn_ln_list.append(torch.nn.LayerNorm(self.hidden_units))
                ff_list.append(PointWiseFeedForward(self.hidden_units, args.dropout_rate))
                ff_ln_list.append(torch.nn.LayerNorm(self.hidden_units))
            self.attention_layers.append(attn_list)
            self.attention_layernorms.append(attn_ln_list)
            self.forward_layers.append(ff_list)
            self.forward_layernorms.append(ff_ln_list)
        self.last_layernorm = torch.nn.LayerNorm(self.hidden_units)
        # ---------- End of added configs ----------
        
        # Feature embedding layers (keep placeholders if needed)
        self.user_sparse_embeddings = torch.nn.ModuleDict()
        for feat_id in self.feat_types.get('user_sparse', []):
            self.user_sparse_embeddings[feat_id] = torch.nn.Embedding(self.feat_statistics[feat_id], args.hidden_units, padding_idx=0)

        self.item_sparse_embeddings = torch.nn.ModuleDict()
        for feat_id in self.feat_types.get('item_sparse', []):
            self.item_sparse_embeddings[feat_id] = torch.nn.Embedding(self.feat_statistics[feat_id], args.hidden_units, padding_idx=0)

        # Placeholder for array and continual features processing
        self.user_array_processors = torch.nn.ModuleDict()
        for feat_id in self.feat_types.get('user_array', []):
            self.user_array_processors[feat_id] = torch.nn.Linear(self.feat_statistics[feat_id], args.hidden_units)  # Placeholder

        self.item_array_processors = torch.nn.ModuleDict()
        for feat_id in self.feat_types.get('item_array', []):
            self.item_array_processors[feat_id] = torch.nn.Linear(self.feat_statistics[feat_id], args.hidden_units)  # Placeholder

        self.user_continual_processors = torch.nn.ModuleDict()
        for feat_id in self.feat_types.get('user_continual', []):
            self.user_continual_processors[feat_id] = torch.nn.Linear(1, args.hidden_units)  # Placeholder

        self.item_continual_processors = torch.nn.ModuleDict()
        for feat_id in self.feat_types.get('item_continual', []):
            self.item_continual_processors[feat_id] = torch.nn.Linear(1, args.hidden_units)  # Placeholder

        # Multimodal embedding features (item_emb)
        self.item_emb_processors = torch.nn.ModuleDict()
        for feat_id in self.ITEM_EMB_FEAT:
            in_dim = self.ITEM_EMB_FEAT_DIMS[feat_id]
            self.item_emb_processors[feat_id] = torch.nn.Linear(in_dim, args.hidden_units)
        # expose as emb_transform for downstream code
        self.emb_transform = self.item_emb_processors

        # DNNs for combining features (match concatenation dims)
        userdim = (
            args.hidden_units * (1 + len(self.USER_SPARSE_FEAT) + len(self.USER_ARRAY_FEAT) + len(self.USER_CONTINUAL_FEAT))
        )
        itemdim = (
            args.hidden_units * (1 + len(self.ITEM_SPARSE_FEAT) + len(self.ITEM_ARRAY_FEAT) + len(self.ITEM_CONTINUAL_FEAT) + len(self.ITEM_EMB_FEAT))
        )
        self.userdnn = torch.nn.Linear(userdim, args.hidden_units)
        self.itemdnn = torch.nn.Linear(itemdim, args.hidden_units)

        # CrossNetwork per embedding set
        self.user_cross_nets = torch.nn.ModuleList([
            CrossNetwork(args.hidden_units, args.cross_layers) for _ in range(self.num_embedding_sets)
        ])
        self.item_cross_nets = torch.nn.ModuleList([
            CrossNetwork(args.hidden_units, args.cross_layers) for _ in range(self.num_embedding_sets)
        ])

        # Simple prediction head (optional)
        self.prediction_head = torch.nn.Linear(args.hidden_units, args.hidden_units)
        
        # Disable feature generator and reconstruction by default (avoid undefined Generator)
        self.feature_generator = None
        self.reconstruction_head = None
        # 缓存最近一次 batch size，用于当 seq_feature 为空时推断 B
        self._last_batch_size = None

    def safe_embedding(self, emb_module, indices):
        """
        对所有embedding查表进行统一边界保护：
        - 将索引裁剪到 [0, emb.num_embeddings - 1]
        - 强制为 long 并移动到正确设备
        """
        max_idx = emb_module.num_embeddings - 1
        safe_idx = torch.clamp(indices, min=0, max=max_idx).to(self.dev)
        if safe_idx.dtype != torch.long:
            safe_idx = safe_idx.long()
        return emb_module(safe_idx)
    def feat2tensor(self, seq_feature, feat_id, expected_len):
        """
        将collate后的特征统一转换为模型所需张量。
        支持两种输入布局：
        1) 训练/堆叠后的 "dict of tensors"：{feat_id: torch.Tensor[B, L, ...]}
        2) 推理/未堆叠的 "list/ndarray of dicts"：长度为B，每个元素是长度为L的dict序列
        """
        # Case A: 训练路径，已堆叠成字典 -> 直接读取
        if isinstance(seq_feature, dict):
            # 取一个样本来推断batch size；若为空则回退到最近一次保存的 batch size
            if len(seq_feature) == 0:
                B = getattr(self, "_last_batch_size", None)
                if B is None:
                    raise ValueError("seq_feature is empty and _last_batch_size is unknown; please set batch size context before calling feat2tensor")
            else:
                any_tensor = next(iter(seq_feature.values()))
                B = any_tensor.size(0)

            t = seq_feature.get(feat_id, None)

            # 缺失时按类型补零
            if t is None:
                if feat_id in self.ITEM_EMB_FEAT:
                    D = int(self.ITEM_EMB_FEAT_DIMS.get(feat_id, self.hidden_units))
                    return torch.zeros(B, expected_len, D, device=self.dev, dtype=torch.float32)
                elif (feat_id in self.ITEM_ARRAY_FEAT) or (feat_id in self.USER_ARRAY_FEAT):
                    return torch.zeros(B, expected_len, 1, device=self.dev, dtype=torch.long)
                elif (feat_id in self.ITEM_CONTINUAL_FEAT) or (feat_id in self.USER_CONTINUAL_FEAT):
                    return torch.zeros(B, expected_len, device=self.dev, dtype=torch.float32)
                else:
                    return torch.zeros(B, expected_len, device=self.dev, dtype=torch.long)

            # 确保在目标设备
            t = t.to(self.dev)

            # 统一维度：加入缺失的时间维
            if t.dim() == 1:
                t = t.unsqueeze(1).expand(-1, expected_len)
            elif t.dim() == 2:
                if feat_id in self.ITEM_EMB_FEAT:
                    if t.size(1) != expected_len:
                        t = t.unsqueeze(1).expand(-1, expected_len, -1)
                else:
                    pass
            elif t.dim() >= 3:
                pass

            # 对齐时间长度到 expected_len（右对齐）
            def pad_or_slice_time(x, L):
                if x.dim() == 2:
                    T = x.size(1)
                    if T == L:
                        return x
                    elif T < L:
                        pad = torch.zeros(x.size(0), L - T, device=x.device, dtype=x.dtype)
                        return torch.cat([pad, x], dim=1)
                    else:
                        return x[:, -L:]
                else:
                    T = x.size(1)
                    if T == L:
                        return x
                    elif T < L:
                        pad_shape = (x.size(0), L - T) + tuple(x.shape[2:])
                        pad = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
                        return torch.cat([pad, x], dim=1)
                    else:
                        slices = [slice(None), slice(-L, None)] + [slice(None)] * (x.dim() - 2)
                        return x[tuple(slices)]

            t = pad_or_slice_time(t, expected_len)

            # 类型与数值安全处理
            if feat_id in self.ITEM_EMB_FEAT:
                t = t.float()
                if torch.isnan(t).any() or torch.isinf(t).any():
                    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            elif (feat_id in self.ITEM_CONTINUAL_FEAT) or (feat_id in self.USER_CONTINUAL_FEAT):
                if t.dim() == 3 and t.size(-1) == 1:
                    t = t.squeeze(-1)
                t = t.float()
                if torch.isnan(t).any() or torch.isinf(t).any():
                    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                # 稀疏/数组特征，作为索引
                t = t.long()
                vocab_size = self.feat_statistics.get(feat_id, None)
                if vocab_size is not None:
                    t = torch.clamp(t, min=0, max=int(vocab_size) - 1)
                else:
                    t = torch.clamp(t, min=0)

                return t

        # Case B: 推理路径，未堆叠的 list/ndarray of dicts
        else:
            # 将 ndarray 转为 list
            if isinstance(seq_feature, np.ndarray):
                seq_feature = seq_feature.tolist()
            B = len(seq_feature)
            if B == 0:
                B = getattr(self, "_last_batch_size", None) or 0
            L = expected_len
            dev = self.dev
            # 判定特征类型
            is_emb = (feat_id in self.ITEM_EMB_FEAT)
            is_array = (feat_id in self.ITEM_ARRAY_FEAT) or (feat_id in self.USER_ARRAY_FEAT)
            is_cont = (feat_id in self.ITEM_CONTINUAL_FEAT) or (feat_id in self.USER_CONTINUAL_FEAT)
            if is_emb:
                D = int(self.ITEM_EMB_FEAT_DIMS.get(feat_id, self.hidden_units))
                out = torch.zeros(B, L, D, device=dev, dtype=torch.float32)
            elif is_array:
                max_arr_len = 1
                for b in range(B):
                    seq_list = seq_feature[b] if b < len(seq_feature) and isinstance(seq_feature[b], (list, tuple)) else []
                    for fd in seq_list:
                        if isinstance(fd, dict):
                            v = fd.get(feat_id, None)
                            if isinstance(v, np.ndarray):
                                v = v.tolist()
                            if isinstance(v, (list, tuple)):
                                if len(v) > max_arr_len:
                                    max_arr_len = len(v)
                out = torch.zeros(B, L, max_arr_len, device=dev, dtype=torch.long)
            elif is_cont:
                out = torch.zeros(B, L, device=dev, dtype=torch.float32)
            else:
                out = torch.zeros(B, L, device=dev, dtype=torch.long)
            for b in range(B):
                seq_list = seq_feature[b] if b < len(seq_feature) and isinstance(seq_feature[b], (list, tuple)) else []
                T = len(seq_list)
                pad = max(0, L - T)
                for t in range(L):
                    if t < pad or (t - pad) >= T:
                        continue
                    fd = seq_list[t - pad]
                    if not isinstance(fd, dict):
                        continue
                    if feat_id not in fd:
                        continue
                    v = fd[feat_id]
                    if is_emb:
                        vt = torch.as_tensor(v, dtype=torch.float32, device=dev)
                        if vt.numel() == out.size(-1):
                            out[b, t] = vt
                        else:
                            # 自动截断或补零
                            sz = min(vt.numel(), out.size(-1))
                            if sz > 0:
                                out[b, t, :sz] = vt.view(-1)[:sz]
                    elif is_array:
                        if isinstance(v, np.ndarray):
                            v = v.tolist()
                        if not isinstance(v, (list, tuple)):
                            v = [int(v) if v is not None else 0]
                        limit = min(len(v), out.size(-1))
                        for s in range(limit):
                            try:
                                out[b, t, s] = int(v[s])
                            except Exception:
                                out[b, t, s] = 0
                    elif is_cont:
                        try:
                            out[b, t] = float(v)
                        except Exception:
                            out[b, t] = 0.0
                    else:
                        try:
                            out[b, t] = int(v)
                        except Exception:
                            out[b, t] = 0
            # 数值/索引安全
            if is_emb:
                out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            elif is_cont:
                out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                vocab_size = self.feat_statistics.get(feat_id, None)
                if vocab_size is not None:
                    out = torch.clamp(out, min=0, max=int(vocab_size) - 1)
                else:
                    out = torch.clamp(out, min=0)
            return out

    def log2feats(self, log_seqs, mask, seq_feature):
        """
        Args:
            log_seqs: 序列ID
            mask: token类型掩码，1表示item token，2表示user token
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典

        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        # 记录 batch size 以便 feat2tensor 在 seq_feature 为空时回退使用
        self._last_batch_size = batch_size
        
        # 为每套embedding生成对应的embedding
        item_embeddings = []
        user_embeddings = []
        
        for i in range(self.num_embedding_sets):
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            # 安全索引：mask处保留原ID，其它置0，并对各自范围钳位
            safe_user_ids = torch.where(user_mask.bool(), torch.clamp(log_seqs, min=0, max=self.user_num), torch.zeros_like(log_seqs))
            safe_item_ids = torch.where(item_mask.bool(), torch.clamp(log_seqs, min=0, max=self.item_num), torch.zeros_like(log_seqs))
            user_embedding = self.safe_embedding(self.user_emb[i], safe_user_ids)
            item_embedding = self.safe_embedding(self.item_emb[i], safe_item_ids)
            item_embeddings.append(item_embedding)
            user_embeddings.append(user_embedding)
        
        # 处理其他特征...
        item_feat_lists = []
        user_feat_lists = []
        
        for i in range(self.num_embedding_sets):
            item_feat_list = [item_embeddings[i]]
            user_feat_list = [user_embeddings[i]]
            
            # 处理稀疏特征
            for k in self.ITEM_SPARSE_FEAT:
                item_feat = self.feat2tensor(seq_feature, k, expected_len=maxlen)
                item_feat_emb = self.safe_embedding(self.sparse_emb[k], item_feat)
                item_feat_list.append(item_feat_emb)
            
            for k in self.USER_SPARSE_FEAT:
                user_feat = self.feat2tensor(seq_feature, k, expected_len=maxlen)
                user_feat_emb = self.safe_embedding(self.sparse_emb[k], user_feat)
                user_feat_list.append(user_feat_emb)
            
            # 处理数组特征
            for k in self.ITEM_ARRAY_FEAT:
                item_feat = self.feat2tensor(seq_feature, k, expected_len=maxlen)
                item_feat_emb = self.safe_embedding(self.sparse_emb[k], item_feat)
                # 对数组特征的embedding进行池化
                item_feat_emb = torch.mean(item_feat_emb, dim=2)
                item_feat_list.append(item_feat_emb)
            
            for k in self.USER_ARRAY_FEAT:
                user_feat = self.feat2tensor(seq_feature, k, expected_len=maxlen)
                user_feat_emb = self.safe_embedding(self.sparse_emb[k], user_feat)
                # 对数组特征的embedding进行池化
                user_feat_emb = torch.mean(user_feat_emb, dim=2)
                user_feat_list.append(user_feat_emb)
            
            # 处理连续特征
            for k in self.ITEM_CONTINUAL_FEAT:
                item_feat = self.feat2tensor(seq_feature, k, expected_len=maxlen)
                # 连续特征需要扩展维度以匹配embedding维度
                item_feat = item_feat.unsqueeze(-1).expand(-1, -1, self.item_emb[0].embedding_dim)
                item_feat_list.append(item_feat)
            
            for k in self.USER_CONTINUAL_FEAT:
                user_feat = self.feat2tensor(seq_feature, k, expected_len=maxlen)
                # 连续特征需要扩展维度以匹配embedding维度
                user_feat = user_feat.unsqueeze(-1).expand(-1, -1, self.user_emb[0].embedding_dim)
                user_feat_list.append(user_feat)
            
            # 处理多模态特征
            for k in self.ITEM_EMB_FEAT:
                item_feat = self.feat2tensor(seq_feature, k, expected_len=maxlen)
                # 多模态特征需要经过线性变换
                item_feat_emb = self.emb_transform[k](item_feat.float())
                item_feat_list.append(item_feat_emb)
            
            item_feat_lists.append(item_feat_list)
            user_feat_lists.append(user_feat_list)
        
        # 为每套embedding应用对应的Cross Network
        item_emb_sets = []
        user_emb_sets = []
        for i in range(self.num_embedding_sets):
                    # merge features
                    filtered_item_feats = [t for t in item_feat_lists[i] if t.size(1) > 0 and t.size(2) > 0]
                    if len(filtered_item_feats) == 0:
                        # 如果所有张量都为空，则创建一个空的张量，维度与预期一致
                        batch_size = log_seqs.size(0)
                        maxlen = log_seqs.size(1)
                        all_item_emb = torch.zeros(batch_size, maxlen, 0, device=log_seqs.device)
                    else:

                        # 调试：打印各部分维度与拼接后的目标维度
                        try:
                            dims = [t.size(2) for t in filtered_item_feats]
                            print(f"[DEBUG] log2feats: set={i}, n_parts={len(filtered_item_feats)}, part_dims={dims}, concat_dim={sum(dims)}")
                        except Exception as e:
                            print(f"[DEBUG] log2feats: shape debug failed: {e}")

                        all_item_emb = torch.cat(filtered_item_feats, dim=2)

                        

                    
                    all_item_emb = torch.relu(self.itemdnn(all_item_emb))
                    
                    # Reshape for CrossNetwork
                    batch_size, maxlen, hidden_units = all_item_emb.shape
                    all_item_emb_reshaped = all_item_emb.view(-1, hidden_units) # (batch_size * maxlen, hidden_units)
                    
                    # 应用对应的Cross Network
                    all_item_emb_processed = self.item_cross_nets[i](all_item_emb_reshaped)
                    
                    # Reshape back
                    all_item_emb = all_item_emb_processed.view(batch_size, maxlen, hidden_units)
                    item_emb_sets.append(all_item_emb)
                    
                    all_user_emb = torch.cat(user_feat_lists[i], dim=2)
                    all_user_emb = torch.relu(self.userdnn(all_user_emb))
                    
                    # Reshape for CrossNetwork
                    batch_size, maxlen, hidden_units = all_user_emb.shape
                    all_user_emb_reshaped = all_user_emb.view(-1, hidden_units) # (batch_size * maxlen, hidden_units)
                    
                    # 应用对应的Cross Network
                    all_user_emb_processed = self.user_cross_nets[i](all_user_emb_reshaped)
                    
                    # Reshape back
                    all_user_emb = all_user_emb_processed.view(batch_size, maxlen, hidden_units)
                    user_emb_sets.append(all_user_emb)
                
        # 为每套embedding应用独立的Transformer层
        seq_emb_sets = []
        for i in range(self.num_embedding_sets):
            seqs = item_emb_sets[i] + user_emb_sets[i]
            seqs_emb = seqs * (self.item_emb[0].embedding_dim**0.5)  # 使用第一套embedding的维度
            poss = torch.arange(1, seqs_emb.shape[1] + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
            mask_2d = (seqs_emb.abs().sum(dim=-1) != 0)
            poss = poss * mask_2d.long()
            # 防止位置索引越界（当实际 L > args.maxlen 时），将索引裁剪到 pos_emb 的合法范围内
            poss = torch.clamp(poss, min=0, max=self.pos_emb.num_embeddings - 1)
            seqs_emb += self.pos_emb(poss)
            seqs_emb = self.emb_dropout(seqs_emb)

            # 支持自定义特征生成器插拔
            if hasattr(self, 'feature_generator') and self.feature_generator is not None:
                seqs_emb = self.feature_generator(seqs_emb)

            maxlen_emb = seqs_emb.shape[1]
            ones_matrix = torch.ones((maxlen_emb, maxlen_emb), dtype=torch.bool, device=self.dev)
            attention_mask_tril = torch.tril(ones_matrix)
            attention_mask_pad = mask_2d.to(self.dev)
            attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

            # 使用第i套embedding对应的Transformer层
            for j in range(len(self.attention_layers[i])):
                if self.norm_first:
                    x = self.attention_layernorms[i][j](seqs_emb)
                    mha_outputs, _ = self.attention_layers[i][j](x, x, x, attn_mask=attention_mask)
                    seqs_emb = seqs_emb + mha_outputs
                    seqs_emb = seqs_emb + self.forward_layers[i][j](self.forward_layernorms[i][j](seqs_emb))
                else:
                    mha_outputs, _ = self.attention_layers[i][j](seqs_emb, seqs_emb, seqs_emb, attn_mask=attention_mask)
                    seqs_emb = self.attention_layernorms[i][j](seqs_emb + mha_outputs)
                    seqs_emb = self.forward_layernorms[i][j](seqs_emb + self.forward_layers[i][j](seqs_emb))

            log_feats = self.last_layernorm(seqs_emb)
            seq_emb_sets.append(log_feats)
        
        # 聚合多套embedding输出
        if hasattr(self, 'aggregation_type') and self.aggregation_type == 'weighted':
            # 加权平均聚合
            weights = getattr(self, 'embedding_weights', None)
            if weights is None:
                weights = torch.ones(len(seq_emb_sets), device=seq_emb_sets[0].device) / len(seq_emb_sets)
            weights = weights.view(-1, 1, 1)
            final_seq_emb = (torch.stack(seq_emb_sets, dim=0) * weights).sum(dim=0)
        elif hasattr(self, 'aggregation_type') and self.aggregation_type == 'adaptive':
            # 自适应融合（可插拔模块，示例：MLP融合）
            fusion_input = torch.stack(seq_emb_sets, dim=0).permute(1, 2, 0, 3).reshape(seq_emb_sets[0].shape[0], seq_emb_sets[0].shape[1], -1)
            if hasattr(self, 'fusion_mlp'):
                final_seq_emb = self.fusion_mlp(fusion_input)
            else:
                final_seq_emb = fusion_input.mean(dim=-1)
        else:
            # 默认平均
            final_seq_emb = torch.stack(seq_emb_sets, dim=0).mean(dim=0)
        return final_seq_emb

    def feat2emb(self, seqs, seq_feature, include_user=True):
        """
        将序列和特征转换为embedding
        
        Args:
            seqs: 序列ID
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            include_user: 是否包含user embedding
        
        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """
        # 规范化 seqs：确保为 [batch_size, maxlen] 的二维 LongTensor 且在正确设备
        if not torch.is_tensor(seqs):
            seqs = torch.tensor(seqs, device=self.dev)
        if seqs.dim() == 1:
            seqs = seqs.unsqueeze(0)
        elif seqs.dim() == 0:
            seqs = seqs.view(1, 1)
        seqs = seqs.to(self.dev)
        if seqs.dtype != torch.long:
            seqs = seqs.long()
        batch_size = seqs.shape[0]
        maxlen = seqs.shape[1]
        # 记录 batch size 以便 feat2tensor 在 seq_feature 为空时回退使用
        self._last_batch_size = batch_size
        
        # 为每套embedding生成对应的embedding
        item_embeddings = []
        user_embeddings = []
        
        for i in range(self.num_embedding_sets):
            # 创建mask
            user_mask = torch.ones_like(seqs, device=self.dev) if include_user else torch.zeros_like(seqs, device=self.dev)
            item_mask = torch.ones_like(seqs, device=self.dev)
            
            # 安全索引：对各自的ID范围进行钳位，其它位置置零
            safe_user_ids = torch.where(user_mask.bool(), torch.clamp(seqs, min=0, max=self.user_num), torch.zeros_like(seqs))
            safe_item_ids = torch.where(item_mask.bool(), torch.clamp(seqs, min=0, max=self.item_num), torch.zeros_like(seqs))
            user_embedding = self.safe_embedding(self.user_emb[i], safe_user_ids)
            item_embedding = self.safe_embedding(self.item_emb[i], safe_item_ids)
            item_embeddings.append(item_embedding)
            user_embeddings.append(user_embedding)
        
        # 处理其他特征...
        item_feat_lists = []
        user_feat_lists = []
        
        for i in range(self.num_embedding_sets):
            item_feat_list = [item_embeddings[i]]
            user_feat_list = [user_embeddings[i]]
            
            # 处理稀疏特征
            for k in self.ITEM_SPARSE_FEAT:
                item_feat = self.feat2tensor(seq_feature, k, expected_len=seqs.shape[1])
                item_feat_emb = self.safe_embedding(self.sparse_emb[k], item_feat)
                item_feat_list.append(item_feat_emb)
            
            if include_user:
                for k in self.USER_SPARSE_FEAT:
                    user_feat = self.feat2tensor(seq_feature, k, expected_len=seqs.shape[1])
                    user_feat_emb = self.safe_embedding(self.sparse_emb[k], user_feat)
                    user_feat_list.append(user_feat_emb)
            
            # 处理数组特征
            for k in self.ITEM_ARRAY_FEAT:
                item_feat = self.feat2tensor(seq_feature, k, expected_len=seqs.shape[1])
                item_feat_emb = self.safe_embedding(self.sparse_emb[k], item_feat)
                # 对数组特征的embedding进行池化
                item_feat_emb = torch.mean(item_feat_emb, dim=2)
                item_feat_list.append(item_feat_emb)
            
            if include_user:
                for k in self.USER_ARRAY_FEAT:
                    user_feat = self.feat2tensor(seq_feature, k, expected_len=seqs.shape[1])
                    user_feat_emb = self.safe_embedding(self.sparse_emb[k], user_feat)
                    # 对数组特征的embedding进行池化
                    user_feat_emb = torch.mean(user_feat_emb, dim=2)
                    user_feat_list.append(user_feat_emb)
            
            # 处理连续特征
            for k in self.ITEM_CONTINUAL_FEAT:
                item_feat = self.feat2tensor(seq_feature, k, expected_len=seqs.shape[1])
                # 连续特征需要扩展维度以匹配embedding维度
                item_feat = item_feat.unsqueeze(-1).expand(-1, -1, self.item_emb[0].embedding_dim)
                item_feat_list.append(item_feat)
            
            if include_user:
                for k in self.USER_CONTINUAL_FEAT:
                    user_feat = self.feat2tensor(seq_feature, k, expected_len=seqs.shape[1])
                    # 连续特征需要扩展维度以匹配embedding维度
                    user_feat = user_feat.unsqueeze(-1).expand(-1, -1, self.user_emb[0].embedding_dim)
                    user_feat_list.append(user_feat)
            
            # 处理多模态特征
            for k in self.ITEM_EMB_FEAT:
                item_feat = self.feat2tensor(seq_feature, k, expected_len=seqs.shape[1])
                # 多模态特征需要经过线性变换
                item_feat_emb = self.emb_transform[k](item_feat.float())
                item_feat_list.append(item_feat_emb)
            
            item_feat_lists.append(item_feat_list)
            if include_user:
                user_feat_lists.append(user_feat_list)
        
        # 为每套embedding应用对应的Cross Network
        item_emb_sets = []
        user_emb_sets = []
        for i in range(self.num_embedding_sets):
            # merge features
            all_item_emb = torch.cat(item_feat_lists[i], dim=2)
            all_item_emb = torch.relu(self.itemdnn(all_item_emb))
            # 应用对应的Cross Network (reshape to 2D then back)
            B, L, H = all_item_emb.shape
            all_item_emb = all_item_emb.view(-1, H)
            all_item_emb = self.item_cross_nets[i](all_item_emb)
            all_item_emb = all_item_emb.view(B, L, H)
            item_emb_sets.append(all_item_emb)
            
            if include_user:
                all_user_emb = torch.cat(user_feat_lists[i], dim=2)
                all_user_emb = torch.relu(self.userdnn(all_user_emb))
                # 应用对应的Cross Network (reshape to 2D then back)
                Bu, Lu, Hu = all_user_emb.shape
                all_user_emb = all_user_emb.view(-1, Hu)
                all_user_emb = self.user_cross_nets[i](all_user_emb)
                all_user_emb = all_user_emb.view(Bu, Lu, Hu)
                user_emb_sets.append(all_user_emb)
        
        # 为每套embedding应用独立的Transformer层
        seq_emb_sets = []
        for i in range(self.num_embedding_sets):
            if include_user:
                seqs_emb = item_emb_sets[i] + user_emb_sets[i]
            else:
                seqs_emb = item_emb_sets[i]

            seqs_emb *= self.item_emb[0].embedding_dim**0.5  # 使用第一套embedding的维度
            poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
            mask_2d = (seqs_emb.abs().sum(dim=-1) != 0)
            poss = poss * mask_2d.long()
            # 防止位置索引越界（当实际 L > args.maxlen 时），将索引裁剪到 pos_emb 的合法范围内
            poss = torch.clamp(poss, min=0, max=self.pos_emb.num_embeddings - 1)
            seqs_emb += self.pos_emb(poss)
            seqs_emb = self.emb_dropout(seqs_emb)

            # 支持自定义特征生成器插拔
            if hasattr(self, 'feature_generator') and self.feature_generator is not None:
                seqs_emb = self.feature_generator(seqs_emb)

            maxlen_emb = seqs_emb.shape[1]
            ones_matrix = torch.ones((maxlen_emb, maxlen_emb), dtype=torch.bool, device=self.dev)
            attention_mask_tril = torch.tril(ones_matrix)
            attention_mask_pad = mask_2d.to(self.dev)
            attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

            # 使用第i套embedding对应的Transformer层
            for j in range(len(self.attention_layers[i])):
                if self.norm_first:
                    x = self.attention_layernorms[i][j](seqs_emb)
                    mha_outputs, _ = self.attention_layers[i][j](x, x, x, attn_mask=attention_mask)
                    seqs_emb = seqs_emb + mha_outputs
                    seqs_emb = seqs_emb + self.forward_layers[i][j](self.forward_layernorms[i][j](seqs_emb))
                else:
                    mha_outputs, _ = self.attention_layers[i][j](seqs_emb, seqs_emb, seqs_emb, attn_mask=attention_mask)
                    seqs_emb = self.attention_layernorms[i][j](seqs_emb + mha_outputs)
                    seqs_emb = self.forward_layernorms[i][j](seqs_emb + self.forward_layers[i][j](seqs_emb))

            log_feats = self.last_layernorm(seqs_emb)
            seq_emb_sets.append(log_feats)
        
        # 聚合多套embedding输出
        if hasattr(self, 'aggregation_type') and self.aggregation_type == 'weighted':
            # 加权平均聚合
            weights = getattr(self, 'embedding_weights', None)
            if weights is None:
                weights = torch.ones(len(seq_emb_sets), device=seq_emb_sets[0].device) / len(seq_emb_sets)
            weights = weights.view(-1, 1, 1)
            final_seq_emb = (torch.stack(seq_emb_sets, dim=0) * weights).sum(dim=0)
        elif hasattr(self, 'aggregation_type') and self.aggregation_type == 'adaptive':
            # 自适应融合（可插拔模块，示例：MLP融合）
            fusion_input = torch.stack(seq_emb_sets, dim=0).permute(1, 2, 0, 3).reshape(seq_emb_sets[0].shape[0], seq_emb_sets[0].shape[1], -1)
            if hasattr(self, 'fusion_mlp'):
                final_seq_emb = self.fusion_mlp(fusion_input)
            else:
                final_seq_emb = fusion_input.mean(dim=-1)
        else:
            # 默认平均
            final_seq_emb = torch.stack(seq_emb_sets, dim=0).mean(dim=0)
        return final_seq_emb

    def forward(
        self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature, neg_feature
    ):
                # Encoder
                log_feats = self.log2feats(user_item, mask, seq_feature)

                # 检查并处理log_feats中的NaN
                if torch.isnan(log_feats).any():
                    log_feats = torch.nan_to_num(log_feats, nan=0.0)

                # Generator
                generated_features = self.feature_generator(log_feats) if self.feature_generator else None

                # Predictor
                reconstructed_features = None
                if generated_features is not None:
                    # 检查并处理generated_features中的NaN
                    if torch.isnan(generated_features).any():
                        generated_features = torch.nan_to_num(generated_features, nan=0.0)

                    # All-Predict-All: 使用生成的特征进行重构
                    reconstructed_features = self.reconstruction_head(generated_features)

                    # 检查并处理reconstructed_features中的NaN
                    if reconstructed_features is not None:
                        for k, v in reconstructed_features.items():
                            if torch.isnan(v).any():
                                reconstructed_features[k] = torch.nan_to_num(v, nan=0.0)

                    # 将原始特征和生成的特征拼接，用于CTR预测
                    log_feats = torch.cat([log_feats, generated_features], dim=-1)

                pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
                neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)

                # 对齐时序长度（取共同的右对齐长度），防止元素乘法在维度1上不匹配
                bL = log_feats.size(1)
                pL = pos_embs.size(1)
                nL = neg_embs.size(1)
                common_len = min(bL, pL, nL)
                if not (bL == pL == nL):
                    if common_len <= 0:
                        raise RuntimeError(f"Invalid common sequence length: {common_len}, shapes: log_feats={log_feats.shape}, pos_embs={pos_embs.shape}, neg_embs={neg_embs.shape}")
                    log_feats = log_feats[:, -common_len:, :]
                    pos_embs = pos_embs[:, -common_len:, :]
                    neg_embs = neg_embs[:, -common_len:, :]
                    if next_mask is not None:
                        # next_mask 可能是 [B, L]，右对齐裁剪
                        if next_mask.dim() == 2 and next_mask.size(1) >= common_len:
                            next_mask = next_mask[:, -common_len:]

                # Cross Network for prediction
                # Note: The input to cross network needs to be carefully prepared.
                # Here we are just showing a simplified example.
                # A more complete implementation would involve concatenating user, item, and generated features.

                pos_logits = (log_feats * pos_embs).sum(dim=-1)
                neg_logits = (log_feats * neg_embs).sum(dim=-1)

                loss_mask = (next_mask == 1).to(self.dev)
                pos_logits = pos_logits * loss_mask
                neg_logits = neg_logits * loss_mask

                return pos_logits, neg_logits, reconstructed_features, generated_features, log_feats

    def predict(self, log_seqs, seq_feature, mask):
                """
                计算用户序列的表征
                Args:
                    log_seqs: 用户序列ID
                    seq_feature: 序列特征list，每个元素为当前时刻的特征字典
                    mask: token类型掩码，1表示item token，2表示user token
                Returns:
                    final_feat: 用户序列的表征，形状为 [batch_size, hidden_units]
                """
                log_feats = self.log2feats(log_seqs, mask, seq_feature)

                final_feat = log_feats[:, -1, :]

                return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
                """
                生成候选库item embedding，用于检索

                Args:
                    item_ids: 候选item ID（re-id形式）
                    retrieval_ids: 候选item ID（检索ID，从0开始编号，检索脚本使用）
                    feat_dict: 训练集所有item特征字典，key为特征ID，value为特征值
                    save_path: 保存路径
                    batch_size: 批次大小
                """
                all_embs = []

                for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
                    end_idx = min(start_idx + batch_size, len(item_ids))

                    item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
                    batch_feat = []
                    for i in range(start_idx, end_idx):
                        batch_feat.append(feat_dict[i])

                    batch_feat = np.array(batch_feat, dtype=object)

                    batch_emb = self.feat2emb(item_seq, [batch_feat], include_user=False).squeeze(0)

                    all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

                # 合并所有批次的结果并保存
                final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
                final_embs = np.concatenate(all_embs, axis=0)
                save_emb(final_embs, Path(save_path, 'embedding.fbin'))
                save_emb(final_ids, Path(save_path, 'id.u64bin'))
