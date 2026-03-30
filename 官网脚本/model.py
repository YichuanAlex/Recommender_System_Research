from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint as cp_checkpoint

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


class BaselineModel(torch.nn.Module):
    """
    Args:
        user_num: 用户数量
        item_num: 物品数量
        feat_statistics: 特征统计信息，key为特征ID，value为特征数量
        feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        args: 全局参数

    Attributes:
        user_num: 用户数量
        item_num: 物品数量
        dev: 设备
        norm_first: 是否先归一化
        maxlen: 序列最大长度
        item_emb: Item Embedding Table
        user_emb: User Embedding Table
        sparse_emb: 稀疏特征Embedding Table
        emb_transform: 多模态特征的线性变换
        userdnn: 用户特征拼接后经过的全连接层
        itemdnn: 物品特征拼接后经过的全连接层
    """

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):  #
        super(BaselineModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        # Use activation checkpointing to reduce memory during training
        # 从命令行参数控制，训练时可关闭以便调试/定位问题
        self.use_checkpoint = getattr(args, 'use_checkpoint', True)
        # 负样本流式chunk大小（越大越快、越占显存）
        try:
            self.neg_chunk_size = max(1, int(getattr(args, 'neg_chunk_size', 1)))
        except Exception:
            self.neg_chunk_size = 1

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()

        # 预计算掩码与位置索引，避免每步重复分配显存
        self.register_buffer(
            'causal_mask', torch.tril(torch.ones((args.maxlen, args.maxlen), dtype=torch.bool)), persistent=False
        )
        self.register_buffer(
            'pos_idx', torch.arange(1, args.maxlen + 1, dtype=torch.long), persistent=False
        )
        # 与DataLoader的pin_memory配合，GPU拷贝使用non_blocking
        self.non_blocking = (self.dev == 'cuda')

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        # 添加温度参数
        self.temp = torch.nn.Parameter(torch.ones([]) * 0.07)  # 初始温度0.07

        self._init_feat_info(feat_statistics, feat_types)

        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(
            self.USER_CONTINUAL_FEAT
        )
        itemdim = (
            args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT))
            + len(self.ITEM_CONTINUAL_FEAT)
            + self.item_emb.embedding_dim * len(self.ITEM_EMB_FEAT)
        )

        self.userdnn = torch.nn.Linear(userdim, args.hidden_units)
        self.itemdnn = torch.nn.Linear(itemdim, args.hidden_units)

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = FlashMultiHeadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate
            )  # 优化：用FlashAttention替代标准Attention
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)

    def _init_feat_info(self, feat_statistics, feat_types):
        """
        将特征统计信息（特征数量）按特征类型分组产生不同的字典，方便声明稀疏特征的Embedding Table

        Args:
            feat_statistics: 特征统计信息，key为特征ID，value为特征数量
            feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        """
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}  # 记录的是不同多模态特征的维度

    def feat2tensor(self, seq_feature, k, dataset):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]
            k: 特征ID
            dataset: Dataset实例，用于调用其feat2tensor方法

        Returns:
            batch_data: 特征值的tensor，形状为 [batch_size, maxlen, max_array_len(if array)]
        """
        return dataset.feat2tensor(seq_feature, k).to(self.dev, non_blocking=getattr(self, 'non_blocking', False))

    def feat2emb(self, seq, feature_array, dataset, mask=None, include_user=False):
        """
        Args:
            seq: 序列ID
            feature_array: 特征list，每个元素为当前时刻的特征字典
            dataset: Dataset实例，用于调用其feat2tensor方法
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征，在两种情况下不打开：1) 训练时在转换正负样本的特征时（因为正负样本都是item）;2) 生成候选库item embedding时。

        Returns:
            seqs_emb: 序列特征的Embedding
        """
        seq = seq.to(self.dev)
        # pre-compute embedding
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq)
            item_feat_list = [item_embedding]

        # batch-process all feature types
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
        ]

        if include_user:
            all_feat_types.extend(
                [
                    (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                    (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                    (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
                ]
            )

        # batch-process each feature type
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                tensor_feature = self.feat2tensor(feature_array, k, dataset)

                if feat_type.endswith('sparse'):
                    feat_list.append(self.sparse_emb[k](tensor_feature.long()))
                elif feat_type.endswith('array'):
                    feat_list.append(self.sparse_emb[k](tensor_feature.long()).sum(2))
                elif feat_type.endswith('continual'):
                    feat_list.append(tensor_feature.unsqueeze(2))

        for k in self.ITEM_EMB_FEAT:
            # 分批处理多模态特征以减少内存使用
            batch_size = len(feature_array)
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feature_array[0])
            
            # 使用较小的输出缓存（hidden维度），避免在emb_dim上进行大规模零张量预分配
            hidden = self.item_emb.embedding_dim
            dtype_target = item_embedding.dtype
            out_k = torch.zeros((batch_size, seq_len, hidden), device=self.dev, dtype=dtype_target)

            indices = []
            values = []
            for i, seq in enumerate(feature_array):
                for j, item in enumerate(seq):
                    if k in item:
                        indices.append([i, j])
                        values.append(item[k])  # numpy.ndarray
            
            if indices:
                idx_t = torch.tensor(indices, device=self.dev, dtype=torch.long)
                try:
                    values_np = np.stack(values).astype(np.float32, copy=False)
                except Exception:
                    values_np = np.array(values, dtype=np.float32)
                vals_t = torch.from_numpy(values_np).to(self.dev)
                # 直接对稀疏出现的位置进行线性变换，然后scatter到较小的输出缓存
                transformed = self.emb_transform[k](vals_t)
                transformed = transformed.to(dtype_target)
                out_k[idx_t[:, 0], idx_t[:, 1]] = transformed
            
            # 追加该特征的变换输出
            item_feat_list.append(out_k)

        # merge features
        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = torch.relu(self.itemdnn(all_item_emb))
        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = torch.relu(self.userdnn(all_user_emb))
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
        return seqs_emb

    def log2feats(self, log_seqs, mask, seq_feature, dataset):
        """
        Args:
            log_seqs: 序列ID
            mask: token类型掩码，1表示item token，2表示user token
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            dataset: Dataset实例，用于调用其feat2tensor方法

        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        seqs = self.feat2emb(log_seqs, seq_feature, dataset, mask=mask, include_user=True)
        seqs *= self.item_emb.embedding_dim**0.5
        # 使用预计算的位置索引，避免每步分配
        if maxlen > self.pos_idx.numel():
            dyn_pos_idx = torch.arange(1, maxlen + 1, dtype=torch.long, device=self.pos_idx.device)
            poss = dyn_pos_idx.unsqueeze(0).expand(batch_size, -1)
        else:
            poss = self.pos_idx[:maxlen].unsqueeze(0).expand(batch_size, -1)
        poss = poss * (log_seqs != 0).long()
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        # 使用预计算的下三角掩码（当 maxlen 超过缓存大小时按需构造）
        if maxlen > self.causal_mask.size(0):
            attention_mask_tril = torch.tril(
                torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.causal_mask.device)
            )
        else:
            attention_mask_tril = self.causal_mask[:maxlen, :maxlen]
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        for i in range(len(self.attention_layers)):
            if self.norm_first:
                def block_fn(seqs_in, attn_mask_in, i=i):
                    x = self.attention_layernorms[i](seqs_in)
                    mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attn_mask_in)
                    y = seqs_in + mha_outputs
                    y = y + self.forward_layers[i](self.forward_layernorms[i](y))
                    return y
                if self.training and self.use_checkpoint:
                    seqs = cp_checkpoint(block_fn, seqs, attention_mask, use_reentrant=False)
                else:
                    x = self.attention_layernorms[i](seqs)
                    mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                    seqs = seqs + mha_outputs
                    seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                def block_fn(seqs_in, attn_mask_in, i=i):
                    mha_outputs, _ = self.attention_layers[i](seqs_in, seqs_in, seqs_in, attn_mask=attn_mask_in)
                    y = self.attention_layernorms[i](seqs_in + mha_outputs)
                    y = self.forward_layernorms[i](y + self.forward_layers[i](y))
                    return y
                if self.training and self.use_checkpoint:
                    seqs = cp_checkpoint(block_fn, seqs, attention_mask, use_reentrant=False)
                else:
                    mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                    seqs = self.attention_layernorms[i](seqs + mha_outputs)
                    seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def compute_infonce_loss(self, seq_embs, pos_embs, neg_embs, loss_mask):
        """
        计算InfoNCE损失，使用每个位置自身的K个负样本，避免构建跨batch的大型相似度矩阵以降低内存占用
        
        Args:
            seq_embs: 序列嵌入 [batch_size, maxlen, hidden_size]
            pos_embs: 正样本嵌入 [batch_size, maxlen, hidden_size]
            neg_embs: 负样本嵌入 [batch_size, maxlen, num_neg, hidden_size]
            loss_mask: 损失掩码 [batch_size, maxlen]
            
        Returns:
            loss: InfoNCE损失, 以及正样本/负样本的平均相似度统计
        """
        # 归一化
        seq_embs = F.normalize(seq_embs, p=2, dim=-1)  # [B, L, H]
        pos_embs = F.normalize(pos_embs, p=2, dim=-1)  # [B, L, H]
        neg_embs = F.normalize(neg_embs, p=2, dim=-1)  # [B, L, K, H]

        # 正样本相似度 [B, L]
        pos_sim = torch.sum(seq_embs * pos_embs, dim=-1)
        
        # 负样本相似度：与每个位置的K个负样本逐一对比 [B, L, K]
        neg_sim = torch.sum(seq_embs.unsqueeze(2) * neg_embs, dim=-1)

        # 只保留有效位置
        valid_mask = loss_mask.bool()
        valid_pos = pos_sim[valid_mask].unsqueeze(1)            # [N, 1]
        valid_neg = neg_sim[valid_mask]                         # [N, K]

        # 组装logits: [N, 1+K]
        logits = torch.cat([valid_pos, valid_neg], dim=1) / self.temp
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss, pos_sim.mean().item(), neg_sim.mean().item()
    
    def compute_infonce_loss_stream(self, seq_embs, pos_embs, neg_embs, loss_mask):
        """
        流式版本：返回sum loss以及用于统计均值的分母，避免在forward中分配大的neg张量。
        Args:
            seq_embs: [B, L, H]
            pos_embs: [B, L, H]
            neg_embs: [B, L, K, H]
            loss_mask: [B, L]
        Returns:
            loss_sum: 标量Tensor（保留计算图），等于该chunk内交叉熵的求和
            pos_sum: float，pos_sim的元素和（不参与梯度）
            neg_sum: float，neg_sim的元素和（不参与梯度）
            pos_count: int，pos_sim元素个数（用于还原均值）
            neg_count: int，neg_sim元素个数（用于还原均值）
            n_valid: int，参与CE的有效位置个数
        """
        # 归一化
        seq_norm = F.normalize(seq_embs, p=2, dim=-1)
        pos_norm = F.normalize(pos_embs, p=2, dim=-1)
        neg_norm = F.normalize(neg_embs, p=2, dim=-1)

        pos_sim = torch.sum(seq_norm * pos_norm, dim=-1)            # [B, L]
        neg_sim = torch.sum(seq_norm.unsqueeze(2) * neg_norm, dim=-1)  # [B, L, K]

        valid_mask = loss_mask.bool()
        valid_pos = pos_sim[valid_mask].unsqueeze(1)               # [N, 1]
        valid_neg = neg_sim[valid_mask]                            # [N, K]
        logits = torch.cat([valid_pos, valid_neg], dim=1) / self.temp
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        loss_sum = F.cross_entropy(logits, labels, reduction='sum')

        # 统计量使用 .sum().item() 以避免图累积
        pos_sum = float(pos_sim.sum().item())
        neg_sum = float(neg_sim.sum().item())
        pos_count = int(pos_sim.numel())
        neg_count = int(neg_sim.numel())
        n_valid = int(labels.numel())
        return loss_sum, pos_sum, neg_sum, pos_count, neg_count, n_valid
    
    def forward(self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature, neg_feature, dataset):
        """
        支持InfoNCE Loss的前向传播，每个正样本配10个负样本
        返回loss: 标量损失值
        """
        log_feats = self.log2feats(user_item, mask, seq_feature, dataset)  # [batch, maxlen, hidden]
        loss_mask = (next_mask == 1).to(self.dev)

        pos_embs = self.feat2emb(pos_seqs, pos_feature, dataset, include_user=False)  # [batch, maxlen, hidden]
        
        # 流式处理负样本，避免分配 [B, L, K, H] 的大张量
        batch, maxlen, num_neg = neg_seqs.shape
        # 聚合器
        loss_sum_total = torch.zeros((), device=self.dev, dtype=log_feats.dtype)
        pos_sum_total = 0.0
        neg_sum_total = 0.0
        pos_cnt_total = 0
        neg_cnt_total = 0
        n_valid_total = 0

        chunk_size = max(1, getattr(self, 'neg_chunk_size', 1))  # 可配置chunk以折中速度/显存
        for i in range(0, batch, chunk_size):
            end_idx = min(i + chunk_size, batch)
            curr_batch_size = end_idx - i
            curr_neg_seqs = neg_seqs[i:end_idx].reshape(curr_batch_size * maxlen * num_neg)
            curr_neg_feature = [item for b in neg_feature[i:end_idx] for row in b for item in row]

            curr_neg_embs = self.feat2emb(
                curr_neg_seqs.view(-1, 1),
                [[f] for f in curr_neg_feature],
                dataset,
                include_user=False
            )  # [curr_batch*maxlen*num_neg, 1, hidden]
            curr_neg_embs = curr_neg_embs.view(curr_batch_size, maxlen, num_neg, -1)

            # 计算当前chunk的sum loss与统计量
            loss_sum, pos_sum, neg_sum, pos_cnt, neg_cnt, n_valid = self.compute_infonce_loss_stream(
                log_feats[i:end_idx], pos_embs[i:end_idx], curr_neg_embs, loss_mask[i:end_idx]
            )
            loss_sum_total = loss_sum_total + loss_sum
            pos_sum_total += pos_sum
            neg_sum_total += neg_sum
            pos_cnt_total += pos_cnt
            neg_cnt_total += neg_cnt
            n_valid_total += n_valid

            # 释放中间变量，降低峰值
            del curr_neg_embs, curr_neg_seqs, curr_neg_feature, loss_sum
            if torch.cuda.is_available() and self.dev == 'cuda' and (i // chunk_size) % 8 == 0:
                torch.cuda.empty_cache()

        # 归一化得到最终loss与均值统计
        # 防止除零
        denom = max(1, n_valid_total)
        loss = loss_sum_total / denom
        pos_sim = pos_sum_total / max(1, pos_cnt_total)
        neg_sim = neg_sum_total / max(1, neg_cnt_total)
        return loss, pos_sim, neg_sim

    def predict(self, log_seqs, seq_feature, mask, dataset):
        """
        计算用户序列的表征
        Args:
            log_seqs: 用户序列ID
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            mask: token类型掩码，1表示item token，2表示user token
            dataset: Dataset实例，用于调用其feat2tensor方法
        Returns:
            final_feat: 用户序列的表征，形状为 [batch_size, hidden_units]
        """
        log_feats = self.log2feats(log_seqs, mask, seq_feature, dataset)

        final_feat = log_feats[:, -1, :]

        # 添加归一化处理
        final_feat = F.normalize(final_feat, p=2, dim=-1)

        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, dataset, batch_size=1024):
        """
        生成候选库item embedding，用于检索

        Args:
            item_ids: 候选item ID（re-id形式）
            retrieval_ids: 候选item ID（检索ID，从0开始编号，检索脚本使用）
            feat_dict: 训练集所有item特征字典，key为特征ID，value为特征值
            save_path: 保存路径
            dataset: Dataset实例，用于调用其feat2tensor方法
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

            batch_emb = self.feat2emb(item_seq, [batch_feat], dataset, include_user=False).squeeze(0)

            # 添加归一化处理
            batch_emb = F.normalize(batch_emb, p=2, dim=-1)

            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))