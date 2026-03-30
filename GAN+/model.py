from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

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


class SwiGLU(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.linear1 = nn.Linear(hidden_units, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
    def forward(self, x):
        return F.silu(self.linear1(x)) * self.linear2(x)

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.swiglu = SwiGLU(hidden_units)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)
    def forward(self, inputs):
        x = self.dropout1(self.conv1(inputs.transpose(-1, -2)))
        x = self.swiglu(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.dropout2(self.conv2(x))
        return x.transpose(-1, -2)


class NaryEncoder(nn.Module):
    def __init__(self, nary_bases, emb_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(base, emb_dim) for base in nary_bases
        ])
        self.nary_bases = nary_bases
        # 添加一个线性层来调整最终输出维度
        total_dim = len(nary_bases) * emb_dim
        self.output_adapter = nn.Linear(total_dim, emb_dim)
    
    def forward(self, x):
        # x: [batch, dim] or [batch, seq, dim]
        outs = []
        for i, base in enumerate(self.nary_bases):
            code = ((x // (base ** i)) % base).long()
            outs.append(self.embeddings[i](code))
        concatenated = torch.cat(outs, dim=-1)
        return self.output_adapter(concatenated)


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

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args, gan_quantizer=None, semantic_emb=None, nary_encoder=None):
        super(BaselineModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self._init_feat_info(feat_statistics, feat_types)

        # 先设置这些属性，然后再计算itemdim
        self.gan_quantizer = gan_quantizer
        self.semantic_emb = semantic_emb
        self.nary_encoder = nary_encoder
        self.binary_emb = nn.Embedding(2, args.hidden_units)  # for binary encoding sum pooling

        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(
            self.USER_CONTINUAL_FEAT
        )
        
        # 计算itemdim，考虑所有特征类型
        itemdim = (
            args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT))
            + len(self.ITEM_CONTINUAL_FEAT)
            + args.hidden_units * len(self.ITEM_EMB_FEAT)
        )
        
        # 为多模态特征添加额外的维度（N-ary编码、二进制编码、GAN语义ID）
        # 注意：这些特征只对第一个embedding特征添加，不是为每个embedding特征添加
        if self.nary_encoder is not None:
            itemdim += args.hidden_units  # 只添加一次N-ary编码
        if self.gan_quantizer is not None and self.semantic_emb is not None:
            itemdim += args.hidden_units  # 只添加一次GAN语义ID
        # 二进制编码只添加一次
        itemdim += args.hidden_units  # 二进制编码

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

    def feat2tensor(self, seq_feature, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [batch_size, maxlen, max_array_len(if array)]
        """
        batch_size = len(seq_feature)

        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
            max_array_len = 0
            max_seq_len = 0

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))

            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]

            return torch.from_numpy(batch_data).to(self.dev)
        else:
            # 如果特征是Sparse类型，直接转换为tensor
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i] = seq_data

            return torch.from_numpy(batch_data).to(self.dev)

    def feat2emb(self, seq, feature_array, mask=None, include_user=False):
        seq = seq.to(self.dev)
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
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
        ]
        if include_user:
            all_feat_types.extend([
                (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
            ])
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue
            for k in feat_dict:
                tensor_feature = self.feat2tensor(feature_array, k)
                if feat_type.endswith('sparse'):
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type.endswith('array'):
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type.endswith('continual'):
                    feat_list.append(tensor_feature.unsqueeze(2))
        
        # 处理多模态embedding特征
        for k in self.ITEM_EMB_FEAT:
            batch_size = len(feature_array)
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feature_array[0])
            
            # 从item_feature字典中提取多模态embedding
            batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)
            for i, seq_features in enumerate(feature_array):
                for j, item_features in enumerate(seq_features):
                    if isinstance(item_features, dict) and k in item_features:
                        # item_features[k]应该是多模态embedding数组
                        emb_data = item_features[k]
                        if isinstance(emb_data, (list, np.ndarray)):
                            if len(emb_data) == emb_dim:
                                batch_emb_data[i, j] = emb_data
                            else:
                                print(f"[WARNING] Embedding dimension mismatch: expected {emb_dim}, got {len(emb_data)}")
                        else:
                            print(f"[WARNING] Unexpected embedding data type: {type(emb_data)}")
            
            tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
            
            # 对于第一个embedding特征，添加N-ary编码、二进制编码和GAN语义ID
            if k == list(self.ITEM_EMB_FEAT.keys())[0]:  # 只对第一个embedding特征添加额外特征
                # N-ary编码
                if self.nary_encoder is not None:
                    nary_emb = self.nary_encoder(tensor_feature)
                    if nary_emb.dim() == 4:
                        nary_emb = nary_emb.view(batch_size, seq_len, -1)
                    # 确保N-ary编码输出维度与hidden_units匹配
                    if nary_emb.shape[-1] != self.item_emb.embedding_dim:
                        if not hasattr(self, 'nary_dim_adapter'):
                            self.nary_dim_adapter = nn.Linear(nary_emb.shape[-1], self.item_emb.embedding_dim).to(self.dev)
                        nary_emb = self.nary_dim_adapter(nary_emb)
                    item_feat_list.append(nary_emb)
                
                # 二进制编码sum pooling
                binary_code = (tensor_feature > 0).long()
                binary_emb = self.binary_emb(binary_code)
                # 确保binary_emb的维度正确
                if binary_emb.dim() == 4:
                    binary_emb = binary_emb.view(batch_size, seq_len, -1)
                # 如果维度不匹配，使用线性层调整
                if binary_emb.shape[-1] != self.item_emb.embedding_dim:
                    if not hasattr(self, 'binary_dim_adapter'):
                        self.binary_dim_adapter = nn.Linear(binary_emb.shape[-1], self.item_emb.embedding_dim).to(self.dev)
                    binary_emb = self.binary_dim_adapter(binary_emb)
                item_feat_list.append(binary_emb)
                
                # GAN量化器
                if self.gan_quantizer is not None and self.semantic_emb is not None:
                    flat_tensor = tensor_feature.view(-1, emb_dim)
                    semantic_id = self.gan_quantizer.get_semantic_id(flat_tensor)
                    semantic_emb = self.semantic_emb(semantic_id).view(batch_size, seq_len, -1)
                    # 确保GAN量化器输出维度与hidden_units匹配
                    if semantic_emb.shape[-1] != self.item_emb.embedding_dim:
                        if not hasattr(self, 'semantic_dim_adapter'):
                            self.semantic_dim_adapter = nn.Linear(semantic_emb.shape[-1], self.item_emb.embedding_dim).to(self.dev)
                        semantic_emb = self.semantic_dim_adapter(semantic_emb)
                    item_feat_list.append(semantic_emb)
            
            # 添加原始的embedding特征
            emb = self.emb_transform[k](tensor_feature)
            if emb.dim() == 4:
                emb = emb.view(batch_size, seq_len, -1)
            item_feat_list.append(emb)
        
        # 调试输出每个特征shape
        for idx, feat in enumerate(item_feat_list):
            print(f"[DEBUG] item_feat_list[{idx}] shape: {feat.shape}")
        
        # merge features
        all_item_emb = torch.cat(item_feat_list, dim=2)
        print(f"[DEBUG] all_item_emb shape before itemdnn: {all_item_emb.shape}")
        print(f"[DEBUG] itemdnn input_dim: {self.itemdnn.in_features}, output_dim: {self.itemdnn.out_features}")
        
        # 检查维度是否匹配
        if all_item_emb.shape[-1] != self.itemdnn.in_features:
            print(f"[ERROR] Dimension mismatch! all_item_emb.shape[-1]={all_item_emb.shape[-1]}, itemdnn.in_features={self.itemdnn.in_features}")
            # 如果维度不匹配，使用线性层调整维度
            if not hasattr(self, 'item_dim_adapter'):
                self.item_dim_adapter = nn.Linear(all_item_emb.shape[-1], self.itemdnn.in_features).to(self.dev)
            all_item_emb = self.item_dim_adapter(all_item_emb)
            print(f"[DEBUG] After dimension adjustment: {all_item_emb.shape}")
        
        all_item_emb = torch.relu(self.itemdnn(all_item_emb))
        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = torch.relu(self.userdnn(all_user_emb))
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
        return seqs_emb

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
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        seqs *= self.item_emb.embedding_dim**0.5
        poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
        poss *= log_seqs != 0
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        for i in range(len(self.attention_layers)):
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)
        # 显式二阶交叉（user token与item token的Hadamard积）
        if hasattr(self, 'user_emb'):
            user_token_idx = (mask == 2).nonzero(as_tuple=True)
            if user_token_idx[0].numel() > 0:
                user_vec = self.user_emb(log_seqs[user_token_idx[0], user_token_idx[1]])
                cross = torch.zeros_like(log_feats)
                cross[user_token_idx[0], user_token_idx[1], :] = log_feats[user_token_idx[0], user_token_idx[1], :] * user_vec
                log_feats = log_feats + cross
        return log_feats

    def forward(
        self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature, neg_feature
    ):
        """
        训练时调用，计算正负样本的logits

        Args:
            user_item: 用户序列ID
            pos_seqs: 正样本序列ID
            neg_seqs: 负样本序列ID
            mask: token类型掩码，1表示item token，2表示user token
            next_mask: 下一个token类型掩码，1表示item token，2表示user token
            next_action_type: 下一个token动作类型，0表示曝光，1表示点击
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            pos_feature: 正样本特征list，每个元素为当前时刻的特征字典
            neg_feature: 负样本特征list，每个元素为当前时刻的特征字典

        Returns:
            pos_logits: 正样本logits，形状为 [batch_size, maxlen]
            neg_logits: 负样本logits，形状为 [batch_size, maxlen]
        """
        log_feats = self.log2feats(user_item, mask, seq_feature)
        loss_mask = (next_mask == 1).to(self.dev)

        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)

        # Target作为Query，历史行为作为Key的Attention
        # 使用target embedding与历史行为的点积来计算attention权重
        target_emb = pos_embs[:, -1, :]  # [batch_size, hidden_units]
        # 计算target与历史行为的相似度
        attention_weights = torch.matmul(log_feats, target_emb.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]
        attention_weights = F.softmax(attention_weights, dim=-1)
        # 加权聚合历史行为
        attn_output = torch.matmul(attention_weights.unsqueeze(1), log_feats).squeeze(1)  # [batch_size, hidden_units]
        
        # 表征侧交互
        cross = pos_embs * log_feats
        # 使用加权平均的attention输出作为最终特征
        final_feat = attn_output  # [batch_size, hidden_units]
        
        # 计算logits - 修复索引问题
        pos_logits = (final_feat * pos_embs[:, -1, :]).sum(dim=-1)  # [batch_size]
        neg_logits = (final_feat * neg_embs[:, -1, :]).sum(dim=-1)  # [batch_size]
        
        # 确保loss_mask的维度正确
        loss_mask_last = loss_mask[:, -1]  # [batch_size]
        
        # 应用mask
        pos_logits = pos_logits * loss_mask_last
        neg_logits = neg_logits * loss_mask_last

        return pos_logits, neg_logits

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
