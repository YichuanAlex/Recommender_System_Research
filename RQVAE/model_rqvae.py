"""
选手可参考以下流程，使用提供的 RQ-VAE 框架代码将多模态emb数据转换为Semantic Id:
1. 使用 MmEmbDataset 读取不同特征 ID 的多模态emb数据.
2. 训练 RQ-VAE 模型, 训练完成后将数据转换为Semantic Id.
3. 参照 Item Sparse 特征格式处理Semantic Id，作为新特征加入Baseline模型训练.
"""

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm

# 导入BaselineModel
from model import BaselineModel

# class MmEmbDataset(torch.utils.data.Dataset):
#     """
#     Build Dataset for RQ-VAE Training

#     Args:
#         data_dir = os.environ.get('TRAIN_DATA_PATH')
#         feature_id = MM emb ID
#     """

#     def __init__(self, data_dir, feature_id):
#         super().__init__()
#         self.data_dir = Path(data_dir)
#         self.mm_emb_id = [feature_id]
#         self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_id)

#         self.mm_emb = self.mm_emb_dict[self.mm_emb_id[0]]
#         self.tid_list, self.emb_list = list(self.mm_emb.keys()), list(self.mm_emb.values())
#         self.emb_list = [torch.tensor(emb, dtype=torch.float32) for emb in self.emb_list]

#         assert len(self.tid_list) == len(self.emb_list)
#         self.item_cnt = len(self.tid_list)

#     def __getitem__(self, index):
#         tid = torch.tensor(self.tid_list[index], dtype=torch.long)
#         emb = self.emb_list[index]
#         return tid, emb

#     def __len__(self):
#         return self.item_cnt

#     @staticmethod
#     def collate_fn(batch):
#         tid, emb = zip(*batch)

#         tid_batch, emb_batch = torch.stack(tid, dim=0), torch.stack(emb, dim=0)
#         return tid_batch, emb_batch


## Kmeans
def kmeans(data, n_clusters, kmeans_iters):
    """
    auto init: n_init = 10 if n_clusters <= 10 else 1
    """
    # 如果数据样本数小于聚类数，调整聚类数为数据样本数
    num_samples = data.shape[0]
    if num_samples < n_clusters:
        print(f"Warning: Number of samples ({num_samples}) is less than n_clusters ({n_clusters}). "
              f"Adjusting n_clusters to {num_samples}.")
        n_clusters = num_samples
        # 如果调整后聚类数为0，则直接返回
        if n_clusters == 0:
            return torch.empty(0, data.shape[1]), torch.empty(0, dtype=torch.long)
    
    km = KMeans(n_clusters=n_clusters, max_iter=kmeans_iters, n_init="auto")

    # sklearn only support cpu
    data_cpu = data.detach().cpu()
    np_data = data_cpu.numpy()
    km.fit(np_data)
    return torch.tensor(km.cluster_centers_), torch.tensor(km.labels_)


## Balanced Kmeans
class BalancedKmeans(torch.nn.Module):
    def __init__(self, num_clusters: int, kmeans_iters: int, tolerance: float, device: str):
        super().__init__()
        self.num_clusters = num_clusters
        self.kmeans_iters = kmeans_iters
        self.tolerance = tolerance
        self.device = device
        self._codebook = None

    def _compute_distances(self, data):
        return torch.cdist(data, self._codebook)

    def _assign_clusters(self, dist):
        samples_cnt = dist.shape[0]
        samples_labels = torch.zeros(samples_cnt, dtype=torch.long, device=self.device)
        clusters_cnt = torch.zeros(self.num_clusters, dtype=torch.long, device=self.device)

        sorted_indices = torch.argsort(dist, dim=-1)
        for i in range(samples_cnt):
            for j in range(self.num_clusters):
                cluster_idx = sorted_indices[i, j]
                if clusters_cnt[cluster_idx] < samples_cnt // self.num_clusters:
                    samples_labels[i] = cluster_idx
                    clusters_cnt[cluster_idx] += 1
                    break

        return samples_labels

    def _update_codebook(self, data, samples_labels):
        _new_codebook = []
        for i in range(self.num_clusters):
            cluster_data = data[samples_labels == i]
            if len(cluster_data) > 0:
                _new_codebook.append(cluster_data.mean(dim=0))
            else:
                _new_codebook.append(self._codebook[i])
        return torch.stack(_new_codebook)

    def fit(self, data):
        num_emb, codebook_emb_dim = data.shape
        data = data.to(self.device)

        # initialize codebook
        indices = torch.randperm(num_emb)[: self.num_clusters]
        self._codebook = data[indices].clone()

        for _ in range(self.kmeans_iters):
            dist = self._compute_distances(data)
            samples_labels = self._assign_clusters(dist)
            _new_codebook = self._update_codebook(data, samples_labels)
            if torch.norm(_new_codebook - self._codebook) < self.tolerance:
                break

            self._codebook = _new_codebook

        return self._codebook, samples_labels

    def predict(self, data):
        data = data.to(self.device)
        dist = self._compute_distances(data)
        samples_labels = self._assign_clusters(dist)
        return samples_labels


## Base RQVAE
class RQEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_channels: list, latent_dim: int):
        super().__init__()

        self.stages = torch.nn.ModuleList()
        in_dim = input_dim

        for out_dim in hidden_channels:
            stage = torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim), torch.nn.ReLU())
            self.stages.append(stage)
            in_dim = out_dim

        self.stages.append(torch.nn.Sequential(torch.nn.Linear(in_dim, latent_dim), torch.nn.ReLU()))

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


class RQDecoder(torch.nn.Module):
    def __init__(self, latent_dim: int, hidden_channels: list, output_dim: int):
        super().__init__()

        self.stages = torch.nn.ModuleList()
        in_dim = latent_dim

        for out_dim in hidden_channels:
            stage = torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim), torch.nn.ReLU())
            self.stages.append(stage)
            in_dim = out_dim

        self.stages.append(torch.nn.Sequential(torch.nn.Linear(in_dim, output_dim), torch.nn.ReLU()))

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


## Generate semantic id
class VQEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        num_clusters,
        codebook_emb_dim: int,
        kmeans_method: str,
        kmeans_iters: int,
        distances_method: str,
        device: str,
    ):
        super(VQEmbedding, self).__init__(num_clusters, codebook_emb_dim)

        self.num_clusters = num_clusters
        self.codebook_emb_dim = codebook_emb_dim
        self.kmeans_method = kmeans_method
        self.kmeans_iters = kmeans_iters
        self.distances_method = distances_method
        self.device = device

    def _create_codebook(self, data):
        # 对输入数据进行去重处理，避免k-means聚类时出现重复点
        unique_data = torch.unique(data, dim=0)
        # 如果去重后的数据点少于聚类数，调整聚类数
        if unique_data.shape[0] < self.num_clusters:
            print(f"Warning: Number of unique data points ({unique_data.shape[0]}) is less than num_clusters ({self.num_clusters}). Adjusting num_clusters.")
            self.num_clusters = unique_data.shape[0]
        
        if self.kmeans_method == 'kmeans':
            _codebook, _ = kmeans(unique_data, self.num_clusters, self.kmeans_iters)
            # Update self.num_clusters to match the actual number of clusters in the codebook
            self.num_clusters = _codebook.shape[0]
        elif self.kmeans_method == 'bkmeans':
            BKmeans = BalancedKmeans(
                num_clusters=self.num_clusters, kmeans_iters=self.kmeans_iters, tolerance=1e-4, device=self.device
            )
            _codebook, _ = BKmeans.fit(unique_data)
            # Update self.num_clusters to match the actual number of clusters in the codebook
            self.num_clusters = _codebook.shape[0]
        else:
            _codebook = torch.randn(self.num_clusters, self.codebook_emb_dim)
        _codebook = _codebook.to(self.device)
        assert _codebook.shape == (self.num_clusters, self.codebook_emb_dim)
        self.codebook = torch.nn.Parameter(_codebook)
        # No need to restore original num_clusters as we should use the actual number

    @torch.no_grad()
    def _compute_distances(self, data):
        _codebook_t = self.codebook.t()
        # Use the actual codebook shape instead of expecting a fixed num_clusters
        assert _codebook_t.shape == (self.codebook_emb_dim, self.codebook.shape[0])
        assert data.shape[-1] == self.codebook_emb_dim

        if self.distances_method == 'cosine':
            data_norm = F.normalize(data, p=2, dim=-1)
            _codebook_t_norm = F.normalize(_codebook_t, p=2, dim=0)
            distances = 1 - torch.mm(data_norm, _codebook_t_norm)
        # l2
        else:
            data_norm_sq = data.pow(2).sum(dim=-1, keepdim=True)
            _codebook_t_norm_sq = _codebook_t.pow(2).sum(dim=0, keepdim=True)
            distances = torch.addmm(data_norm_sq + _codebook_t_norm_sq, data, _codebook_t, beta=1.0, alpha=-2.0)
        return distances

    @torch.no_grad()
    def _create_semantic_id(self, data):
        distances = self._compute_distances(data)
        _semantic_id = torch.argmin(distances, dim=-1)
        return _semantic_id

    def _update_emb(self, _semantic_id):
        update_emb = super().forward(_semantic_id)
        return update_emb

    def forward(self, data):
        self._create_codebook(data)
        _semantic_id = self._create_semantic_id(data)
        update_emb = self._update_emb(_semantic_id)

        return update_emb, _semantic_id


## Residual Quantizer
class RQ(torch.nn.Module):
    """
    Args:
        num_codebooks, codebook_size, codebook_emb_dim -> Build codebook
        if_shared_codebook -> If use same codebook
        kmeans_method, kmeans_iters -> Initialize codebook
        distances_method -> Generate semantic_id

        loss_beta -> Calculate RQ-VAE loss
    """

    def __init__(
        self,
        num_codebooks: int,
        codebook_size: list,
        codebook_emb_dim,
        shared_codebook: bool,
        kmeans_method,
        kmeans_iters,
        distances_method,
        loss_beta: float,
        device: str,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        assert len(self.codebook_size) == self.num_codebooks
        self.codebook_emb_dim = codebook_emb_dim
        self.shared_codebook = shared_codebook

        self.kmeans_method = kmeans_method
        self.kmeans_iters = kmeans_iters
        self.distances_method = distances_method
        self.loss_beta = loss_beta
        self.device = device

        if self.shared_codebook:
            self.vqmodules = torch.nn.ModuleList(
                [
                    VQEmbedding(
                        self.codebook_size[0],
                        self.codebook_emb_dim,
                        self.kmeans_method,
                        self.kmeans_iters,
                        self.distances_method,
                        self.device,
                    )
                    for _ in range(self.num_codebooks)
                ]
            )

        else:
            self.vqmodules = torch.nn.ModuleList(
                [
                    VQEmbedding(
                        self.codebook_size[idx],
                        self.codebook_emb_dim,
                        self.kmeans_method,
                        self.kmeans_iters,
                        self.distances_method,
                        self.device,
                    )
                    for idx in range(self.num_codebooks)
                ]
            )

    def quantize(self, data):
        """
        Exa:
            i-th quantize: input[i]( i.e. res[i-1] ) = VQ[i] + res[i]
            vq_emb_list: [vq1, vq1+vq2, ...]
            res_emb_list: [res1, res2, ...]
            semantic_id_list: [vq1_sid, vq2_sid, ...]

        Returns:
            vq_emb_list[0] -> [batch_size, codebook_emb_dim]
            semantic_id_list -> [batch_size, num_codebooks]
        """
        res_emb = data.detach().clone()

        vq_emb_list, res_emb_list = [], []
        semantic_id_list = []
        vq_emb_aggre = torch.zeros_like(data)

        for i in range(self.num_codebooks):
            vq_emb, _semantic_id = self.vqmodules[i](res_emb)

            res_emb -= vq_emb
            vq_emb_aggre += vq_emb

            res_emb_list.append(res_emb)
            vq_emb_list.append(vq_emb_aggre)
            semantic_id_list.append(_semantic_id.unsqueeze(dim=-1))

        semantic_id_list = torch.cat(semantic_id_list, dim=-1)
        return vq_emb_list, res_emb_list, semantic_id_list

    def _rqvae_loss(self, vq_emb_list, res_emb_list):
        rqvae_loss_list = []
        for idx, quant in enumerate(vq_emb_list):
            # stop gradient
            loss1 = (res_emb_list[idx].detach() - quant).pow(2.0).mean()
            loss2 = (res_emb_list[idx] - quant.detach()).pow(2.0).mean()
            partial_loss = loss1 + self.loss_beta * loss2
            rqvae_loss_list.append(partial_loss)

        rqvae_loss = torch.sum(torch.stack(rqvae_loss_list))
        return rqvae_loss

    def forward(self, data):
        vq_emb_list, res_emb_list, semantic_id_list = self.quantize(data)
        rqvae_loss = self._rqvae_loss(vq_emb_list, res_emb_list)

        return vq_emb_list, semantic_id_list, rqvae_loss


class RQVAE(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_channels: list,
        latent_dim: int,
        num_codebooks: int,
        codebook_size: list,
        shared_codebook: bool,
        kmeans_method,
        kmeans_iters,
        distances_method,
        loss_beta: float,
        device: str,
    ):
        super().__init__()
        # 添加维度对齐层，将输入维度统一转换为1024维
        self.input_dim = input_dim
        self.target_dim = 1024
        
        # 如果输入维度不是1024，则添加线性变换层
        if input_dim != self.target_dim:
            self.dim_alignment = torch.nn.Linear(input_dim, self.target_dim)
        else:
            self.dim_alignment = None
        
        # 编码器和解码器都使用目标维度
        self.encoder = RQEncoder(self.target_dim, hidden_channels, latent_dim).to(device)
        self.decoder = RQDecoder(latent_dim, hidden_channels[::-1], self.target_dim).to(device)
        self.rq = RQ(
            num_codebooks,
            codebook_size,
            latent_dim,
            shared_codebook,
            kmeans_method,
            kmeans_iters,
            distances_method,
            loss_beta,
            device,
        ).to(device)

    def encode(self, x):
        # 移除了重复的维度对齐逻辑
        # 现在假设输入x已经经过维度对齐
        return self.encoder(x)

    def decode(self, z_vq):
        if isinstance(z_vq, list):
            z_vq = z_vq[-1]
        decoded = self.decoder(z_vq)
        # 如果原始输入维度不是1024，可能需要添加逆变换（视具体情况而定）
        return decoded

    def compute_loss(self, x_hat, x_gt, rqvae_loss):
        # 如果原始输入维度不是1024，需要将x_gt也变换到目标维度进行损失计算
        if self.dim_alignment is not None and self.input_dim != self.target_dim:
            x_gt = self.dim_alignment(x_gt)
        recon_loss = F.mse_loss(x_hat, x_gt, reduction="mean")
        total_loss = recon_loss + rqvae_loss
        return recon_loss, rqvae_loss, total_loss

    def _get_codebook(self, x_gt):
        # 如果需要维度对齐，则先进行线性变换
        if self.dim_alignment is not None and self.input_dim != self.target_dim:
            x_gt = self.dim_alignment(x_gt)
        z_e = self.encode(x_gt)
        vq_emb_list, semantic_id_list, rqvae_loss = self.rq(z_e)
        return semantic_id_list

    def forward(self, x_gt):
        # 添加调试信息，检查输入维度
        print(f"RQVAE input shape: {x_gt.shape}")
        print(f"RQVAE expected input_dim: {self.input_dim}")
        print(f"RQVAE target_dim: {self.target_dim}")
        if self.dim_alignment is not None:
            print(f"Dimension alignment layer exists")
        else:
            print(f"No dimension alignment layer")
        
        # 检查输入数据中唯一样本的数量
        unique_samples = torch.unique(x_gt, dim=0)
        print(f"Unique samples in batch: {unique_samples.shape[0]} out of {x_gt.shape[0]}")
        
        # 保存原始输入用于损失计算
        original_x_gt = x_gt
        
        # 如果需要维度对齐，则先进行线性变换
        if self.dim_alignment is not None and self.input_dim != self.target_dim:
            x_gt = self.dim_alignment(x_gt)
        
        # 使用经过维度对齐的输入
        z_e = self.encode(x_gt)  # 现在传入的是已经对齐的输入
        vq_emb_list, semantic_id_list, rqvae_loss = self.rq(z_e)
        x_hat = self.decode(vq_emb_list)
        
        # 在计算损失时，如果需要，对原始输入进行维度对齐
        recon_loss, rqvae_loss, total_loss = self.compute_loss(x_hat, original_x_gt, rqvae_loss)
        return x_hat, semantic_id_list, recon_loss, rqvae_loss, total_loss


# 新增的结合BaselineModel与RQ-VAE功能的模型类
class RQVAEEnhancedModel(BaselineModel):
    """
    结合BaselineModel与RQ-VAE功能的新模型类
    继承BaselineModel的所有功能，并添加RQ-VAE的量化和解码能力
    """
    def __init__(self, user_num, item_num, feat_statistics, feat_types, args, rqvae_config):
        """
        初始化RQVAEEnhancedModel
        
        Args:
            user_num: 用户数量
            item_num: 物品数量
            feat_statistics: 特征统计信息
            feat_types: 特征类型
            args: 全局参数
            rqvae_config: RQ-VAE配置参数
        """
        # 初始化BaselineModel
        super(RQVAEEnhancedModel, self).__init__(user_num, item_num, feat_statistics, feat_types, args)
        
        # 初始化RQ-VAE组件
        self.rqvae = RQVAE(
            input_dim=rqvae_config['input_dim'],
            hidden_channels=rqvae_config['hidden_channels'],
            latent_dim=rqvae_config['latent_dim'],
            num_codebooks=rqvae_config['num_codebooks'],
            codebook_size=rqvae_config['codebook_size'],
            shared_codebook=rqvae_config['shared_codebook'],
            kmeans_method=rqvae_config['kmeans_method'],
            kmeans_iters=rqvae_config['kmeans_iters'],
            distances_method=rqvae_config['distances_method'],
            loss_beta=rqvae_config['loss_beta'],
            device=args.device
        )
        
        # 添加用于处理语义ID的嵌入层
        self.semantic_id_embedding = torch.nn.Embedding(
            num_embeddings=sum(rqvae_config['codebook_size']),  # 总的语义ID数量
            embedding_dim=args.hidden_units
        )
        
    def forward_with_rqvae(self, user_item, pos_seqs, neg_seqs, mask, next_mask, 
                          next_action_type, seq_feature, pos_feature, neg_feature, 
                          multimodal_emb):
        """
        结合RQ-VAE的前向传播
        
        Args:
            user_item: 用户序列ID
            pos_seqs: 正样本序列ID
            neg_seqs: 负样本序列ID
            mask: token类型掩码
            next_mask: 下一个token类型掩码
            next_action_type: 下一个token动作类型
            seq_feature: 序列特征
            pos_feature: 正样本特征
            neg_feature: 负样本特征
            multimodal_emb: 多模态嵌入
        """
        # 使用RQ-VAE处理多模态嵌入
        x_hat, semantic_id_list, recon_loss, rqvae_loss, total_loss = self.rqvae(multimodal_emb)
        
        # 将语义ID转换为嵌入向量
        semantic_embeddings = self.semantic_id_embedding(semantic_id_list)
        
        # 调用BaselineModel的前向传播
        pos_logits, neg_logits = super().forward(
            user_item, pos_seqs, neg_seqs, mask, next_mask, 
            next_action_type, seq_feature, pos_feature, neg_feature
        )
        
        # 返回原始logits以及RQ-VAE的损失
        return pos_logits, neg_logits, recon_loss, rqvae_loss, total_loss, semantic_embeddings
    
    def predict_with_rqvae(self, log_seqs, seq_feature, mask, multimodal_emb):
        """
        结合RQ-VAE的预测方法
        
        Args:
            log_seqs: 用户序列ID
            seq_feature: 序列特征
            mask: token类型掩码
            multimodal_emb: 多模态嵌入
        """
        # 使用RQ-VAE处理多模态嵌入
        x_hat, semantic_id_list, _, _, _ = self.rqvae(multimodal_emb)
        
        # 将语义ID转换为嵌入向量
        semantic_embeddings = self.semantic_id_embedding(semantic_id_list)
        
        # 调用BaselineModel的预测方法
        final_feat = super().predict(log_seqs, seq_feature, mask)
        
        # 返回用户序列表征以及语义嵌入
        return final_feat, semantic_embeddings
    
    def save_item_emb_with_rqvae(self, item_ids, retrieval_ids, feat_dict, save_path, 
                                multimodal_emb_dict, batch_size=1024):
        """
        生成候选库item embedding，结合RQ-VAE的语义ID
        
        Args:
            item_ids: 候选item ID
            retrieval_ids: 检索ID
            feat_dict: 特征字典
            save_path: 保存路径
            multimodal_emb_dict: 多模态嵌入字典
            batch_size: 批次大小
        """
        all_embs = []
        all_semantic_ids = []
        
        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings with RQ-VAE"):
            end_idx = min(start_idx + batch_size, len(item_ids))
            
            # 获取批次数据
            batch_item_ids = item_ids[start_idx:end_idx]
            batch_feat = [feat_dict[i] for i in range(start_idx, end_idx)]
            
            # 获取多模态嵌入
            batch_multimodal_emb = torch.stack([
                torch.tensor(multimodal_emb_dict[item_id], dtype=torch.float32) 
                for item_id in batch_item_ids
            ]).to(self.dev)
            
            # 使用RQ-VAE处理多模态嵌入
            _, semantic_id_list, _, _, _ = self.rqvae(batch_multimodal_emb)
            all_semantic_ids.append(semantic_id_list.detach().cpu().numpy())
            
            # 生成嵌入
            item_seq = torch.tensor(batch_item_ids, device=self.dev).unsqueeze(0)
            batch_emb = self.feat2emb(item_seq, [batch_feat], include_user=False).squeeze(0)
            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))
        
        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        final_semantic_ids = np.concatenate(all_semantic_ids, axis=0)
        
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))
        save_emb(final_semantic_ids, Path(save_path, 'semantic_ids.bin'))
