import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from pathlib import Path
from dataset import load_mm_emb
import json

# ===================== GAN实现 =====================
class GANGenerator(nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_channels:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GANDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_channels):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_channels:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GANModel(nn.Module):
    def __init__(self, input_dim, hidden_channels, latent_dim, num_classes, device):
        super().__init__()
        self.generator = GANGenerator(input_dim, hidden_channels, latent_dim).to(device)
        self.discriminator = GANDiscriminator(latent_dim, hidden_channels[::-1]).to(device)
        self.num_classes = num_classes
        self.device = device
        self.classifier = nn.Linear(latent_dim, num_classes).to(device)

    def forward(self, x):
        # x: [batch, input_dim]
        z_fake = self.generator(x)
        logits = self.classifier(z_fake)  # [batch, num_classes]
        semantic_id = F.gumbel_softmax(logits, tau=1, hard=True)
        d_fake = self.discriminator(z_fake)
        return z_fake, semantic_id, d_fake

    def discriminate(self, z):
        return self.discriminator(z)

    def get_semantic_id(self, x):
        with torch.no_grad():
            z_fake = self.generator(x)
            logits = self.classifier(z_fake)
            semantic_id = torch.argmax(logits, dim=-1)
        return semantic_id

# ===================== 兼容原接口的包装类 =====================
class GANQuantizerWrapper:
    """
    兼容原RQVAE接口的包装类，便于主流程无缝替换。
    """
    def __init__(self, input_dim, hidden_channels, latent_dim, num_classes, device):
        self.gan = GANModel(input_dim, hidden_channels, latent_dim, num_classes, device)
        self.device = device

    def fit(self, x_real, optimizer_g, optimizer_d, n_steps=1000):
        print(f'[DEBUG] GANQuantizerWrapper.fit called, x_real.shape={x_real.shape}, n_steps={n_steps}')
        # x_real: [batch, input_dim]
        for step in range(n_steps):
            # 1. 训练判别器
            z_fake, _, d_fake = self.gan(x_real)
            d_real = self.gan.discriminate(x_real)
            d_loss = torch.mean(F.relu(1. - d_real)) + torch.mean(F.relu(1. + d_fake))
            optimizer_d.zero_grad()
            d_loss.backward(retain_graph=True)
            optimizer_d.step()
            # 2. 训练生成器
            z_fake, _, d_fake = self.gan(x_real)
            g_loss = -torch.mean(d_fake)
            # 判别性对比损失
            z_enc = self.gan.generator(x_real)
            contrastive = nt_xent_loss(z_fake, z_enc)
            # 谱聚类loss
            spec_loss = spectral_loss(z_fake, n_clusters=8)
            # AutoEncoder重构loss
            if hasattr(self.gan, 'encoder') and hasattr(self.gan, 'decoder'):
                z_e = self.gan.encoder(x_real)
                x_recon = self.gan.decoder(z_e)
                recon_loss = F.mse_loss(x_recon, x_real)
                total_loss = g_loss + 0.1 * contrastive + 0.1 * spec_loss + 0.1 * recon_loss
            else:
                total_loss = g_loss + 0.1 * contrastive + 0.1 * spec_loss
            optimizer_g.zero_grad()
            total_loss.backward()
            optimizer_g.step()
            if step % 100 == 0:
                print(f"[DEBUG] Step {step}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}, Contrastive={contrastive.item():.4f}, Spectral={spec_loss.item():.4f}")

    def get_semantic_id(self, x):
        print(f'[DEBUG] GANQuantizerWrapper.get_semantic_id called, x.shape={x.shape}')
        return self.gan.get_semantic_id(x)

    def forward(self, x):
        return self.gan(x)

# ===================== 用法示例 =====================
# 假设原RQVAE用法如下：
# rqvae = RQVAE(...)
# semantic_id = rqvae._get_codebook(x)
# 替换为：
# gan_quantizer = GANQuantizerWrapper(input_dim, hidden_channels, latent_dim, num_classes, device)
# semantic_id = gan_quantizer.get_semantic_id(x)

# ===================== 下面保留原RQ-VAE代码以便参考 =====================
"""
选手可参考以下流程，使用提供的 RQ-VAE 框架代码将多模态emb数据转换为Semantic Id:
1. 使用 MmEmbDataset 读取不同特征 ID 的多模态emb数据.
2. 训练 RQ-VAE 模型, 训练完成后将数据转换为Semantic Id.
3. 参照 Item Sparse 特征格式处理Semantic Id，作为新特征加入Baseline模型训练.
"""

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

class MmEmbDataset(torch.utils.data.Dataset):
    """
    Build Dataset for RQ-VAE Training - 按需加载版本

    Args:
        data_dir = os.environ.get('TRAIN_DATA_PATH')
        feature_id = MM emb ID
    """

    def __init__(self, data_dir, feature_id):
        print(f'[DEBUG] MmEmbDataset.__init__ called with data_dir={data_dir}, feature_id={feature_id}')
        super().__init__()
        try:
            self.data_dir = Path(data_dir)
            self.feature_id = feature_id
            self.emb_dir = Path(data_dir, "creative_emb", f"emb_{feature_id}")
            
            # 检查embedding目录是否存在
            if not self.emb_dir.exists():
                raise ValueError(f"Embedding directory {self.emb_dir} does not exist")
            
            # 获取所有JSON文件
            self.json_files = list(self.emb_dir.glob("*.json"))
            if not self.json_files:
                raise ValueError(f"No JSON files found in {self.emb_dir}")
            
            print(f'[DEBUG] Found {len(self.json_files)} JSON files in {self.emb_dir}')
            
            # 创建文件索引
            self.file_index = []
            self.total_items = 0
            
            # 快速扫描所有文件，建立索引
            for json_file in self.json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            file_items = len(data)
                            self.file_index.append({
                                'file': json_file,
                                'start_idx': self.total_items,
                                'end_idx': self.total_items + file_items,
                                'data': None  # 延迟加载
                            })
                            self.total_items += file_items
                except Exception as e:
                    print(f'[WARNING] Failed to process {json_file}: {e}')
                    continue
            
            print(f'[DEBUG] Total items indexed: {self.total_items}')
            
            # 验证数据完整性
            if self.total_items > 0:
                sample_tid, sample_emb = self.__getitem__(0)
                print(f'[DEBUG] Sample item: tid={sample_tid}, emb_shape={sample_emb.shape}')
                
        except Exception as e:
            print(f'[ERROR] Failed to initialize MmEmbDataset: {e}')
            import traceback
            traceback.print_exc()
            # 创建一个空的dataset作为fallback
            self.file_index = []
            self.total_items = 0
            print('[WARNING] Created empty MmEmbDataset as fallback')

    def _load_file_data(self, file_info):
        """延迟加载文件数据"""
        if file_info['data'] is None:
            try:
                with open(file_info['file'], 'r') as f:
                    file_info['data'] = json.load(f)
                print(f'[DEBUG] Loaded file: {file_info["file"]}')
            except Exception as e:
                print(f'[ERROR] Failed to load file {file_info["file"]}: {e}')
                file_info['data'] = {}
        return file_info['data']

    def _find_file_for_index(self, index):
        """找到包含指定索引的文件"""
        for file_info in self.file_index:
            if file_info['start_idx'] <= index < file_info['end_idx']:
                return file_info, index - file_info['start_idx']
        raise IndexError(f"Index {index} out of range")

    def __getitem__(self, index):
        try:
            file_info, local_index = self._find_file_for_index(index)
            data = self._load_file_data(file_info)
            
            # 获取数据
            items = list(data.items())
            if local_index >= len(items):
                raise IndexError(f"Local index {local_index} out of range for file {file_info['file']}")
            
            tid, emb = items[local_index]
            
            # 确保emb是tensor
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float32)
            
            return tid, emb
            
        except Exception as e:
            print(f'[ERROR] Failed to get item {index}: {e}')
            # 返回一个默认值
            return 0, torch.zeros(32, dtype=torch.float32)  # 假设维度为32

    def __len__(self):
        return self.total_items

    @staticmethod
    def collate_fn(batch):
        try:
            tids, embs = zip(*batch)
            # 确保所有embedding的维度一致
            emb_tensors = []
            for emb in embs:
                if isinstance(emb, torch.Tensor):
                    emb_tensors.append(emb)
                else:
                    emb_tensors.append(torch.tensor(emb, dtype=torch.float32))
            
            # 检查维度一致性
            if len(emb_tensors) > 0:
                target_dim = emb_tensors[0].shape[-1]
                for i, emb in enumerate(emb_tensors):
                    if emb.shape[-1] != target_dim:
                        print(f'[WARNING] Embedding {i} has different dimension: {emb.shape} vs expected {target_dim}')
                        # 调整维度
                        if emb.shape[-1] > target_dim:
                            emb_tensors[i] = emb[..., :target_dim]
                        else:
                            # 用零填充
                            padding = torch.zeros(target_dim - emb.shape[-1], dtype=emb.dtype)
                            emb_tensors[i] = torch.cat([emb, padding])
            
            return tids, torch.stack(emb_tensors)
        except Exception as e:
            print(f'[ERROR] Failed in collate_fn: {e}')
            # 返回一个默认的batch
            batch_size = len(batch)
            return [0] * batch_size, torch.zeros(batch_size, 32, dtype=torch.float32)  # 假设维度为32


## Kmeans
def kmeans(data, n_clusters, kmeans_iters):
    """
    auto init: n_init = 10 if n_clusters <= 10 else 1
    """
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
        if self.kmeans_method == 'kmeans':
            _codebook, _ = kmeans(data, self.num_clusters, self.kmeans_iters)
        elif self.kmeans_method == 'bkmeans':
            BKmeans = BalancedKmeans(
                num_clusters=self.num_clusters, kmeans_iters=self.kmeans_iters, tolerance=1e-4, device=self.device
            )
            _codebook, _ = BKmeans.fit(data)
        else:
            _codebook = torch.randn(self.num_clusters, self.codebook_emb_dim)
        _codebook = _codebook.to(self.device)
        assert _codebook.shape == (self.num_clusters, self.codebook_emb_dim)
        self.codebook = torch.nn.Parameter(_codebook)

    @torch.no_grad()
    def _compute_distances(self, data):
        _codebook_t = self.codebook.t()
        assert _codebook_t.shape == (self.codebook_emb_dim, self.num_clusters)
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
        self.encoder = RQEncoder(input_dim, hidden_channels, latent_dim).to(device)
        self.decoder = RQDecoder(latent_dim, hidden_channels[::-1], input_dim).to(device)
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
        return self.encoder(x)

    def decode(self, z_vq):
        if isinstance(z_vq, list):
            z_vq = z_vq[-1]
        return self.decoder(z_vq)

    def compute_loss(self, x_hat, x_gt, rqvae_loss):
        recon_loss = F.mse_loss(x_hat, x_gt, reduction="mean")
        total_loss = recon_loss + rqvae_loss
        return recon_loss, rqvae_loss, total_loss

    def _get_codebook(self, x_gt):
        z_e = self.encode(x_gt)
        vq_emb_list, semantic_id_list, rqvae_loss = self.rq(z_e)
        return semantic_id_list

    def forward(self, x_gt):
        z_e = self.encode(x_gt)
        vq_emb_list, semantic_id_list, rqvae_loss = self.rq(z_e)
        x_hat = self.decode(vq_emb_list)
        recon_loss, rqvae_loss, total_loss = self.compute_loss(x_hat, x_gt, rqvae_loss)
        return x_hat, semantic_id_list, recon_loss, rqvae_loss, total_loss


def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(logits, labels)

def spectral_loss(emb, n_clusters):
    sim = torch.mm(emb, emb.t())
    D = torch.diag(sim.sum(1))
    L = D - sim
    eigvals, eigvecs = torch.linalg.eigh(L)
    spectral_emb = eigvecs[:, :n_clusters]
    return torch.var(spectral_emb, dim=0).mean()
