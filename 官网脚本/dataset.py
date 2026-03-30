import json
import pickle
import struct
from pathlib import Path

import os
import numpy as np
import torch
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID # 这个是什么？
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型 # 这几种不同类型是什么意思？
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        # 优化内存：将控制项提前设置，供 _load_data_and_offsets 使用
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id
        # 是否缓存用户数据（高内存占用点，可关闭以节省内存）
        self.use_user_cache = getattr(args, 'cache_user_data', False)
        # 是否懒加载 item 特征（高内存占用点，建议开启以节省内存）
        self.lazy_item_feat = getattr(args, 'lazy_item_feat', True)
        # 文件名在不同数据集（训练/测试）下不同，由各自的 _load_data_and_offsets 设定
        # 仅当子类未预先设置时，才使用默认文件名，避免覆盖子类自定义（如 MyTestDataset 的 predict_* 文件）
        if not hasattr(self, 'data_file_name'):
            self.data_file_name = 'seq.jsonl'
        if not hasattr(self, 'offsets_file_name'):
            self.offsets_file_name = 'seq_offsets.pkl'

        self._load_data_and_offsets()

        # 高内存占用点：一次性加载全量 item_feat_dict 到内存，量级大时会吃满内存。
        # 优化：默认开启 lazy_item_feat，仅在需要时用默认值/按需加载，避免常驻大字典。
        if not self.lazy_item_feat:
            self.item_feat_dict = {}
            with open(Path(data_dir, "item_feat_dict.json"), 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    for k, v in item.items():
                        self.item_feat_dict[k] = v
                    if len(self.item_feat_dict) % 10000 == 0:  # 每加载10000个特征打印一次进度
                        print(f"Loaded {len(self.item_feat_dict)} item features")
        else:
            # 懒加载模式：不常驻 item_feat_dict，负样本特征用默认值+按需的多模态向量
            self.item_feat_dict = None
            print("[Dataset] lazy_item_feat=True: 不加载 item_feat_dict，负样本特征将使用默认值并按需加载多模态特征")
        
        # 延迟加载多模态特征
        self.mm_emb_dict = {}
        self.mm_emb_path = Path(data_dir, "creative_emb")
        self.mm_emb_ids = args.mm_emb_id
        # 缓存大小与阈值（可通过环境变量 MM_CACHE_MAX_BYTES 配置，默认4GB）
        self.mm_cache_bytes = 0
        try:
            self.mm_cache_max_bytes = int(os.environ.get('MM_CACHE_MAX_BYTES', str(4 * 1024 ** 3)))
        except Exception:
            self.mm_cache_max_bytes = 4 * 1024 ** 3

        # 加载索引器
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])  # 物品数
            self.usernum = len(indexer['u'])  # 用户数
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        # 加载偏移量
        with open(Path(self.data_dir, self.offsets_file_name), 'rb') as f:
            self.seq_offsets = pickle.load(f)
        
        # 高内存占用点：预加载所有用户数据到内存，避免多进程重复读取文件，但会占用大量内存。
        # 优化：通过 args.cache_user_data 控制，默认关闭，按需打开。
        if self.use_user_cache:
            self.user_data_cache = {}
            with open(self.data_dir / self.data_file_name, 'rb') as data_file:
                for uid in range(len(self.seq_offsets)):
                    data_file.seek(self.seq_offsets[uid])
                    line = data_file.readline()
                    self.user_data_cache[uid] = json.loads(line)
        else:
            self.user_data_cache = None

    def _load_user_data(self, uid):
        """
        从缓存/文件加载单个用户的数据
        """
        if self.user_data_cache is not None:
            return self.user_data_cache[uid]
        # 按需从文件定位读取，避免将所有用户数据常驻内存
        with open(self.data_dir / self.data_file_name, 'rb') as data_file:
            data_file.seek(self.seq_offsets[uid])
            line = data_file.readline()
            return json.loads(line)

    def _random_neqs(self, l, r, s, num_neg):
        """
        一次性生成num_neg個不在序列s中的隨機整數，用於多負樣本采樣
        Args:
            l: 隨機整數的最小值
            r: 隨機整數的最大值
            s: 用戶已交互item集合
            num_neg: 負樣本數量
        Returns:
            neg_list: 長度為num_neg的負樣本id列表
        """
        neg_list = []
        tries = 0
        while len(neg_list) < num_neg:
            t = np.random.randint(l, r)
            # 原逻辑依赖 self.item_feat_dict 检查有效性，内存大。懒加载时放宽为仅避开正样本集合。
            membership_ok = True if self.item_feat_dict is None else (str(t) in self.item_feat_dict)
            if (t not in s) and membership_ok:
                neg_list.append(t)
            tries += 1
            if tries > num_neg * 10:  # 防止死循環
                # 補齊
                remain = num_neg - len(neg_list)
                all_items = set(range(l, r)) - set(s)
                if self.item_feat_dict is not None:
                    all_items = [i for i in all_items if str(i) in self.item_feat_dict]
                else:
                    all_items = list(all_items)
                if len(all_items) >= remain:
                    neg_list.extend(np.random.choice(list(all_items), remain, replace=False))
                else:
                    neg_list.extend(np.random.choice(list(all_items), remain, replace=True))
                break
        return neg_list

    def __getitem__(self, uid):
        """
        獲取單個用戶的數據，並進行padding處理，生成模型需要的數據格式
        """
        user_sequence = self._load_user_data(uid)  # 動態加載用戶數據

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1, 10], dtype=np.int32)  # [maxlen+1, 10]
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1, 10], dtype=object)  # [maxlen+1, 10]

        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])  # 該用戶交互過的item列表

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                neg_ids = self._random_neqs(1, self.itemnum + 1, ts, 10)
                neg[idx] = neg_ids
                # 懒加载时不访问大字典，直接使用默认特征并按需加载多模态
                if self.item_feat_dict is not None:
                    neg_feat[idx] = [self.fill_missing_feat(self.item_feat_dict.get(str(neg_id), None), neg_id) for neg_id in neg_ids]
                else:
                    neg_feat[idx] = [self.fill_missing_feat(None, neg_id) for neg_id in neg_ids]
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        # neg_feat: [maxlen+1, 10]，需補全
        for i in range(neg_feat.shape[0]):
            for j in range(neg_feat.shape[1]):
                if neg_feat[i, j] is None:
                    neg_feat[i, j] = self.feature_default_value

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100',
            '117',
            '111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = [] # 说明数据里没有给这部分的特征吗？
        feat_types['item_continual'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        # 为多模态特征设置默认值，使用静态维度映射，避免访问尚未加载的mm_emb_dict导致KeyError
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

        return feat_default_value, feat_types, feat_statistics

    def _load_mm_emb_for_id(self, feat_id, item_id):
        """
        延迟加载指定特征ID和物品ID的多模态特征
        """
        if feat_id not in self.mm_emb_dict:
            self.mm_emb_dict[feat_id] = {}
        
        if item_id not in self.mm_emb_dict[feat_id]:
            emb_file = self.mm_emb_path / f"{feat_id}_{item_id}.npy"
            if emb_file.exists():
                # 使用 memmap 降低峰值内存
                arr = np.load(emb_file, mmap_mode='r')
                self.mm_emb_dict[feat_id][item_id] = arr
                # 估算并累计缓存大小
                try:
                    self.mm_cache_bytes += int(arr.nbytes)
                except Exception:
                    try:
                        self.mm_cache_bytes += int(arr.size * arr.dtype.itemsize)
                    except Exception:
                        pass
                # 超过阈值则基于大小清理缓存
                if self.mm_cache_bytes > self.mm_cache_max_bytes:
                    self._clear_mm_emb_cache()
            else:
                return None
        
        return self.mm_emb_dict[feat_id][item_id]

    def _clear_mm_emb_cache(self, keep_ids=None):
        """
        清理多模态特征缓存，只保留指定的ID
        """
        if keep_ids is None:
            self.mm_emb_dict.clear()
            self.mm_cache_bytes = 0
            return
        
        for feat_id in list(self.mm_emb_dict.keys()):
            if feat_id in self.mm_emb_dict:
                curr_dict = self.mm_emb_dict[feat_id]
                self.mm_emb_dict[feat_id] = {k: v for k, v in curr_dict.items() if k in keep_ids}
        
        # 重新计算缓存大小
        total_bytes = 0
        for feat_id in self.mm_emb_dict:
            for v in self.mm_emb_dict[feat_id].values():
                try:
                    total_bytes += int(v.nbytes)
                except Exception:
                    try:
                        total_bytes += int(v.size * v.dtype.itemsize)
                    except Exception:
                        pass
        self.mm_cache_bytes = total_bytes

    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        
        # 处理多模态特征
        if item_id != 0:
            orig_id = self.indexer_i_rev[item_id]
            for feat_id in self.feature_types['item_emb']:
                emb = self._load_mm_emb_for_id(feat_id, orig_id)
                if emb is not None and isinstance(emb, np.ndarray):
                    filled_feat[feat_id] = emb
        
        # 基于缓存大小的清理，超过阈值时触发
        if getattr(self, 'mm_cache_bytes', 0) > getattr(self, 'mm_cache_max_bytes', 0):
            self._clear_mm_emb_cache()

        return filled_feat

    def feat2tensor(self, seq_feature, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [batch_size, maxlen, max_array_len(if array)]
        """
        batch_size = len(seq_feature)

        if k in self.feature_types['item_array'] or k in self.feature_types['user_array']:
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

            return torch.from_numpy(batch_data)
        else:
            # 如果特征是Sparse类型或Continual类型
            is_sparse = (k in self.feature_types.get('item_sparse', [])) or (k in self.feature_types.get('user_sparse', []))
            dtype = np.int64 if is_sparse else np.float32
            batch_data = np.zeros((batch_size, len(seq_feature[0])), dtype=dtype)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                if is_sparse:
                    # Embedding 需要整数索引
                    batch_data[i, :len(seq_data)] = np.asarray(seq_data, dtype=np.int64)
                else:
                    batch_data[i, :len(seq_data)] = np.asarray(seq_data, dtype=np.float32)

            return torch.from_numpy(batch_data)

    @staticmethod
    def collate_fn(batch):
        """
        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            pos: 正样本ID, torch.Tensor形式
            neg: 负样本ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            next_token_type: 下一个token类型, torch.Tensor形式
            next_action_type: 下一个行为类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            pos_feat: 正样本特征, list形式
            neg_feat: 负样本特征, list形式
        """
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        # 使用 np.asarray 避免不必要的复制
        seq = torch.from_numpy(np.asarray(seq))
        pos = torch.from_numpy(np.asarray(pos))
        neg = torch.from_numpy(np.asarray(neg))
        token_type = torch.from_numpy(np.asarray(token_type))
        next_token_type = torch.from_numpy(np.asarray(next_token_type))
        next_action_type = torch.from_numpy(np.asarray(next_action_type))
        seq_feat = list(seq_feat)
        pos_feat = list(pos_feat)
        neg_feat = list(neg_feat)
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat


class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        # 智能文件名检测：优先使用 predict_seq.jsonl，不存在则回退到 seq.jsonl
        data_dir_path = Path(data_dir)
        
        # 检测数据文件
        if (data_dir_path / 'predict_seq.jsonl').exists():
            self.data_file_name = 'predict_seq.jsonl'
            print("[MyTestDataset] 使用 predict_seq.jsonl")
        elif (data_dir_path / 'seq.jsonl').exists():
            self.data_file_name = 'seq.jsonl'
            print("[MyTestDataset] predict_seq.jsonl 不存在，回退到 seq.jsonl")
        else:
            raise FileNotFoundError(f"在 {data_dir} 中找不到 predict_seq.jsonl 或 seq.jsonl")
            
        # 检测偏移文件
        if (data_dir_path / 'predict_seq_offsets.pkl').exists():
            self.offsets_file_name = 'predict_seq_offsets.pkl'
            print("[MyTestDataset] 使用 predict_seq_offsets.pkl")
        elif (data_dir_path / 'seq_offsets.pkl').exists():
            self.offsets_file_name = 'seq_offsets.pkl'
            print("[MyTestDataset] predict_seq_offsets.pkl 不存在，回退到 seq_offsets.pkl")
        else:
            raise FileNotFoundError(f"在 {data_dir} 中找不到 predict_seq_offsets.pkl 或 seq_offsets.pkl")
            
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        # 使用检测到的文件名加载偏移量
        with open(Path(self.data_dir, self.offsets_file_name), 'rb') as f:
            self.seq_offsets = pickle.load(f)
        
        # 根据内存优化选项决定是否预加载
        if getattr(self, 'use_user_cache', False):
            self.user_data_cache = {}
            with open(self.data_dir / self.data_file_name, 'rb') as data_file:
                for uid in range(len(self.seq_offsets)):
                    data_file.seek(self.seq_offsets[uid])
                    line = data_file.readline()
                    self.user_data_cache[uid] = json.loads(line)
        else:
            self.user_data_cache = None

    def _process_cold_start_feat(self, feat):
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

    def __getitem__(self, uid):
        # 其余逻辑继承父类实现
        return super().__getitem__(uid)

    def __len__(self):
        return super().__len__()

    @staticmethod
    def collate_fn(batch):
        return MyDataset.collate_fn(batch)


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions)) # struct.pack的作用？
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('*.json'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict