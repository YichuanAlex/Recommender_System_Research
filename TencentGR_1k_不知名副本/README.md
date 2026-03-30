# TencentGR_1k 项目运行指南

## 项目结构

```
TencentGR_1k/
├── README.md
├── TencentGR_1k\
│   ├── README.md
│   ├── creative_emb\
│   │   ├── emb_81_32\
│   │   ├── emb_82_1024\
│   │   ├── emb_83_3584\
│   │   ├── emb_84_4096\
│   │   ├── emb_85_3584\
│   │   └── emb_86_1024\
│   ├── indexer.pkl
│   ├── item_feat_dict.json
│   ├── predict_seq.jsonl
│   ├── seq.jsonl
│   └── seq_offsets.pkl
├── __pycache__\
├── checkpoints\
├── dataset.py
├── infer.py
├── inference_results\
├── logs\
├── main.py
├── model.py
├── record.txt
├── requirements.txt
├── run.sh
├── run_inference.py
├── start_training.py
└── tf_events\
```

## 环境准备

1. 确保已安装 Python 3.10 或更高版本。
2. 安装项目依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 运行项目

为了避免直接运行 `main.py` 时因缺少环境变量或数据文件而报错，我们提供了以下几种方式来启动项目：

### 方法一：使用 Python 脚本启动（推荐）

直接运行我们提供的 `start_training.py` 脚本，它会自动设置必要的环境变量并启动训练：

```bash
python start_training.py
```

### 方法二：使用批处理脚本启动（Windows）

在 Windows 系统上，可以运行 `run.sh` 脚本：

```cmd
bash run.sh
```

### 方法三：使用 Shell 脚本启动（Linux/macOS）

在 Linux 或 macOS 系统上，可以运行 `run.sh` 脚本：

```bash
bash run.sh
```

## 手动设置环境变量

如果需要手动设置环境变量，请确保设置了以下变量：

- `TRAIN_DATA_PATH`: 训练数据路径，默认为 `TencentGR_1k`
- `TRAIN_LOG_PATH`: 训练日志路径，默认为 `logs`
- `TRAIN_TF_EVENTS_PATH`: TensorBoard 事件路径，默认为 `tf_events`
- `TRAIN_CKPT_PATH`: 模型检查点保存路径，默认为 `checkpoints`

设置完成后，可以直接运行 `main.py`：

```bash
python main.py
```

## 推理过程

项目包含一个推理脚本 `run_inference.py`，用于生成推荐结果。推理过程包括以下步骤：

1. 加载训练好的模型
2. 处理测试数据
3. 生成用户和候选项目的嵌入向量
4. 使用FAISS库进行近似最近邻检索
5. 生成最终的推荐列表

### 运行推理

```bash
python run_inference.py
```

推理结果将保存在 `inference_results` 目录下。

## 代码修改说明

### 1. 数据集处理 (`dataset.py`)

- 修复了 `MyTestDataset` 类中 `user_id` 变量未定义的问题，在循环前初始化 `user_id = None`。
- 优化了多模态特征加载逻辑，添加了对缺失嵌入向量的处理。

### 2. 推理过程 (`infer.py`)

- 修复了 `get_candidate_emb` 函数中处理 `predict_seq.jsonl` 文件的逻辑，正确处理JSON数组格式。
- 添加了重复项目检查，避免重复处理相同项目。
- 修改了 `read_result_ids` 函数，使其能够正确处理空的 `id100.u64bin` 文件。
- 移除了对外部FAISS demo可执行文件的依赖，直接使用FAISS库进行ANN检索。
- 在 `requirements.txt` 中添加了 `faiss-cpu==1.9.0` 依赖。
- 实现了完整的FAISS检索流程，包括索引创建、参数设置和结果保存。
- 修复了加载候选库嵌入向量时维度不匹配的问题，现在会正确从文件头部读取嵌入向量的维度。

### 3. 环境配置

- 在 `start_training.py` 中设置了 `PYTHONPATH` 环境变量，确保Python能找到项目模块。
- 在 `run_inference.py` 中设置了必要的环境变量，确保推理过程能正确运行。

## 迁移到其他服务器

要将项目迁移到其他服务器并顺利运行，需要修改以下内容：

1. **环境变量**：
   - 检查并更新 `start_training.py` 和 `run_inference.py` 中的环境变量设置，确保路径正确。
   - 如果数据文件位置不同，需要相应修改 `TRAIN_DATA_PATH` 等环境变量。

2. **数据文件路径**：
   - 确保 `TencentGR_1k` 目录及其子目录中的所有数据文件都已正确复制到新服务器。
   - 检查 `creative_emb` 目录中的多模态特征嵌入文件是否完整。

3. **依赖安装**：
   - 在新服务器上运行 `pip install -r requirements.txt` 安装所有依赖。
   - 确保新服务器上安装了正确版本的Python（3.10或更高版本）。

4. **FAISS库**：
   - 确保新服务器上能够正确安装 `faiss-cpu==1.9.0`。
   - 如果需要GPU加速，可以安装 `faiss-gpu` 版本，并相应修改代码。

5. **模型检查点**：
   - 如果需要在新服务器上继续训练，确保 `checkpoints` 目录中的模型文件已正确复制。

6. **推理结果目录**：
   - 确保 `inference_results` 目录存在且有写入权限。

通过以上步骤，项目应该能够在新服务器上顺利运行。

## 注意事项

1. 项目依赖于 `TencentGR_1k` 目录下的数据文件。请确保 `TencentGR_1k` 目录结构完整且包含所有必要的数据文件。
2. 对于多模态特征嵌入文件（`creative_emb` 目录下的文件），项目会自动处理 `.pkl` 和 `.json` 格式的文件。如果缺少 `.pkl` 文件，项目会尝试从 `.json` 文件加载数据。
3. 推理过程中使用的FAISS库需要足够的内存来加载候选库的嵌入向量，请确保服务器有足够的内存资源。
4. 项目可能遇到NumPy版本兼容性问题，特别是与FAISS库一起使用时。如果出现`numpy.core.multiarray failed to import`错误，请尝试将NumPy版本降级到1.x系列，例如通过`pip install numpy==1.24.4`命令。
5. 如果在推理过程中遇到`cannot reshape array of size 130 into shape (32)`错误，这通常是由于多模态特征嵌入维度不匹配导致的。请检查`creative_emb`目录中的特征文件是否完整，并确保特征维度与代码中的定义一致（示例数据集特征ID 81-86对应的维度分别为32, 1024, 3584, 4096, 3584, 1024）。