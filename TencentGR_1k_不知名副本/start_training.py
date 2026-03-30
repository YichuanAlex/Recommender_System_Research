import os
import subprocess
import sys
from pathlib import Path

# 设置默认的环境变量
os.environ.setdefault('TRAIN_DATA_PATH', 'TencentGR_1k')
os.environ.setdefault('TRAIN_LOG_PATH', 'logs')
os.environ.setdefault('TRAIN_TF_EVENTS_PATH', 'tf_events')
os.environ.setdefault('TRAIN_CKPT_PATH', 'checkpoints')

# 确保必要的目录存在
Path(os.environ['TRAIN_LOG_PATH']).mkdir(parents=True, exist_ok=True)
Path(os.environ['TRAIN_TF_EVENTS_PATH']).mkdir(parents=True, exist_ok=True)
Path(os.environ['TRAIN_CKPT_PATH']).mkdir(parents=True, exist_ok=True)

# 调用主训练脚本
subprocess.run([sys.executable, 'main.py'] + sys.argv[1:])