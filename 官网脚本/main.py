import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
import gc

# AMP compatibility layer to support both old (torch.cuda.amp) and new (torch.amp) APIs
try:
    from torch import amp as torch_amp  # PyTorch >= 2.0 preferred API
    _HAVE_TORCH_AMP = True
except Exception:  # pragma: no cover
    torch_amp = None
    _HAVE_TORCH_AMP = False


def amp_autocast(device_type: str, dtype, enabled: bool):
    if _HAVE_TORCH_AMP and hasattr(torch_amp, "autocast"):
        return torch_amp.autocast(device_type=device_type, dtype=dtype, enabled=enabled)
    else:
        # Older API does not accept device_type
        return torch.cuda.amp.autocast(dtype=dtype, enabled=enabled)


def create_grad_scaler(enabled: bool):
    if _HAVE_TORCH_AMP and hasattr(torch_amp, "GradScaler"):
        try:
            # New API prefers specifying device as positional arg
            return torch_amp.GradScaler('cuda', enabled=enabled)
        except TypeError:
            # Fallback to older signature
            return torch_amp.GradScaler(enabled=enabled)
    else:
        return torch.cuda.amp.GradScaler(enabled=enabled)

from dataset import MyDataset
from model import BaselineModel


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--l2_emb', default=0.0001, type=float)

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    # Memory-optimization: gradient accumulation steps (does not change batch_size)
    parser.add_argument('--accumulation_steps', default=1, type=int)

    # Optional: control torch.compile (PyTorch 2.x). Disabled by default to avoid Dynamo recompilation overhead on dynamic data paths.
    parser.add_argument('--enable_compile', action='store_true')
    parser.add_argument('--compile_mode', default='reduce-overhead', choices=['default', 'reduce-overhead', 'max-autotune'])

    # Dataset/Dataloader memory knobs
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    pin_group = parser.add_mutually_exclusive_group()
    pin_group.add_argument('--pin_memory', dest='pin_memory', action='store_true')
    pin_group.add_argument('--no_pin_memory', dest='pin_memory', action='store_false')
    parser.set_defaults(pin_memory=True)

    parser.add_argument('--cache_user_data', action='store_true', help='预加载用户数据到内存，减少文件随机访问（高内存占用，默认关闭）')
    lazy_group = parser.add_mutually_exclusive_group()
    lazy_group.add_argument('--lazy_item_feat', dest='lazy_item_feat', action='store_true')
    lazy_group.add_argument('--no_lazy_item_feat', dest='lazy_item_feat', action='store_false')
    parser.set_defaults(lazy_item_feat=True)

    # Gradient checkpointing control
    ckpt_group = parser.add_mutually_exclusive_group()
    ckpt_group.add_argument('--use_checkpoint', dest='use_checkpoint', action='store_true')
    ckpt_group.add_argument('--no_checkpoint', dest='use_checkpoint', action='store_false')
    # 新增：负样本流式chunk大小（越大越快，但峰值显存越高）
    parser.add_argument('--neg_chunk_size', type=int, default=1)
    parser.set_defaults(use_checkpoint=True)

    args = parser.parse_args()

    # auto-select device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[Warning] CUDA not available, fallback to CPU.")
        args.device = 'cpu'
    
    return args


def main():
    args = get_args()

    # Enable TF32 for better throughput and lower memory pressure
    if args.device == 'cuda' and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'a')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))

    data_path = os.environ.get('TRAIN_DATA_PATH')

    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    # Dataloader knobs
    persistent_workers = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=args.pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=args.pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    # torch.compile is opt-in below via --enable_compile to avoid overhead on dynamic code paths
    # Optional: PyTorch 2.0+ graph compile (opt-in)
    try:
        if hasattr(torch, 'compile') and args.enable_compile:
            print(f"[Info] Enabling torch.compile with mode='{args.compile_mode}'")
            model = torch.compile(model, mode=args.compile_mode)
    except Exception as e:
        print(f"[Info] torch.compile disabled due to error: {e}")

    # Initialize parameters (best effort)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    # Set special embeddings' zero index to zero
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0
    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
            print(f"Loaded checkpoint from {args.state_dict_path}, start from epoch {epoch_start_idx}")
        except Exception as e:
            print(f'Failed loading state_dicts from {args.state_dict_path}: {e}')
            raise RuntimeError('Failed loading state_dicts!')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # AMP setup: prefer bfloat16 when supported
    scaler_enabled = (args.device == 'cuda' and torch.cuda.is_available())
    try:
        use_bf16 = torch.cuda.is_bf16_supported() if scaler_enabled and hasattr(torch.cuda, 'is_bf16_supported') else False
    except Exception:
        use_bf16 = False
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = create_grad_scaler(scaler_enabled)

    # Cosine schedule with warmup
    total_steps = args.num_epochs * math.ceil(len(train_loader) / args.accumulation_steps)
    warmup_steps = int(total_steps * 0.05)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    global_step = 0
    print("Start training")

    # Use non_blocking transfers when pinned and CUDA
    non_blocking = (args.device == 'cuda' and args.pin_memory and torch.cuda.is_available())

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        # monitor and tidy fragmentation
        if args.device == 'cuda' and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        model.train()
        if args.inference_only:
            break

        epoch_loss_sum = 0.0
        epoch_pos_sim_sum = 0.0
        epoch_neg_sim_sum = 0.0
        grad_norms = []

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch

            # ensure numpy internal types are float32 (best effort)
            try:
                if isinstance(seq_feat, np.ndarray) and seq_feat.dtype == object:
                    pass
                if isinstance(pos_feat, np.ndarray) and pos_feat.dtype == object:
                    pass
                if isinstance(neg_feat, np.ndarray) and neg_feat.dtype == object:
                    pass
            except Exception:
                pass

            seq = seq.to(args.device, non_blocking=non_blocking)
            pos = pos.to(args.device, non_blocking=non_blocking)
            neg = neg.to(args.device, non_blocking=non_blocking)
            token_type = token_type.to(args.device, non_blocking=non_blocking)
            next_token_type = next_token_type.to(args.device, non_blocking=non_blocking)

            # autocast forward without changing hyperparams/sampling/gradient semantics
            with amp_autocast('cuda', amp_dtype, scaler_enabled):
                loss, pos_sim, neg_sim = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, dataset
                )

                # gradient accumulation
                loss = loss / args.accumulation_steps

                # l2 regularization on embedding params
                for param in model.item_emb.parameters():
                    loss = loss + (args.l2_emb * torch.norm(param) / args.accumulation_steps)

            # backward
            if scaler_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # grad norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)

            # optimizer step when accumulation boundary or last step
            if ((step + 1) % args.accumulation_steps == 0) or ((step + 1) == len(train_loader)):
                if scaler_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # accumulate stats for logging
            epoch_loss_sum += loss.item() * args.accumulation_steps
            epoch_pos_sim_sum += pos_sim
            epoch_neg_sim_sum += neg_sim
            global_step += 1

            # periodically clear multimodal cache & cuda cache
            if (global_step % 100) == 0:
                try:
                    dataset._clear_mm_emb_cache()
                except Exception:
                    pass
                if args.device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            if global_step % 100 == 0:
                # tb scalars
                writer.add_scalar('Loss/train', loss.item() * args.accumulation_steps, global_step)
                writer.add_scalar('Gradient/Norm', total_norm, global_step)
                writer.add_scalar('Similarity/Positive', pos_sim, global_step)
                writer.add_scalar('Similarity/Negative', neg_sim, global_step)

                # logfile json
                log_json = json.dumps({
                    'global_step': global_step,
                    'loss': loss.item(),
                    'pos_sim': pos_sim,
                    'neg_sim': neg_sim,
                    'grad_norm': total_norm,
                    'epoch': epoch,
                    'time': time.time()
                })
                log_file.write(log_json + '\n')
                log_file.flush()
                print(log_json)

            # help GC: explicitly drop references
            del seq, pos, neg, token_type, next_token_type
            del seq_feat, pos_feat, neg_feat, next_action_type
            # loss tensor will be recreated next iteration; drop reference ASAP
            del loss
        
        avg_epoch_loss = epoch_loss_sum / len(train_loader)
        avg_pos_sim = epoch_pos_sim_sum / len(train_loader)
        avg_neg_sim = epoch_neg_sim_sum / len(train_loader)
        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        
        writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
        writer.add_scalar('Similarity/Positive_epoch', avg_pos_sim, epoch)
        writer.add_scalar('Similarity/Negative_epoch', avg_neg_sim, epoch)
        writer.add_scalar('Gradient/Norm_epoch', avg_grad_norm, epoch)
        
        print(f"Epoch {epoch} training loss: {avg_epoch_loss:.4f}")
        print(f"  Avg Positive Similarity: {avg_pos_sim:.4f}")
        print(f"  Avg Negative Similarity: {avg_neg_sim:.4f}")
        print(f"  Avg Gradient Norm: {avg_grad_norm:.4f}")

        # epoch end cleanup
        try:
            dataset._clear_mm_emb_cache()
        except Exception:
            pass
        if args.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        model.eval()
        valid_loss_sum = 0
        valid_pos_sim_sum = 0
        valid_neg_sim_sum = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Validation Epoch {epoch}"):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                seq = seq.to(args.device, non_blocking=non_blocking)
                pos = pos.to(args.device, non_blocking=non_blocking)
                neg = neg.to(args.device, non_blocking=non_blocking)
                token_type = token_type.to(args.device, non_blocking=non_blocking)
                next_token_type = next_token_type.to(args.device, non_blocking=non_blocking)
                
                with amp_autocast('cuda', amp_dtype, scaler_enabled):
                    loss, pos_sim, neg_sim = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, dataset
                    )
                valid_loss_sum += loss.item()
                valid_pos_sim_sum += pos_sim
                valid_neg_sim_sum += neg_sim

                # release
                del seq, pos, neg, token_type, next_token_type
                del seq_feat, pos_feat, neg_feat, next_action_type, loss
                
        avg_valid_loss = valid_loss_sum / len(valid_loader)
        avg_valid_pos_sim = valid_pos_sim_sum / len(valid_loader)
        avg_valid_neg_sim = valid_neg_sim_sum / len(valid_loader)
        
        writer.add_scalar('Loss/valid', avg_valid_loss, epoch)
        writer.add_scalar('Similarity/Positive_valid', avg_valid_pos_sim, epoch)
        writer.add_scalar('Similarity/Negative_valid', avg_valid_neg_sim, epoch)

        print(f"Epoch {epoch} validation loss: {avg_valid_loss:.4f}")
        print(f"  Val Positive Similarity: {avg_valid_pos_sim:.4f}")
        print(f"  Val Negative Similarity: {avg_valid_neg_sim:.4f}")

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={avg_valid_loss:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Training Done")
    writer.close()
    log_file.close()


if __name__ == '__main__':
    main()