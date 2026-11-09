# Adapted directly from train.py. Keep its optimizer, AMP step order and checkpoint keys.
from __future__ import print_function, division
import json
import os
import os.path as osp
import sys
import time
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from utils.common import Edict
from utils.log import LOG_INFO
from module_baseline import StereoBaseline
from module_baseline.loss_functions import stereo_loss
from module_baseline.debug import ShapeDebug
from module_baseline.training import (training_parser, seed_everything, make_dataset, make_loader,
                                     move_batch, check_device, model_parameters, batch_metrics, evaluate, write_json)

torch.backends.cudnn.benchmark = True
args = None
opts = None


def make_options(arguments):
    # 沿用原 train.py 的 opts 名称；仅映射独立双目配置和输出目录。
    options = Edict()
    options.name = arguments.name
    options.model_dir = arguments.output
    options.runs_dir = str(Path(arguments.output) / 'runs')
    options.snapshot_path = arguments.restore_ckpt
    options.pretrain_path = arguments.pretrain_ckpt
    options.total_epochs = arguments.epochs
    options.lr = arguments.learning_rate
    options.wdecay = arguments.weight_decay
    options.grad_clip = arguments.grad_clip
    options.lec_weight = arguments.lec_weight
    return options


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(model, num_steps):
    # 论文 §4.1：AdamW + OneCycleLR，先短暂升温，再按余弦衰减学习率。
    optimizer = optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=opts.wdecay, eps=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, opts.lr, num_steps + 100,
        pct_start=0.01, cycle_momentum=False, anneal_strategy='cos'
    )
    return optimizer, scheduler


def require_checkpoint(path, label):
    if path is not None and not osp.isfile(path):
        raise SystemExit(f'{label} not found: {path}')


def train(epoch_total, load_state):
    # 原四路 Dataset 换成独立双目数据集：完整 Rig TIFF -> 左光心径向 GT -> 工作轴索引。
    data = make_dataset(args, 'train')
    dbloader = make_loader(data, args, shuffle=not args.smoke_test)
    if not len(dbloader):
        raise ValueError('Training batch_size exceeds the available training samples')
    total_num_steps = len(data) * opts.total_epochs // max(args.batch_size, 1)
    validation = make_dataset(args, 'val') if args.val_count and not args.smoke_test else None
    val_loader = make_loader(validation, args) if validation is not None else None
    device = check_device(args.device)
    opts.net_opts = Edict(model_parameters(args, device))
    core_net = StereoBaseline(**opts.net_opts).to(device)
    # 与原版一致，CUDA 路径保留 DataParallel；CPU 分支用于以后的小尺寸调试。
    net = nn.DataParallel(core_net, device_ids=[device.index] if device.index is not None else None) if device.type == 'cuda' else core_net
    LOG_INFO('Parameter Count: %d' % count_parameters(net))
    optimizer, scheduler = fetch_optimizer(net, total_num_steps)
    scaler = GradScaler('cuda', enabled=opts.net_opts.mixed_precision)
    output = Path(opts.model_dir)
    output.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=opts.runs_dir)
    write_json(output / 'args.json', dict(cmd=' '.join(sys.argv), start_time=datetime.now().isoformat(), args=vars(args)))
    write_json(output / 'geometry.json', data.metadata)
    write_json(output / 'resolved_config.json', dict(model=dict(opts.net_opts), training=vars(args), geometry=data.metadata))

    start_epoch, total_iters = 0, 0
    if load_state:
        load_path = opts.snapshot_path or opts.pretrain_path
        snapshot = torch.load(load_path, map_location='cpu', weights_only=False)
        # 原字段 net_state_dict；兼容 DataParallel 前缀，不做四目权重的猜测映射。
        state = snapshot.get('net_state_dict', snapshot.get('model', snapshot))
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        try:
            core_net.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError('Checkpoint is not compatible with this StereoBaseline / volume mode; four-camera PDFOmni weights cannot be loaded directly') from exc
        if opts.snapshot_path:
            if snapshot.get('geometry') != data.metadata:
                raise ValueError('Resume geometry differs: check cameras, calibration, masks, candidates and sweep angles')
            for key in ('epochs', 'batch_size', 'scheduler', 'optimizer', 'learning_rate', 'weight_decay'):
                if snapshot['training_args'][key] != getattr(args, key):
                    raise ValueError('Resume must retain optimizer/scheduler planning parameter: ' + key)
            for key in ('num_depth', 'sweep_size', 'volume_update_mode', 'corr_levels', 'corr_radius', 'share_encoder', 'ot_iter',
                        'gru_small_kernel', 'gru_large_kernel', 'gru_large_mode'):
                if snapshot['model_config'][key] != opts.net_opts[key]:
                    raise ValueError('Resume model configuration mismatch: ' + key)
            start_epoch = snapshot['epoch'] + 1
            total_iters = snapshot['global_step']
            optimizer.load_state_dict(snapshot['optimizer'])
            scheduler.load_state_dict(snapshot['scheduler'])
            scaler.load_state_dict(snapshot['scaler'])
            torch.set_rng_state(snapshot['rng']['torch'])
            np.random.set_state(snapshot['rng']['numpy'])
            random.setstate(snapshot['rng']['python'])
            if device.type == 'cuda' and snapshot['rng']['cuda'] is not None:
                if len(snapshot['rng']['cuda']) != torch.cuda.device_count():
                    raise ValueError('Resume requires the same visible CUDA device count for RNG restoration')
                torch.cuda.set_rng_state_all(snapshot['rng']['cuda'])
        LOG_INFO('checkpoint %s is loaded' % load_path)

    LOG_INFO('OmniThings raw GT: %s' % json.dumps(data.inspect_gt()))
    LOG_INFO('train=%d val=%d; mode=%s; D/H/W=%d/%d/%d' %
             (len(data), len(validation) if validation is not None else 0, args.volume_update_mode,
              args.num_depth, *args.sweep_size))
    LOG_EVERY = max(1, int(args.log_every))
    stop = args.max_steps > 0 and total_iters >= args.max_steps
    for epoch in range(start_epoch, epoch_total):
        if stop:
            break
        net.train()
        if args.fix_bn:
            for module in net.modules():
                # 与原 PDFOmni.freeze_bn 一致，只冻结 BatchNorm2d。
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start_epoch_time = time.perf_counter()
        train_loss, usable_batches, optimizer_updates = 0., 0, 0
        for step, data_blob in enumerate(dbloader):
            batch = move_batch(data_blob, device)
            optimizer.zero_grad(set_to_none=True)
            trace = ShapeDebug(core_net) if args.shape_debug and total_iters == 0 else None
            try:
                out = net(batch['imgs'], batch['grids'], batch['valid_mask'], iters=args.train_iters,
                          return_intermediates=trace is not None)
                if trace is not None:
                    trace.record_output(out)
                    write_json(output / 'shapes.json', trace.records)
            finally:
                if trace is not None:
                    trace.close()
            valid = batch['gt_valid'] & out['valid_mask']
            if not valid.any():
                if args.smoke_test:
                    raise ValueError('Smoke batch has no valid supervision; inspect calibrated FOV and GT reprojection')
                LOG_INFO('Skipping unsupervised frames: %s' % batch['frame'].tolist())
                continue
            # 沿用原 Sequence + lambda*LEC；GT 已在 D 层工作轴上，LEC 内不再除以 2。
            loss, loss_parts = stereo_loss(out, batch['gt_index'], batch['gt_valid'], opts.lec_weight,
                                            args.loss_gamma, supervise_init=args.supervise_init)
            loss_seq, loss_lec = loss_parts['sequence_loss'], loss_parts['lec_loss']
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite stereo loss')
            train_loss += float(loss.detach().cpu())
            usable_batches += 1
            epoch_loss = train_loss / usable_batches

            # 原 train.py 的 AMP 更新顺序：unscale -> clip -> step -> update。
            # 如果 GradScaler 因溢出跳过 optimizer，同样跳过 scheduler。
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if opts.grad_clip is not None and opts.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), opts.grad_clip)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            stepped = scaler.get_scale() >= scale_before_step
            if stepped:
                scheduler.step()
                optimizer_updates += 1
            total_iters += 1

            metrics = batch_metrics(out, batch)
            metrics.update(epoch=epoch, step=total_iters, frames=batch['frame'].tolist(),
                           loss=float(loss.detach()), sequence_loss=float(loss_seq.detach()), lec_loss=float(loss_lec.detach()),
                           learning_rate=scheduler.get_last_lr()[0], optimizer_step=stepped)
            with (output / 'steps.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics) + '\n')
            if step % LOG_EVERY == 0 or args.smoke_test:
                LOG_INFO('Iter %d training loss = %.4f, seq = %.4f, lec = %.4f, avg = %.4f' %
                         (total_iters, metrics['loss'], metrics['sequence_loss'], metrics['lec_loss'], epoch_loss))
                for tag in ('loss', 'sequence_loss', 'lec_loss', 'valid_fraction', 'mean_initial_entropy',
                            'mean_initial_normalized_entropy', 'init_index_mae'):
                    if metrics[tag] is not None:
                        writer.add_scalar('train/' + tag, metrics[tag], total_iters)
                for t, error in enumerate(metrics['iteration_index_mae']):
                    writer.add_scalar('train/index_mae_iter_%02d' % t, error, total_iters)
                writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], total_iters)
            del out, loss
            if args.max_steps and total_iters >= args.max_steps:
                stop = True
                break
        if not usable_batches:
            raise ValueError('Epoch has no valid supervision; inspect pair selection / local GT')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - start_epoch_time
        peak_memory = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None
        report = dict(epoch=epoch, train_loss=epoch_loss, usable_batches=usable_batches, optimizer_updates=optimizer_updates,
                      seconds=epoch_seconds, peak_cuda_allocated_mib=peak_memory,
                      peak_cuda_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20 if device.type == 'cuda' else None)
        # 按任务要求，从官方训练集划验证集，官方测试集留到 eval_stereo.py 的最终评估。
        if val_loader is not None:
            report['validation'] = evaluate(core_net, val_loader, device, data.axis, args.valid_iters)
            for key, value in report['validation'].items():
                if isinstance(value, (int, float)):
                    writer.add_scalar('val/' + key, value, total_iters)
        writer.add_scalar('train/epoch_loss', epoch_loss, total_iters)
        writer.add_scalar('train/epoch_seconds', epoch_seconds, epoch)
        if peak_memory is not None:
            writer.add_scalar('train/peak_cuda_allocated_mib', peak_memory, epoch)
        write_json(output / ('epoch_%03d.json' % epoch), report)
        LOG_INFO(json.dumps(report))

        # 保留原 checkpoint 字段，并补充双目几何、scaler、工作深度轴和 RNG 状态。
        snapshot = dict(net_state_dict=net.state_dict(), net_opts=dict(opts.net_opts), epoch=epoch,
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), epoch_loss=float(epoch_loss),
                        scaler=scaler.state_dict(), global_step=total_iters, geometry=data.metadata,
                        model_config=dict(opts.net_opts), training_args=vars(args),
                        rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                                 cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None))
        savefilename = output / ('%s_e%d.pth' % (opts.name, epoch))
        torch.save(snapshot, savefilename)
        temporary = output / 'last.tmp'
        torch.save(snapshot, temporary)
        temporary.replace(output / 'last.pt')
        writer.flush()
    writer.close()


def main():
    global args, opts
    parser = training_parser()
    args = parser.parse_args()
    if args.db_root is not None:
        args.root = str(Path(args.db_root) / 'omnithings')
    if args.smoke_test:
        args.epochs, args.max_steps = 1, 1
    try:
        if args.train_iters < 1 or args.epochs < 1 or args.batch_size < 1 or args.lec_weight < 0:
            raise ValueError('epochs, batch_size and train_iters must be positive; lec_weight must be non-negative')
        if args.num_workers < 0 or args.learning_rate <= 0 or args.grad_clip < 0:
            raise ValueError('Invalid worker count, learning rate or gradient clip')
        if args.restore_ckpt and args.pretrain_ckpt:
            raise ValueError('Use either --restore_ckpt or --pretrain_ckpt')
        if (Path(args.output) / 'last.pt').exists() and not args.restore_ckpt:
            raise ValueError('Output already has last.pt; use --restore_ckpt or a new --output')
        opts = make_options(args)
        require_checkpoint(opts.snapshot_path, '--restore_ckpt')
        require_checkpoint(opts.pretrain_path, '--pretrain_ckpt')
        seed_everything(args.seed)
        train(opts.total_epochs, opts.snapshot_path is not None or opts.pretrain_path is not None)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
