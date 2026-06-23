"""训练分类器模型 — 混合精度 + 梯度累积 + Focal Loss + Smooth L1 Loss"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import argparse
import time
from tqdm import tqdm
from collections import defaultdict

from config import MODELS_DIR
from codes.datasets.luna16_dataset import create_classifier_dataloader
from codes.models.classifier import NoduleClassifier
from codes.losses import FocalLoss


def train_epoch(model, dataloader, optimizer, cls_loss_fn, device, epoch,
                scaler, accumulation_steps, use_amp):
    """训练一个epoch"""
    model.train()
    running = defaultdict(float)
    batch_count = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]', dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        bboxes = batch.get('bbox', None)
        if bboxes is not None:
            bboxes = bboxes.to(device, non_blocking=True)

        if labels.dim() > 1:
            labels = labels.view(-1)[:images.size(0)]
        labels = labels.long().flatten()

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                losses = model.compute_loss(
                    outputs['cls_logits'], outputs['reg_output'],
                    labels, gt_boxes=bboxes
                )
                total_loss = losses['total_loss'] / accumulation_steps
        else:
            outputs = model(images)
            losses = model.compute_loss(
                outputs['cls_logits'], outputs['reg_output'],
                labels, gt_boxes=bboxes
            )
            total_loss = losses['total_loss'] / accumulation_steps

        if use_amp:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            pred_labels = torch.argmax(outputs['cls_logits'], dim=1)
            correct = (pred_labels == labels).sum().item()

        running['loss'] += losses['total_loss'].item()
        running['cls_loss'] += losses['cls_loss'].item()
        running['reg_loss'] += losses['reg_loss'].item()
        running['correct'] += correct
        running['samples'] += labels.size(0)
        batch_count += 1

        # ✅ 每 50 步在进度条显示，每 200 步打印一行
        if (batch_idx + 1) % 50 == 0:
            avg_loss = running['loss'] / batch_count
            avg_acc = running['correct'] / running['samples']
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{avg_acc:.3f}',
                'cls': f'{running["cls_loss"]/batch_count:.4f}',
                'reg': f'{running["reg_loss"]/batch_count:.4f}'
            })

        if (batch_idx + 1) % 200 == 0:
            avg_loss = running['loss'] / batch_count
            avg_acc = running['correct'] / running['samples']
            print(f"\n  [步骤 {batch_idx+1}] loss={avg_loss:.4f} | cls={running['cls_loss']/batch_count:.4f} | reg={running['reg_loss']/batch_count:.4f} | acc={avg_acc:.3f}")

    if (batch_idx + 1) % accumulation_steps != 0:
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()

    return {
        'loss': running['loss'] / batch_count,
        'cls_loss': running['cls_loss'] / batch_count,
        'reg_loss': running['reg_loss'] / batch_count,
        'accuracy': running['correct'] / running['samples']
    }


def validate_epoch(model, dataloader, cls_loss_fn, device, epoch, use_amp):
    """验证一个epoch"""
    model.eval()
    running = defaultdict(float)
    batch_count = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]  ', leave=False, dynamic_ncols=True)

        for batch in pbar:
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            bboxes = batch.get('bbox', None)
            if bboxes is not None:
                bboxes = bboxes.to(device, non_blocking=True)

            if labels.dim() > 1:
                labels = labels.view(-1)[:images.size(0)]
            labels = labels.long().flatten()

            if use_amp:
                with torch.cuda.amp.autocast(): 
                    outputs = model(images)
                    losses = model.compute_loss(
                        outputs['cls_logits'], outputs['reg_output'],
                        labels, gt_boxes=bboxes
                    )
            else:
                outputs = model(images)
                losses = model.compute_loss(
                    outputs['cls_logits'], outputs['reg_output'],
                    labels, gt_boxes=bboxes
                )

            pred_labels = torch.argmax(outputs['cls_logits'], dim=1)
            correct = (pred_labels == labels).sum().item()

            running['loss'] += losses['total_loss'].item()
            running['cls_loss'] += losses['cls_loss'].item()
            running['reg_loss'] += losses['reg_loss'].item()
            running['correct'] += correct
            running['samples'] += labels.size(0)
            batch_count += 1

            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.3f}",
                'acc': f"{correct / labels.size(0):.3f}"
            })

    return {
        'loss': running['loss'] / batch_count,
        'cls_loss': running['cls_loss'] / batch_count,
        'reg_loss': running['reg_loss'] / batch_count,
        'accuracy': running['correct'] / running['samples']
    }


def main():
    parser = argparse.ArgumentParser(description='训练分类器')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--data-dir', type=str, default='processed/classifier_samples')
    parser.add_argument('--accumulation-steps', type=int, default=2, help='梯度累积步数')
    parser.add_argument('--no-amp', action='store_true', help='禁用混合精度')
    parser.add_argument('--resume', type=str, default=None, help='恢复检查点路径')
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 分类器训练 | Focal Loss + Smooth L1 Loss")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available() and not args.no_amp

    print(f"📌 设备: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   混合精度: {'✅ 开启' if use_amp else '❌ 关闭'}")

    print("\n📦 加载数据...")
    data_dir = Path(args.data_dir)

    train_loader = create_classifier_dataloader(
        data_dir=data_dir, mode='train',
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    try:
        val_loader = create_classifier_dataloader(
            data_dir=data_dir, mode='val',
            batch_size=args.batch_size, num_workers=args.num_workers
        )
    except:
        val_loader = None
        print("⚠️  无法创建验证集")

    eff_bs = args.batch_size * args.accumulation_steps
    print(f"   训练集: {len(train_loader.dataset)} 样本, 等效批次: {eff_bs}")
    if val_loader:
        print(f"   验证集: {len(val_loader.dataset)} 样本")

    print("\n🔧 创建模型...")
    model = NoduleClassifier().to(device)
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    cls_loss_fn = FocalLoss(gamma=2.0, alpha=0.25, reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None   

    start_epoch = 1
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    if args.resume:
        print(f"\n📂 恢复: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f"   从 Epoch {start_epoch} 继续, 最佳损失: {best_val_loss:.4f}")

    model_dir = MODELS_DIR / 'classifier'
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🎯 开始训练 {args.epochs} epochs（每个 epoch 自动保存）")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n📊 Epoch {epoch}/{args.epochs} | LR: {current_lr:.2e}")
        print("-" * 40)

        train_metrics = train_epoch(
            model, train_loader, optimizer, cls_loss_fn, device, epoch,
            scaler, args.accumulation_steps, use_amp
        )

        if val_loader:
            val_metrics = validate_epoch(model, val_loader, cls_loss_fn, device, epoch, use_amp)
        else:
            val_metrics = {'loss': 0.0, 'cls_loss': 0.0, 'reg_loss': 0.0, 'accuracy': 0.0}

        scheduler.step()
        epoch_time = time.time() - epoch_start

        print(f"   训练: Loss={train_metrics['loss']:.4f} "
              f"(cls={train_metrics['cls_loss']:.4f}, reg={train_metrics['reg_loss']:.4f}) "
              f"Acc={train_metrics['accuracy']:.4f}")
        print(f"   验证: Loss={val_metrics['loss']:.4f} "
              f"(cls={val_metrics['cls_loss']:.4f}, reg={val_metrics['reg_loss']:.4f}) "
              f"Acc={val_metrics['accuracy']:.4f} | 耗时 {epoch_time:.1f}s")

        # ========== 每个 epoch 都保存 ==========
        checkpoint_path = model_dir / f"classifier_epoch_{epoch:02d}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['loss']
        }, checkpoint_path)

        # 最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss']
            }, model_dir / 'classifier_best.pth')
            print(f"   ✅ 最佳模型 (loss={best_val_loss:.4f}) | 检查点: {checkpoint_path.name}")
        else:
            patience_counter += 1
            print(f"   💾 已保存: {checkpoint_path.name} | 未改善 ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\n⏹ 早停！最佳 val_loss = {best_val_loss:.4f}")
            break

    print("\n" + "=" * 60)
    print(f"✅ 训练完成! 最佳验证损失: {best_val_loss:.4f}")
    print(f"📁 模型目录: {model_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()