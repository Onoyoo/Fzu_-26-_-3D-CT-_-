"""
RPN网络训练脚本 - 3D区域提议网络
"""

import os
from pathlib import Path
import argparse
from datetime import datetime

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import MODELS_DIR, LOGS_DIR
from codes.models.rpn import RegionProposalNetwork3D
from codes.datasets.luna16_dataset import create_rpn_dataloader
from codes.losses import FocalLoss as FocalLoss3D


def train_epoch(model, dataloader, optimizer, cls_loss_fn, device):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    num_batches = 0
    
    scaler = GradScaler()
    
    for batch in tqdm(dataloader, desc="训练中"):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            cls_logits = outputs['cls_logits']
            B = cls_logits.shape[0]
            d = cls_logits.shape[3] // 2
            h = cls_logits.shape[4] // 2
            w = cls_logits.shape[5] // 2
            cls_simple = cls_logits[:, :, 0, d, h, w]
            
            if labels.dim() > 1:
                labels_flat = labels.view(labels.size(0), -1)[:, 0]
            else:
                labels_flat = labels.squeeze()
            
            cls_loss = cls_loss_fn(cls_simple, labels_flat)
            loss = cls_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        num_batches += 1
        
        if num_batches % 10 == 0:
            torch.cuda.empty_cache()
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'cls_loss': total_cls_loss / max(num_batches, 1)
    }


def validate(model, dataloader, cls_loss_fn, device):
    if dataloader is None or len(dataloader) == 0:
        return {'loss': 0.0, 'cls_loss': 0.0}
    
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="验证中"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            
            cls_logits = outputs['cls_logits']
            B = cls_logits.shape[0]
            d = cls_logits.shape[3] // 2
            h = cls_logits.shape[4] // 2
            w = cls_logits.shape[5] // 2
            cls_simple = cls_logits[:, :, 0, d, h, w]
            
            if labels.dim() > 1:
                labels_flat = labels.view(labels.size(0), -1)[:, 0]
            else:
                labels_flat = labels.squeeze()
            
            cls_loss = cls_loss_fn(cls_simple, labels_flat)
            loss = cls_loss
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            num_batches += 1
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'cls_loss': total_cls_loss / max(num_batches, 1)
    }


def main():
    parser = argparse.ArgumentParser(description='训练RPN网络')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=2, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num-workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--data-dir', type=str, default='processed/rpn_samples', help='数据目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的checkpoint路径')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    data_dir = Path(args.data_dir)
    
    train_loader = create_rpn_dataloader(
        data_dir=data_dir,
        mode='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    val_loader = create_rpn_dataloader(
        data_dir=data_dir,
        mode='val',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"训练集大小: {len(train_loader.dataset)} 个样本")
    if val_loader:
        print(f"验证集大小: {len(val_loader.dataset)} 个样本")
    
    model = RegionProposalNetwork3D()
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数: {total_params:,}")
    
    cls_loss_fn = FocalLoss3D(gamma=2.0, reduction='mean')
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    start_epoch = 0
    best_val_loss = float('inf')  # ✅ 只在这里定义一次

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"恢复训练从 epoch {start_epoch}，之前最佳验证损失: {best_val_loss:.4f}")
    else:
        print("从头开始训练")
    
    log_dir = LOGS_DIR / f"rpn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # ✅ 删掉这行：best_val_loss = float('inf')
    
    model_dir = MODELS_DIR / 'rpn'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n开始训练 {args.epochs} 个epoch...")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        # ... 后面的代码不变 ...
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, cls_loss_fn, device
        )
        
        val_metrics = validate(
            model, val_loader, cls_loss_fn, device
        )
        
        scheduler.step()
        
        print(f"训练: Loss={train_metrics['loss']:.4f}, Cls={train_metrics['cls_loss']:.4f}")
        print(f"验证: Loss={val_metrics['loss']:.4f}, Cls={val_metrics['cls_loss']:.4f}")
        
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Cls_Loss/train', train_metrics['cls_loss'], epoch)
        writer.add_scalar('Cls_Loss/val', val_metrics['cls_loss'], epoch)
        writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = model_dir / 'rpn_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': args
            }, best_model_path)
            print(f"✓ 最佳模型已保存: {best_model_path}")
        
                    # 每个 epoch 都保存 checkpoint
        checkpoint_path = model_dir / f'rpn_checkpoint_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
            'config': args
        }, checkpoint_path)
    
    writer.close()
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存在: {model_dir}")
    
    return best_val_loss


if __name__ == '__main__':
    main()