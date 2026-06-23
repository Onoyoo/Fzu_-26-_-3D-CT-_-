"""两阶段肺结节检测推理 + FROC 评估"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict

from config import MODELS_DIR, RAW_DATA_DIR, CSV_DIR
from codes.models.rpn import RegionProposalNetwork3D
from codes.models.classifier import NoduleClassifier
from codes.datasets.preprocess import CTScanPreprocessor, load_annotations

def load_models(device):
    """加载 RPN 和分类器"""
    rpn = RegionProposalNetwork3D().to(device)
    ckpt = torch.load(MODELS_DIR / 'rpn' / 'rpn_best.pth', map_location=device)
    if 'model_state_dict' in ckpt:
        rpn.load_state_dict(ckpt['model_state_dict'])
    else:
        rpn.load_state_dict(ckpt)
    rpn.eval()
    print(f" RPN 加载成功")

    classifier = NoduleClassifier().to(device)
    ckpt2 = torch.load(MODELS_DIR / 'classifier' / 'classifier_best.pth', map_location=device)
    classifier.load_state_dict(ckpt2['model_state_dict'])
    classifier.eval()
    print(f" 分类器加载成功 (Epoch {ckpt2['epoch']}, val_loss={ckpt2['val_loss']:.4f})")

    return rpn, classifier


def extract_rpn_proposals(rpn, patch, device, top_k=50):
    """RPN 前向 → top_k 候选框"""
    with torch.no_grad():
        out = rpn(patch)
        cls_logits = out['cls_logits']
        reg_output = out['reg_output']
        anchors = out['anchors']

    cls_prob = torch.softmax(cls_logits, dim=1)[:, 1, :, :, :, :].flatten()
    reg_output = reg_output.permute(0, 2, 3, 4, 5, 1).flatten(0, -2)

    proposals = anchors.clone()
    proposals[:, :3] += reg_output[:, :3]
    proposals[:, 3:] *= torch.exp(reg_output[:, 3:])

    scores, idxs = cls_prob.topk(min(top_k, len(cls_prob)))
    proposals = proposals[idxs]

    return proposals, scores


def nms_3d(boxes, scores, thresh):
    """3D IoU NMS"""
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break
        iou = box_iou_3d(boxes[i], boxes[order[1:]])
        mask = iou < thresh
        order = order[1:][mask]

    return torch.tensor(keep, device=boxes.device)


def box_iou_3d(box, boxes):
    """单个框 vs 多个框的 3D IoU"""
    b1 = box_to_corners(box.unsqueeze(0))
    b2 = box_to_corners(boxes)

    inter_z1 = torch.max(b1[0, 0], b2[:, 0])
    inter_y1 = torch.max(b1[0, 1], b2[:, 1])
    inter_x1 = torch.max(b1[0, 2], b2[:, 2])
    inter_z2 = torch.min(b1[0, 3], b2[:, 3])
    inter_y2 = torch.min(b1[0, 4], b2[:, 4])
    inter_x2 = torch.min(b1[0, 5], b2[:, 5])

    inter_d = torch.clamp(inter_z2 - inter_z1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_vol = inter_d * inter_h * inter_w

    vol1 = (b1[0, 3] - b1[0, 0]) * (b1[0, 4] - b1[0, 1]) * (b1[0, 5] - b1[0, 2])
    vol2 = (b2[:, 3] - b2[:, 0]) * (b2[:, 4] - b2[:, 1]) * (b2[:, 5] - b2[:, 2])

    union = vol1 + vol2 - inter_vol
    iou = inter_vol / (union + 1e-8)
    return iou


def box_to_corners(boxes):
    """(center_z, center_y, center_x, d, h, w) → (z1, y1, x1, z2, y2, x2)"""
    c = boxes[:, :3]
    s = boxes[:, 3:] / 2
    z1y1x1 = c - s
    z2y2x2 = c + s
    return torch.cat([z1y1x1, z2y2x2], dim=1)


def classify_proposals(classifier, ct_tensor, proposals, device, crop_size=64):
    """对 RPN 提案用分类器打分"""
    D, H, W = ct_tensor.shape[2:]
    detections = []

    for prop in proposals:
        zc, yc, xc = prop[:3].long()
        zc = int(zc.clamp(0, D-1))
        yc = int(yc.clamp(0, H-1))
        xc = int(xc.clamp(0, W-1))

        z_start = max(0, zc - crop_size // 2)
        z_end = min(D, zc + crop_size // 2)
        y_start = max(0, yc - crop_size // 2)
        y_end = min(H, yc + crop_size // 2)
        x_start = max(0, xc - crop_size // 2)
        x_end = min(W, xc + crop_size // 2)

        patch = ct_tensor[:, :, z_start:z_end, y_start:y_end, x_start:x_end]

        #  Padding 到 64³（和训练时一致）
        if patch.shape[2:] != (crop_size,) * 3:
            pad_d = crop_size - patch.shape[2]
            pad_h = crop_size - patch.shape[3]
            pad_w = crop_size - patch.shape[4]
            pad = [
                pad_w // 2, pad_w - pad_w // 2,
                pad_h // 2, pad_h - pad_h // 2,
                pad_d // 2, pad_d - pad_d // 2,
            ]
            patch = torch.nn.functional.pad(patch, pad, mode='constant', value=0)

        with torch.no_grad():
            out = classifier(patch)
            cls_prob = torch.softmax(out['cls_logits'], dim=1)
            score = cls_prob[0, 1].item()

        if score > 0.1:
            detections.append({
                'center': prop[:3].cpu().numpy(),
                'size': prop[3:].cpu().numpy(),
                'score': score
            })

    return sorted(detections, key=lambda x: x['score'], reverse=True)


def evaluate_froc(all_detections, annotations_df, iou_thresh=0.3):
    """计算 FROC 曲线和 CPM"""
    fp_rates = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    sensitivities = []

    ct_results = defaultdict(lambda: {'gt': [], 'preds': []})

    for uid, dets in all_detections.items():
        ct_anns = annotations_df[annotations_df['seriesuid'] == uid]
        ct_results[uid]['gt'] = [
            (row['coordZ'], row['coordY'], row['coordX'], row['diameter_mm'])
            for _, row in ct_anns.iterrows()
        ]
        ct_results[uid]['preds'] = dets

    for fp_rate in fp_rates:
        total_tp = 0

        for uid, data in ct_results.items():
            gt_list = data['gt']
            preds = data['preds']
            matched = set()
            fp_count = 0

            for p in preds:
                if fp_count >= fp_rate:
                    break

                best_iou = 0
                best_gt = -1
                for gi, gt in enumerate(gt_list):
                    if gi in matched:
                        continue
                    iou = compute_iou(p['center'], p['size'], gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gi

                if best_iou >= iou_thresh:
                    total_tp += 1
                    matched.add(best_gt)
                else:
                    fp_count += 1

        total_gt = max(sum(len(v['gt']) for v in ct_results.values()), 1)
        sensitivities.append(total_tp / total_gt)

    cpm = np.mean(sensitivities)
    print(f"\n{'='*50}")
    print(f"FROC 评估结果:")
    for fp, sens in zip(fp_rates, sensitivities):
        print(f"  {fp:.3f} FPs/scan: {sens:.4f}")
    print(f"  CPM: {cpm:.4f}")
    print(f"{'='*50}")
    return cpm


def compute_iou(center, size, gt):
    """计算预测框和 GT 框的 IoU"""
    z1, y1, x1 = center - np.array(size) / 2
    z2, y2, x2 = center + np.array(size) / 2

    gz, gy, gx = gt[:3]
    gd = gt[3]
    gz1, gy1, gx1 = gz - gd/2, gy - gd/2, gx - gd/2
    gz2, gy2, gx2 = gz + gd/2, gy + gd/2, gx + gd/2

    inter_z = max(0, min(z2, gz2) - max(z1, gz1))
    inter_y = max(0, min(y2, gy2) - max(y1, gy1))
    inter_x = max(0, min(x2, gx2) - max(x1, gx1))
    inter_vol = inter_z * inter_y * inter_x

    vol_pred = (z2 - z1) * (y2 - y1) * (x2 - x1)
    vol_gt = gd * gd * gd
    union = vol_pred + vol_gt - inter_vol
    return inter_vol / (union + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, default=9, help='评估用的 subset')
    parser.add_argument('--max-cts', type=int, default=10, help='最多评估 CT 数')
    parser.add_argument('--top-k', type=int, default=200, help='RPN 候选数')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    rpn, classifier = load_models(device)
    preprocessor = CTScanPreprocessor()
    annotations_df = load_annotations()

    subset_dir = RAW_DATA_DIR / f'subset{args.subset}'
    ct_files = sorted(subset_dir.glob('*.mhd'))[:args.max_cts]
    print(f"评估 {len(ct_files)} 个 CT 文件")

    all_detections = {}
    crop_size = 96
    stride = 48

    for ct_path in tqdm(ct_files, desc="推理中"):
        uid = ct_path.stem
        try:
            ct_array, spacing, origin, direction = preprocessor.load_ct_scan(ct_path)
            if ct_array is None:
                continue

            resampled = preprocessor.resample_ct_monai(ct_array)
            normalized = preprocessor.normalize_ct_monai(resampled)
            ct_tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0).to(device)
            D, H, W = ct_tensor.shape[2:]

            all_props = []
            all_scores = []

            for z in range(0, max(D - crop_size, 1), stride):
                for y in range(0, max(H - crop_size, 1), stride):
                    for x in range(0, max(W - crop_size, 1), stride):
                        z_end = min(z + crop_size, D)
                        y_end = min(y + crop_size, H)
                        x_end = min(x + crop_size, W)
                        patch = ct_tensor[:, :, z:z_end, y:y_end, x:x_end]

                        if patch.shape[2:] != (crop_size,) * 3:
                            pad = [0, max(0, crop_size - patch.shape[4]),
                                   0, max(0, crop_size - patch.shape[3]),
                                   0, max(0, crop_size - patch.shape[2])]
                            patch = torch.nn.functional.pad(patch, pad)

                        props, sc = extract_rpn_proposals(rpn, patch, device, top_k=30)
                        props[:, 0] += z
                        props[:, 1] += y
                        props[:, 2] += x
                        all_props.append(props)
                        all_scores.append(sc)

            if all_props:
                all_props = torch.cat(all_props)
                all_scores = torch.cat(all_scores)
                keep = nms_3d(all_props, all_scores, 0.5)
                proposals = all_props[keep[:args.top_k]]
            else:
                proposals = torch.zeros(0, 6, device=device)

            detections = classify_proposals(classifier, ct_tensor, proposals, device)
            all_detections[uid] = detections

        except Exception as e:
            print(f"  {ct_path.name}: {e}")
            all_detections[uid] = []

    cpm = evaluate_froc(all_detections, annotations_df)

    import json
    results = {}
    for uid, dets in all_detections.items():
        results[uid] = [{'center': d['center'].tolist(), 'size': d['size'].tolist(), 'score': d['score']} for d in dets]
    Path('results').mkdir(exist_ok=True)
    with open('results/detections.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n 结果保存到 results/detections.json")
    print(f" CPM: {cpm:.4f}")

if __name__ == '__main__':
    main()