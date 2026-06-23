"""后处理模块 - NMS和结果导出"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import csv
import time
from datetime import datetime


class NMS3D:
    """3D非极大值抑制"""
    
    def __init__(self, nms_threshold: float = 0.3, max_detections: int = 100):
        """
        初始化
        
        Args:
            nms_threshold: IoU阈值
            max_detections: 最大检测数量
        """
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
    
    def apply(self, detections: List[Dict]) -> List[Dict]:
        """
        应用NMS
        
        Args:
            detections: 检测结果列表，每个元素包含:
                - 'center': 中心点 (z, y, x)
                - 'size': 尺寸 (d, h, w)
                - 'confidence': 置信度
                
        Returns:
            过滤后的检测结果
        """
        if not detections:
            return []
        
        # 按置信度排序
        sorted_indices = np.argsort([-d['confidence'] for d in detections])
        
        # 准备边界框数据
        boxes = np.array([
            [
                d['center'][0] - d['size'][0] / 2,  # z_min
                d['center'][1] - d['size'][1] / 2,  # y_min
                d['center'][2] - d['size'][2] / 2,  # x_min
                d['center'][0] + d['size'][0] / 2,  # z_max
                d['center'][1] + d['size'][1] / 2,  # y_max
                d['center'][2] + d['size'][2] / 2,  # x_max
                d['confidence']
            ] for d in detections
        ])
        
        # 应用NMS
        keep_indices = self._nms_3d(boxes)
        
        # 限制最大数量
        keep_indices = keep_indices[:self.max_detections]
        
        # 返回过滤后的检测结果
        filtered_detections = [detections[i] for i in keep_indices]
        
        return filtered_detections
    
    def _nms_3d(self, boxes: np.ndarray) -> np.ndarray:
        """
        3D NMS实现
        
        Args:
            boxes: 边界框数组 (N, 7)，每行: [z1, y1, x1, z2, y2, x2, score]
            
        Returns:
            保留的索引
        """
        if boxes.shape[0] == 0:
            return np.array([], dtype=np.int32)
        
        # 提取坐标和分数
        z1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x1 = boxes[:, 2]
        z2 = boxes[:, 3]
        y2 = boxes[:, 4]
        x2 = boxes[:, 5]
        scores = boxes[:, 6]
        
        # 计算体积
        volumes = (z2 - z1) * (y2 - y1) * (x2 - x1)
        
        # 按分数排序
        order = np.argsort(-scores)
        
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # 计算IoU
            z1_inter = np.maximum(z1[i], z1[order[1:]])
            y1_inter = np.maximum(y1[i], y1[order[1:]])
            x1_inter = np.maximum(x1[i], x1[order[1:]])
            
            z2_inter = np.minimum(z2[i], z2[order[1:]])
            y2_inter = np.minimum(y2[i], y2[order[1:]])
            x2_inter = np.minimum(x2[i], x2[order[1:]])
            
            # 计算交集体积
            inter_volumes = np.maximum(0, z2_inter - z1_inter) * \
                           np.maximum(0, y2_inter - y1_inter) * \
                           np.maximum(0, x2_inter - x1_inter)
            
            # 计算并集体积
            union_volumes = volumes[i] + volumes[order[1:]] - inter_volumes
            
            # 计算IoU
            ious = inter_volumes / (union_volumes + 1e-8)
            
            # 保留IoU小于阈值的框
            indices = np.where(ious <= self.nms_threshold)[0]
            order = order[indices + 1]
        
        return np.array(keep, dtype=np.int32)


class ResultsExporter:
    """结果导出器"""
    
    def __init__(self, output_dir: Path):
        """
        初始化
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_detections(self, detections: List[Dict], 
                         ct_file: Path,
                         processing_time: float) -> Path:
        """
        导出检测结果
        
        Args:
            detections: 检测结果
            ct_file: CT文件路径
            processing_time: 处理时间
            
        Returns:
            输出文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ct_file.stem}_detections_{timestamp}"
        
        # JSON格式
        json_path = self.output_dir / f"{filename}.json"
        self._export_to_json(detections, ct_file, processing_time, json_path)
        
        # CSV格式（用于提交）
        csv_path = self.output_dir / f"{filename}.csv"
        self._export_to_csv(detections, ct_file, csv_path)
        
        # 文本格式（可读）
        txt_path = self.output_dir / f"{filename}.txt"
        self._export_to_text(detections, ct_file, processing_time, txt_path)
        
        return json_path
    
    def _export_to_json(self, detections: List[Dict], ct_file: Path,
                       processing_time: float, output_path: Path):
        """导出为JSON格式"""
        result = {
            'ct_file': str(ct_file.name),
            'seriesuid': ct_file.stem,
            'timestamp': datetime.now().isoformat(),
            'processing_time_seconds': processing_time,
            'num_detections': len(detections),
            'detections': detections
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    def _export_to_csv(self, detections: List[Dict], ct_file: Path,
                      output_path: Path):
        """导出为CSV格式（LUNA16提交格式）"""
        # LUNA16提交格式: seriesuid,coordX,coordY,coordZ,diameter_mm
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入标题
            writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'confidence'])
            
            # 写入检测结果
            for detection in detections:
                # 提取信息
                seriesuid = ct_file.stem
                
                # 注意: CT坐标顺序通常是 (z, y, x)，但DICOM是 (x, y, z)
                # 这里假设输入已经是物理坐标
                coordZ, coordY, coordX = detection['center']
                
                # 计算平均直径（假设结节是椭球体）
                diameter = np.mean(detection['size'])
                
                confidence = detection['confidence']
                
                writer.writerow([seriesuid, coordX, coordY, coordZ, diameter, confidence])
    
    def _export_to_text(self, detections: List[Dict], ct_file: Path,
                       processing_time: float, output_path: Path):
        """导出为可读文本格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"肺结节检测结果\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"CT文件: {ct_file.name}\n")
            f.write(f"序列ID: {ct_file.stem}\n")
            f.write(f"处理时间: {processing_time:.2f} 秒\n")
            f.write(f"检测数量: {len(detections)}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if detections:
                f.write("检测到的结节:\n")
                f.write("-" * 60 + "\n")
                
                for i, detection in enumerate(detections, 1):
                    center = detection['center']
                    size = detection['size']
                    confidence = detection['confidence']
                    
                    f.write(f"\n结节 #{i}:\n")
                    f.write(f"  置信度: {confidence:.4f}\n")
                    f.write(f"  中心位置: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) mm\n")
                    f.write(f"  尺寸: ({size[0]:.1f}, {size[1]:.1f}, {size[2]:.1f}) mm\n")
                    f.write(f"  直径: {np.mean(size):.1f} mm\n")
                    f.write(f"  体积: {size[0] * size[1] * size[2]:.1f} mm³\n")
            else:
                f.write("未检测到结节\n")
            
            f.write("\n" + "=" * 60 + "\n")
    
    def export_summary(self, all_results: List[Dict], total_time: float) -> Path:
        """
        导出批量处理汇总
        
        Args:
            all_results: 所有结果列表
            total_time: 总处理时间
            
        Returns:
            汇总文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_dir / f"summary_{timestamp}.json"
        
        # 计算统计信息
        total_detections = sum(len(r['detections']) for r in all_results)
        avg_detections = total_detections / len(all_results) if all_results else 0
        
        # 置信度分布
        all_confidences = []
        for result in all_results:
            for detection in result['detections']:
                all_confidences.append(detection['confidence'])
        
        confidences_stats = {
            'mean': np.mean(all_confidences) if all_confidences else 0,
            'std': np.std(all_confidences) if all_confidences else 0,
            'min': np.min(all_confidences) if all_confidences else 0,
            'max': np.max(all_confidences) if all_confidences else 0
        }
        
        # 尺寸分布
        all_sizes = []
        for result in all_results:
            for detection in result['detections']:
                all_sizes.append(np.mean(detection['size']))
        
        sizes_stats = {
            'mean': np.mean(all_sizes) if all_sizes else 0,
            'std': np.std(all_sizes) if all_sizes else 0,
            'min': np.min(all_sizes) if all_sizes else 0,
            'max': np.max(all_sizes) if all_sizes else 0
        }
        
        # 构建汇总
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(all_results),
            'total_detections': total_detections,
            'average_detections_per_file': avg_detections,
            'total_processing_time_seconds': total_time,
            'average_processing_time_per_file': total_time / len(all_results) if all_results else 0,
            'confidence_statistics': confidences_stats,
            'size_statistics': sizes_stats,
            'files_with_detections': len([r for r in all_results if r['detections']]),
            'files_without_detections': len([r for r in all_results if not r['detections']]),
            'results': all_results
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 同时生成文本格式的汇总
        txt_summary_path = self.output_dir / f"summary_{timestamp}.txt"
        self._export_summary_text(summary, txt_summary_path)
        
        return summary_path
    
    def _export_summary_text(self, summary: Dict, output_path: Path):
        """导出文本格式的汇总"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("肺结节检测批量处理汇总\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"处理文件总数: {summary['total_files']}\n")
            f.write(f"检测结节总数: {summary['total_detections']}\n")
            f.write(f"平均每文件检测数: {summary['average_detections_per_file']:.2f}\n\n")
            
            f.write(f"总处理时间: {summary['total_processing_time_seconds']:.2f} 秒\n")
            f.write(f"平均每文件处理时间: {summary['average_processing_time_per_file']:.2f} 秒\n\n")
            
            f.write(f"有检测的文件数: {summary['files_with_detections']}\n")
            f.write(f"无检测的文件数: {summary['files_without_detections']}\n\n")
            
            f.write("置信度统计:\n")
            f.write(f"  平均值: {summary['confidence_statistics']['mean']:.4f}\n")
            f.write(f"  标准差: {summary['confidence_statistics']['std']:.4f}\n")
            f.write(f"  最小值: {summary['confidence_statistics']['min']:.4f}\n")
            f.write(f"  最大值: {summary['confidence_statistics']['max']:.4f}\n\n")
            
            f.write("结节尺寸统计:\n")
            f.write(f"  平均直径: {summary['size_statistics']['mean']:.2f} mm\n")
            f.write(f"  标准差: {summary['size_statistics']['std']:.2f} mm\n")
            f.write(f"  最小直径: {summary['size_statistics']['min']:.2f} mm\n")
            f.write(f"  最大直径: {summary['size_statistics']['max']:.2f} mm\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("各文件检测详情:\n")
            f.write("=" * 60 + "\n\n")
            
            for i, result in enumerate(summary['results'], 1):
                f.write(f"文件 #{i}: {result['file']}\n")
                f.write(f"  序列ID: {result['seriesuid']}\n")
                f.write(f"  处理时间: {result['processing_time']:.2f} 秒\n")
                f.write(f"  检测数量: {len(result['detections'])}\n")
                
                if result['detections']:
                    # 显示前3个检测的详细信息
                    for j, detection in enumerate(result['detections'][:3], 1):
                        center = detection['center']
                        size = detection['size']
                        confidence = detection['confidence']
                        
                        f.write(f"  检测 #{j}: 置信度={confidence:.3f}, "
                               f"中心=({center[0]:.1f},{center[1]:.1f},{center[2]:.1f}), "
                               f"尺寸=({size[0]:.1f},{size[1]:.1f},{size[2]:.1f})\n")
                    
                    if len(result['detections']) > 3:
                        f.write(f"  ... 还有 {len(result['detections']) - 3} 个检测\n")
                
                f.write("\n")


def test_nms():
    """测试NMS"""
    print("测试3D NMS...")
    
    # 创建测试检测结果
    detections = [
        {
            'center': (50, 50, 50),
            'size': (20, 20, 20),
            'confidence': 0.9
        },
        {
            'center': (55, 55, 55),
            'size': (22, 22, 22),
            'confidence': 0.8
        },
        {
            'center': (100, 100, 100),
            'size': (15, 15, 15),
            'confidence': 0.7
        }
    ]
    
    # 应用NMS
    nms = NMS3D(nms_threshold=0.5, max_detections=2)
    filtered = nms.apply(detections)
    
    print(f"原始检测数量: {len(detections)}")
    print(f"过滤后检测数量: {len(filtered)}")
    
    for i, detection in enumerate(filtered, 1):
        print(f"检测 #{i}: 置信度={detection['confidence']:.3f}, "
              f"中心={detection['center']}, 尺寸={detection['size']}")


def test_exporter():
    """测试结果导出器"""
    print("\n测试结果导出器...")
    
    # 创建测试检测结果
    detections = [
        {
            'center': (50.0, 50.0, 50.0),
            'size': (20.0, 18.0, 22.0),
            'confidence': 0.92,
            'feature': np.random.randn(256).tolist()
        },
        {
            'center': (120.0, 80.0, 60.0),
            'size': (12.0, 10.0, 14.0),
            'confidence': 0.87,
            'feature': np.random.randn(256).tolist()
        }
    ]
    
    # 创建导出器
    output_dir = Path('./test_output')
    exporter = ResultsExporter(output_dir)
    
    # 测试单个文件导出
    ct_file = Path('test_ct.mhd')
    processing_time = 12.5
    
    json_path = exporter.export_detections(detections, ct_file, processing_time)
    print(f"JSON结果导出到: {json_path}")
    
    # 测试批量汇总导出
    all_results = [
        {
            'file': 'test1.mhd',
            'seriesuid': 'test1',
            'detections': detections,
            'processing_time': 12.5
        },
        {
            'file': 'test2.mhd',
            'seriesuid': 'test2',
            'detections': [],
            'processing_time': 8.3
        }
    ]
    
    summary_path = exporter.export_summary(all_results, 20.8)
    print(f"汇总结果导出到: {summary_path}")
    
    # 清理测试文件
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    print("测试完成!")


if __name__ == "__main__":
    test_nms()
    test_exporter()