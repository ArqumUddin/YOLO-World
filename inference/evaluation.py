import json
import os
import io
import re
import contextlib
from typing import Dict, Any, List, Optional, Tuple
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

from .detection_result import FrameDetections, Detection, BoundingBox


class COCOEvaluator:
    """
    COCO-style evaluation for YOLO-World detections.

    - Full COCOEval metrics for datasets
    - Per-image metrics (precision/recall/F1/IoU) at configurable IoU thresholds
    """

    def __init__(
        self,
        annotations_path: str,
        iou_thresholds: Optional[List[float]] = None,
        min_score: Optional[float] = None,
        per_image_metrics: bool = True,
    ) -> None:
        self.annotations_path = annotations_path
        self.iou_thresholds = iou_thresholds or [0.5]
        self.min_score = min_score
        self.per_image_metrics = per_image_metrics

        self.coco = COCO(annotations_path)
        self.cat_name_to_id = self._build_category_maps()
        self.cat_id_to_name = {
            category_id: category_name
            for category_name, category_id in self.cat_name_to_id.items()
        }
        self.image_id_by_file_name = self._build_image_id_maps()

    def _build_category_maps(self) -> Dict[str, int]:
        categories = self.coco.loadCats(self.coco.getCatIds())
        category_name_to_id: Dict[str, int] = {}
        for category in categories:
            category_name = str(category.get('name', '')).strip().lower()
            if category_name:
                category_name_to_id[category_name] = int(category['id'])
        return category_name_to_id

    def _build_image_id_maps(self) -> Dict[str, int]:
        image_id_by_file_name: Dict[str, int] = {}
        for image_record in self.coco.dataset.get('images', []):
            file_name = image_record.get('file_name')
            if not file_name:
                continue
            image_id_by_file_name[file_name] = int(image_record['id'])
        return image_id_by_file_name

    def _resolve_image_id(self, file_name: str) -> int:
        if file_name in self.image_id_by_file_name:
            return self.image_id_by_file_name[file_name]
        else:
            raise ValueError(f"File Name {file_name} does not exist in current mapping")

    def _resolve_category_id(self, class_name: str) -> Optional[int]:
        if class_name is None:
            return None
        key = str(class_name).strip().lower()
        return self.cat_name_to_id.get(key)

    def _detections_to_coco_results(
        self,
        detections_by_filename: Dict[str, FrameDetections],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        coco_results: List[Dict[str, Any]] = []
        conversion_stats = {
            "unknown_categories": 0,
            "unknown_category_examples": [],
            "missing_images": 0,
            "missing_image_examples": [],
        }

        for file_name, frame_detections in detections_by_filename.items():
            image_id = self._resolve_image_id(file_name)
            if image_id is None:
                conversion_stats["missing_images"] += 1
                if len(conversion_stats["missing_image_examples"]) < 5:
                    conversion_stats["missing_image_examples"].append(file_name)
                continue

            for detection in frame_detections.detections:
                category_id = self._resolve_category_id(detection.class_name)
                if category_id is None:
                    conversion_stats["unknown_categories"] += 1
                    if len(conversion_stats["unknown_category_examples"]) < 5:
                        conversion_stats["unknown_category_examples"].append(detection.class_name)
                    continue
                coco_results.append({
                    "image_id": image_id,
                    "category_id": int(category_id),
                    "bbox": self._xyxy_to_xywh(detection.bbox),
                    "score": float(detection.confidence),
                })
        return coco_results, conversion_stats

    def _run_coco_eval(self, coco_results: List[Dict[str, Any]], verbose: bool = True) -> Dict[str, Any]:
        if not coco_results:
            raise ValueError("No detections available for COCO evaluation.")

        coco_detections = self.coco.loadRes(coco_results)
        coco_evaluator = COCOeval(self.coco, coco_detections, iouType='bbox')
        coco_evaluator.evaluate()
        coco_evaluator.accumulate()

        summary_lines = self._capture_coco_summary(coco_evaluator)
        if verbose and summary_lines:
            print("\n".join(summary_lines))

        summary_payload = self._parse_coco_summary(summary_lines)
        summary_payload["stats_raw"] = (
            list(coco_evaluator.stats) if coco_evaluator.stats is not None else []
        )
        per_class = self._per_class_metrics(coco_evaluator)
        return {
            "summary": summary_payload,
            "per_class": per_class,
        }

    def _capture_coco_summary(self, coco_evaluator) -> List[str]:
        stdout_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer):
            coco_evaluator.summarize()
        summary_text = stdout_buffer.getvalue()
        if not summary_text:
            return []
        summary_lines = summary_text.splitlines()
        cleaned_lines = []
        for line in summary_lines:
            if not line.strip():
                continue
            cleaned_lines.append(line.rstrip())
        return cleaned_lines

    def _parse_coco_summary(self, summary_lines: List[str]) -> Dict[str, Any]:
        parsed_metrics = []
        named_metrics: Dict[str, float] = {}
        summary_pattern = re.compile(
            r"Average\s+(Precision|Recall)\s+\((AP|AR)\)\s+@\[\s*IoU=([0-9.]+(?::[0-9.]+)?)\s*"
            r"\|\s*area=\s*([a-zA-Z]+)\s*\|\s*maxDets=\s*([0-9]+)\s*\]\s*=\s*([0-9.]+)"
        )

        for summary_line in summary_lines:
            summary_match = summary_pattern.search(summary_line)
            if not summary_match:
                continue
            _metric_label, metric_kind, iou_range, area_name, max_detections, metric_value = (
                summary_match.groups()
            )
            metric_entry = {
                "type": metric_kind,
                "iou": iou_range,
                "area": area_name,
                "maxDets": int(max_detections),
                "value": float(metric_value),
            }
            parsed_metrics.append(metric_entry)

            if metric_kind == "AP" and area_name == "all" and metric_entry["maxDets"] == 100:
                if iou_range == "0.50:0.95":
                    named_metrics["AP"] = metric_entry["value"]
                elif iou_range == "0.50":
                    named_metrics["AP50"] = metric_entry["value"]
                elif iou_range == "0.75":
                    named_metrics["AP75"] = metric_entry["value"]
            if metric_kind == "AP" and iou_range == "0.50:0.95" and metric_entry["maxDets"] == 100:
                if area_name == "small":
                    named_metrics["AP_small"] = metric_entry["value"]
                elif area_name == "medium":
                    named_metrics["AP_medium"] = metric_entry["value"]
                elif area_name == "large":
                    named_metrics["AP_large"] = metric_entry["value"]
            if metric_kind == "AR" and iou_range == "0.50:0.95" and area_name == "all":
                if metric_entry["maxDets"] == 1:
                    named_metrics["AR_1"] = metric_entry["value"]
                elif metric_entry["maxDets"] == 10:
                    named_metrics["AR_10"] = metric_entry["value"]
                elif metric_entry["maxDets"] == 100:
                    named_metrics["AR_100"] = metric_entry["value"]
            if metric_kind == "AR" and iou_range == "0.50:0.95" and metric_entry["maxDets"] == 100:
                if area_name == "small":
                    named_metrics["AR_small"] = metric_entry["value"]
                elif area_name == "medium":
                    named_metrics["AR_medium"] = metric_entry["value"]
                elif area_name == "large":
                    named_metrics["AR_large"] = metric_entry["value"]

        return {
            "named": named_metrics,
            "metrics": parsed_metrics,
            "lines": summary_lines,
        }

    def _per_class_metrics(self, coco_evaluator) -> Dict[str, Dict[str, float]]:
        precision_tensor = coco_evaluator.eval.get('precision')
        if precision_tensor is None:
            return {}

        iou_thresholds = coco_evaluator.params.iouThrs
        max_detections = coco_evaluator.params.maxDets
        category_ids = coco_evaluator.params.catIds

        try:
            max_detections_index = max_detections.index(100)
        except ValueError:
            max_detections_index = len(max_detections) - 1

        def iou_index(target_iou: float) -> int:
            idx = np.where(np.isclose(iou_thresholds, target_iou))[0]
            return int(idx[0]) if idx.size > 0 else 0

        iou_index_50 = iou_index(0.5)
        iou_index_75 = iou_index(0.75)

        per_class_metrics: Dict[str, Dict[str, float]] = {}
        for category_index, category_id in enumerate(category_ids):
            category_name = self.cat_id_to_name.get(category_id, str(category_id))
            precision_all = precision_tensor[:, :, category_index, 0, max_detections_index]
            precision_all = precision_all[precision_all > -1]
            average_precision = float(np.mean(precision_all)) if precision_all.size else 0.0

            precision_50 = precision_tensor[iou_index_50, :, category_index, 0, max_detections_index]
            precision_50 = precision_50[precision_50 > -1]
            average_precision_50 = float(np.mean(precision_50)) if precision_50.size else 0.0

            precision_75 = precision_tensor[iou_index_75, :, category_index, 0, max_detections_index]
            precision_75 = precision_75[precision_75 > -1]
            average_precision_75 = float(np.mean(precision_75)) if precision_75.size else 0.0

            per_class_metrics[category_name] = {
                "AP": average_precision,
                "AP50": average_precision_50,
                "AP75": average_precision_75,
            }
        return per_class_metrics

    def _compute_per_image_metrics(
        self,
        gt_by_class: Dict[int, List[List[float]]],
        pred_by_class: Dict[int, List[Tuple[List[float], float]]],
    ) -> Dict[str, Dict[str, float]]:
        metrics_by_iou: Dict[str, Dict[str, float]] = {}
        for iou_threshold in self.iou_thresholds:
            true_pos = 0
            false_pos = 0
            false_neg = 0
            iou_total = 0.0
            match_count = 0

            all_category_ids = set(gt_by_class.keys()) | set(pred_by_class.keys())
            for category_id in all_category_ids:
                gt_boxes_for_class = gt_by_class.get(category_id, [])
                predictions_for_class = pred_by_class.get(category_id, [])
                predictions_sorted = sorted(
                    predictions_for_class, key=lambda x: x[1], reverse=True
                )
                matched_gt_indices = set()

                for predicted_box, _pred_score in predictions_sorted:
                    best_iou_for_pred = 0.0
                    best_gt_index = None
                    for gt_index, gt_bbox in enumerate(gt_boxes_for_class):
                        if gt_index in matched_gt_indices:
                            continue
                        iou_value = self._iou(predicted_box, gt_bbox)
                        if iou_value > best_iou_for_pred:
                            best_iou_for_pred = iou_value
                            best_gt_index = gt_index
                    if best_iou_for_pred >= iou_threshold and best_gt_index is not None:
                        true_pos += 1
                        matched_gt_indices.add(best_gt_index)
                        iou_total += best_iou_for_pred
                        match_count += 1
                    else:
                        false_pos += 1

                false_neg += max(0, len(gt_boxes_for_class) - len(matched_gt_indices))

            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            mean_iou = iou_total / match_count if match_count > 0 else 0.0

            metrics_by_iou[f"iou_{iou_threshold:.2f}"] = {
                "tp": int(true_pos),
                "fp": int(false_pos),
                "fn": int(false_neg),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "mean_iou": float(mean_iou),
                "matches": int(match_count),
            }
        return metrics_by_iou

    def _per_image_prf_with_aggregate(
        self,
        detections_by_filename: Dict[str, FrameDetections],
    ) -> Dict[str, Any]:
        per_image_results = []
        aggregate_results: Dict[str, Dict[str, float]] = {}
        aggregate_tp = {thr: 0 for thr in self.iou_thresholds}
        aggregate_fp = {thr: 0 for thr in self.iou_thresholds}
        aggregate_fn = {thr: 0 for thr in self.iou_thresholds}
        aggregate_iou = {thr: 0.0 for thr in self.iou_thresholds}
        aggregate_matches = {thr: 0 for thr in self.iou_thresholds}

        for image_record in self.coco.dataset.get('images', []):
            image_file_name = image_record.get('file_name')
            if not image_file_name:
                continue
            image_id = int(image_record['id'])
            frame_detections = detections_by_filename.get(image_file_name)
            if frame_detections is None:
                frame_detections = FrameDetections(
                    frame_id=image_id,
                    detections=[],
                    frame_width=image_record.get('width'),
                    frame_height=image_record.get('height'),
                )

            annotation_ids = self.coco.getAnnIds(imgIds=[image_id])
            annotations = self.coco.loadAnns(annotation_ids)

            gt_boxes_by_class: Dict[int, List[List[float]]] = {}
            for annotation in annotations:
                if annotation.get('iscrowd', 0):
                    continue
                category_id = int(annotation['category_id'])
                bbox_xyxy = self._bbox_xywh_to_xyxy(annotation['bbox'])
                gt_boxes_by_class.setdefault(category_id, []).append(bbox_xyxy)

            predictions_by_class: Dict[int, List[Tuple[List[float], float]]] = {}
            for detection in frame_detections.detections:
                category_id = self._resolve_category_id(detection.class_name)
                if category_id is None:
                    continue
                if self.min_score is not None and detection.confidence < self.min_score:
                    continue
                bbox_xyxy = [
                    float(detection.bbox.x_min),
                    float(detection.bbox.y_min),
                    float(detection.bbox.x_max),
                    float(detection.bbox.y_max),
                ]
                predictions_by_class.setdefault(category_id, []).append(
                    (bbox_xyxy, float(detection.confidence))
                )

            image_metrics = self._compute_per_image_metrics(gt_boxes_by_class, predictions_by_class)
            per_image_results.append({
                "file_name": image_file_name,
                "image_id": int(image_id),
                "metrics": image_metrics,
            })

            for iou_threshold in self.iou_thresholds:
                threshold_key = f"iou_{iou_threshold:.2f}"
                threshold_metrics = image_metrics[threshold_key]
                aggregate_tp[iou_threshold] += threshold_metrics["tp"]
                aggregate_fp[iou_threshold] += threshold_metrics["fp"]
                aggregate_fn[iou_threshold] += threshold_metrics["fn"]
                aggregate_iou[iou_threshold] += (
                    threshold_metrics["mean_iou"] * threshold_metrics["matches"]
                )
                aggregate_matches[iou_threshold] += threshold_metrics["matches"]

        for iou_threshold in self.iou_thresholds:
            true_pos = aggregate_tp[iou_threshold]
            false_pos = aggregate_fp[iou_threshold]
            false_neg = aggregate_fn[iou_threshold]
            precision = (
                true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
            )
            recall = (
                true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
            )
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            mean_iou = (
                aggregate_iou[iou_threshold] / aggregate_matches[iou_threshold]
                if aggregate_matches[iou_threshold] > 0
                else 0.0
            )
            aggregate_results[f"iou_{iou_threshold:.2f}"] = {
                "tp": int(true_pos),
                "fp": int(false_pos),
                "fn": int(false_neg),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "mean_iou": float(mean_iou),
                "matches": int(aggregate_matches[iou_threshold]),
            }

        return {
            "per_image": per_image_results,
            "aggregate": aggregate_results,
            "min_score": self.min_score,
        }
    
    @staticmethod
    def _xyxy_to_xywh(bbox: BoundingBox) -> List[float]:
        return [
            float(bbox.x_min),
            float(bbox.y_min),
            float(bbox.width),
            float(bbox.height),
        ]
    
    @staticmethod
    def _bbox_xywh_to_xyxy(bbox_xywh: List[float]) -> List[float]:
        x, y, w, h = bbox_xywh
        return [x, y, x + w, y + h]

    @staticmethod
    def _iou(box_a: List[float], box_b: List[float]) -> float:
        a_x_min, a_y_min, a_x_max, a_y_max = box_a
        b_x_min, b_y_min, b_x_max, b_y_max = box_b

        intersection_x_min = max(a_x_min, b_x_min)
        intersection_y_min = max(a_y_min, b_y_min)
        intersection_x_max = min(a_x_max, b_x_max)
        intersection_y_max = min(a_y_max, b_y_max)

        intersection_width = max(0.0, intersection_x_max - intersection_x_min)
        intersection_height = max(0.0, intersection_y_max - intersection_y_min)
        intersection_area = intersection_width * intersection_height
        if intersection_area <= 0:
            return 0.0

        area_a = max(0.0, a_x_max - a_x_min) * max(0.0, a_y_max - a_y_min)
        area_b = max(0.0, b_x_max - b_x_min) * max(0.0, b_y_max - b_y_min)
        union_area = area_a + area_b - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def write_results(output_path: str, results: Dict[str, Any]) -> None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    def evaluate_dataset(
        self,
        detections_by_filename: Dict[str, FrameDetections],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        coco_results, conversion_stats = self._detections_to_coco_results(detections_by_filename)
        coco_eval_results = self._run_coco_eval(coco_results, verbose=verbose)
        per_image_results = (
            self._per_image_prf_with_aggregate(detections_by_filename)
            if self.per_image_metrics
            else None
        )
        return {
            "coco_eval": coco_eval_results,
            "per_image": per_image_results,
            "conversion_stats": conversion_stats,
        }
