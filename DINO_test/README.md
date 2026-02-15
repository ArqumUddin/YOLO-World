# Feature Reference Evaluation Results

## Dataset Info (Ground Truth Counts)
| Category | Count |
| :--- | :--- |
| chair | 756 |
| bird | 130 |
| grill | 384 |
| umbrella | 224 |
| hospital sign | 41 |

## Experiment Summary

| Experiment | Overall Precision | FP Corrected |
| :--- | :--- | :--- |
| **Exp1 (Baseline YOLO)** | 0.8448 | N/A |
| **Exp2 (Isaac Ref Validation)** | 0.8587 | 92 |
| **Exp3 (Real Ref Validation)** | 0.8578 | 103 |

## Detailed Results

### Exp1: Baseline YOLO
* **Overall Precision**: 0.8448

| Label | Precision | TP | FP |
| :--- | :--- | :--- | :--- |
| chair | 0.8494 | 626 | 111 |
| bird | 0.8651 | 109 | 17 |
| grill | 0.0000 | 0 | 0 |
| umbrella | 0.8169 | 174 | 39 |
| hospital sign | 0.0000 | 0 | 0 |

### Exp2: Isaac Ref Validation
* **Overall Precision**: 0.8587
* **Corrections**: 92

| Label | Precision | TP | FP |
| :--- | :--- | :--- | :--- |
| chair | 0.8832 | 620 | 82 |
| bird | 0.9316 | 109 | 8 |
| grill | 0.3906 | 25 | 39 |
| umbrella | 0.9140 | 170 | 16 |
| hospital sign | 0.0000 | 0 | 7 |

### Exp3: Real Ref Validation
* **Overall Precision**: 0.8578
* **Corrections**: 103

| Label | Precision | TP | FP |
| :--- | :--- | :--- | :--- |
| chair | 0.8674 | 615 | 94 |
| bird | 0.9558 | 108 | 5 |
| grill | 0.4127 | 26 | 37 |
| umbrella | 0.9508 | 174 | 9 |
| hospital sign | 0.0000 | 0 | 8 |
