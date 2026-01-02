# Co-Finetuning YOLO-World on Custom Dataset + LVIS

This directory contains configurations for co-finetuning YOLO-World on a mixture of **LVIS v1.0** and your **Custom Dataset**.

## Prerequisites

1. **LVIS Dataset**:
   - Download LVIS v1.0 images and annotations.
   - Annotations should be at `data/lvis/annotations/lvis_v1_train.json`.
   - Images should be at `data/lvis/train2017/`.

2. **Custom Dataset**:
   - Format your dataset in **COCO format**.
   - Annotations: `data/custom/annotations/custom_dataset.json`.
   - Images: `data/custom/images/`.

3. **Text Prompts**:
   - **LVIS**: Run the tool to generate text prompts:
     ```bash
     python tools/generate_lvis_class_texts.py data/lvis/annotations/lvis_v1_train.json data/texts/lvis_v1_class_texts.json
     ```
   - **Custom**: Create a JSON file `data/texts/custom_texts.json` containing a list of class names (list of lists) corresponding to your category IDs.
     - Example: `[["cat"], ["dog"], ["car"]]` (for classes 0, 1, 2).

## Training

Choose the configuration corresponding to your desired model size (S, M, L, X).

**Example (Large Model):**
```bash
python tools/train.py configs/cofinetune/yolo_world_v2_l_cofinetune_custom_lvis.py
```

## Note on Hyperparameters

- **Learning Rate**: Set to `2e-4` to preserve pretrained knowledge.
- **Epochs**: Default is 80. Adjust `max_epochs` in the config if needed.
- **Num Classes**: The configs assume 1203 (LVIS) + 80 (Custom) = 1283 classes. **You MUST update `num_classes` in the config file to match your actual total class count.**
