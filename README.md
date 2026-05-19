# Image Classifier — EfficientNet-B0 Transfer Learning

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c?logo=pytorch)

Fine-tuned EfficientNet-B0 for multi-class image classification, with a Streamlit UI for testing it live.

The training uses a two-phase approach: first only the classification head trains (backbone frozen), then the whole network fine-tunes at a lower learning rate. This way you get decent accuracy fast without needing a massive dataset or GPU hours.

---

## Results

Tested on a 5-class animal dataset (~1000 images per class):

| Phase | Val Accuracy |
|---|---|
| Head only (5 epochs) | 88.3% |
| Full fine-tune (10 epochs) | **94.1%** |

Training took ~12 minutes on a single GPU (RTX 3060). CPU works too, just slower.

---

## How to use

### 1. Prepare your data

Folder structure expected:
```
data/
├── train/
│   ├── class_a/
│   ├── class_b/
│   └── ...
└── val/
    ├── class_a/
    ├── class_b/
    └── ...
```

Works with any image classification dataset in this format. [Kaggle](https://www.kaggle.com/datasets) has hundreds of them.

### 2. Install and train

```bash
git clone https://github.com/akmaltoyirov42-spec/image-classifier.git
cd image-classifier

python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train
```

### 3. Run the app

```bash
streamlit run app/streamlit_app.py
```

---

## Tests

```bash
pytest tests/ -v
```

These don't need a GPU or a trained model — they just check the model architecture (output shapes, frozen layers, etc.).

---

## Why EfficientNet-B0

I compared it against ResNet-50 on the same dataset. EfficientNet got ~2% better accuracy and trained 30% faster. It's also smaller — the saved model is 20MB vs 98MB. For a web demo that matters.

---

## Deploy to Hugging Face Spaces

The app runs as-is on [Hugging Face Spaces](https://huggingface.co/spaces) (free). Steps:
1. Create a new Space, pick Streamlit as the SDK
2. Upload `app/streamlit_app.py`, `src/`, `requirements.txt`, and your trained `model/` folder
3. It goes live automatically

---

## Files

```
├── src/
│   ├── model.py      EfficientNet-B0 with custom head
│   ├── dataset.py    data loading + augmentation
│   ├── train.py      two-phase training loop
│   └── predict.py    inference
├── app/
│   └── streamlit_app.py
└── tests/
    └── test_model.py
```
