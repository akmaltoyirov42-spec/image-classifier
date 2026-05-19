# Image Classifier — EfficientNet-B0

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c?logo=pytorch)

Fine-tuned EfficientNet-B0 for image classification with a Streamlit UI. Upload an image, get the top-5 predictions with confidence scores.

---

## Results

Tested on a 5-class animal dataset (~1000 images per class):

| Phase | Val Accuracy |
|---|---|
| Head only — 5 epochs | 88.3% |
| Full fine-tune — 10 epochs | **94.1%** |

Training took ~12 min on RTX 3060. Works on CPU too, just slower.

---

## How to use

Data folder structure:
```
data/
├── train/
│   ├── class_name/
│   └── ...
└── val/
    ├── class_name/
    └── ...
```

```bash
git clone https://github.com/akmaltoyirov42-spec/image-classifier.git
cd image-classifier

pip install -r requirements.txt

python -m src.train
streamlit run app/streamlit_app.py
```

## Tests

```bash
pytest tests/ -v
```

Checks model output shapes and frozen/unfrozen layers — no GPU needed.

---

## Why EfficientNet over ResNet

Compared both on the same dataset. EfficientNet got ~2% better accuracy, trained 30% faster, and the saved model is 20MB vs 98MB. For a web demo that difference matters.

---

## Deploy free on Hugging Face Spaces

Create a Streamlit Space, upload `app/`, `src/`, `requirements.txt`, and your `model/` folder. Done.

---

## Stack

PyTorch, torchvision, Streamlit
