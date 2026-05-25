# image classifier — EfficientNet-B0

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c?logo=pytorch)

learning transfer learning with PyTorch. fine-tuned EfficientNet-B0 on a small animal dataset and put a streamlit UI on top so you can upload images and get predictions.

---

## results

tested on 5 animal classes (~1000 images each):

| phase | val accuracy |
|---|---|
| just train the head, 5 epochs | 88.3% |
| unfreeze everything, 10 more epochs | **94.1%** |

took ~12 min on my RTX 3060. CPU works too, just slower.

---

## how to use it

data folder should look like:
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

## tests

```bash
pytest tests/ -v
```

just checks model output shapes and frozen layers. no GPU needed.

---

## why EfficientNet

tried ResNet first because it's the classic choice. then tried EfficientNet:
- ~2% better accuracy
- 30% faster training
- 20mb saved model instead of 98mb

for a web demo where you wait for the model to load, 20mb is way nicer than 98mb. switched and didn't look back.

---

## deploy free on hugging face spaces

create a streamlit space, upload `app/`, `src/`, `requirements.txt`, and your `model/` folder. that's it.

---

PyTorch, torchvision, Streamlit
