# image classifier — EfficientNet-B0

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c?logo=pytorch)

fine-tuned EfficientNet-B0 on a small animal dataset using two-phase transfer learning. streamlit UI on top so you can upload images and see top-5 predictions.

---

## results

tested on 5 animal classes (~1000 images each):

| phase | val accuracy |
|---|---|
| train head only, 5 epochs | 88.3% |
| unfreeze all, 10 more epochs | **94.1%** |

took ~12 min on RTX 3060. CPU works too, just slower.

---

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

ResNet was the obvious choice but EfficientNet-B0 turned out better:
- ~2% higher accuracy
- 30% faster training
- 20mb saved model instead of 98mb

for a web demo where you wait for the model to load, 20mb is much nicer.

---

## deploy free on hugging face spaces

create a streamlit space, upload `app/`, `src/`, `requirements.txt`, and your `model/` folder. done.

---

## what's next

want to add Grad-CAM visualization so you can see which part of the image the model is looking at. also planning to swap in EfficientNet-B3 for a bigger accuracy bump and benchmark the trade-off.

---

PyTorch, torchvision, Streamlit
