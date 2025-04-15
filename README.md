# 🧠 Code-Clone-Classifier-Neural-Network

A neural network project to classify code clones. This is just an experimental setup using ~1000 AI-generated code pair samples for training and evaluation.

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Code-Clone-Classifier-Neural-Network.git
cd Code-Clone-Classifier-Neural-Network
```

### 2. Set Up a Python Virtual Environment

> Requires Python 3.8+

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

Make sure `pip` is up to date:

```bash
pip install --upgrade pip
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧪 Training & Running

Once everything is set up:

```bash
python training.py
```

Make sure to modify any training parameters inside `train.py` or a config file if provided.

---

## 📁 Project Structure

```
├── data/                 # Dataset of code pairs
├── models/               # Saved models and checkpoints
├── train.py              # Main training loop
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

---

## 📌 Notes

- GPU strongly recommended for training. See the troubleshooting section below if you run into CUDA memory errors.
- Dataset is AI-generated and small-scale; results may vary.

---

## 🛠️ Troubleshooting

### CUDA Out of Memory?

Try:
- Lowering batch size in your training loop
- Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Running training on CPU (slower but safer)

---
