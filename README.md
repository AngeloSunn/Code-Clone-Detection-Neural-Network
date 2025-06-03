# 🧠 Code-Clone-Classifier-Neural-Network

A neural network project to classify code clones. This is just an experimental setup using ~1000 AI-generated code pair samples for training and evaluation.

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Code-Clone-Classifier-Neural-Network.git
cd Code-Clone-Classifier-Neural-Network
```

Download TrainigData: 
https://upload.uni-jena.de/data/68244f0a2b2ee3.83254283/Data9010CDLH.zip
unzip into Data9010CDLH/

Download BigCloneBench:
https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBaFhiTTZNS3RfeUxqX3RrMjlHSm5jOUJLb0l2Q2c%5FZT1vVlRWSm0&cid=8BFCB70AA333DB15&id=8BFCB70AA333DB15%21261604&parId=8BFCB70AA333DB15%21260467&o=OneUp
untar into BigCloneBench/

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

after that, use
```bash
python predict.py
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
