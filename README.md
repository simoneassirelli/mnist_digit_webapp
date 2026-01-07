# 🖊️ MNIST Digit Recognizer Web App

An end-to-end machine learning project that trains a convolutional neural network (CNN) on the MNIST handwritten digits dataset and deploys it as an interactive web application.

Users can draw a digit (0–9) directly in the browser and receive real-time predictions from a PyTorch model served via a FastAPI backend.

---

## 🚀 Live Demo

- **Web UI**: https://<YOUR_USERNAME>.github.io/<REPO_NAME>/
- **API**: https://mnist-digit-webapp.onrender.com

> ⚠️ Note: The API may take a few seconds to respond on the first request due to free-tier cold starts.

---

## 🧠 How It Works

1. A user draws a digit on an HTML canvas.
2. The drawing is sent to a FastAPI backend as a PNG image.
3. The image is preprocessed to match MNIST format:
   - Grayscale
   - Inverted colors
   - Cropped, resized to 28×28
   - Centered using center-of-mass
   - Normalized
4. A trained CNN predicts the digit.
5. The prediction and confidence score are returned and displayed in the UI.

---

## 🏗️ Project Structure

```
mnist-digit-webapp/
│
├── artifacts/
│   └── mnist_cnn.pt
├── data/
├── docs/
│   └── index.html
├── src/
│   ├── api/
│   ├── data/
│   ├── models/
│   ├── preprocessing/
│   ├── serving/
│   ├── training/
│   └── utils/
├── render.yaml
├── requirements.txt
└── README.md
```

---

## 🧪 Model Details

- **Dataset**: MNIST
- **Model**: Convolutional Neural Network (CNN)
- **Framework**: PyTorch
- **Input**: 1 × 28 × 28 grayscale image
- **Output**: Digit class (0–9) + confidence score
- **Test Accuracy**: ~99%

---

## 🖥️ Tech Stack

**Machine Learning**
- PyTorch
- Torchvision
- NumPy

**Backend**
- FastAPI
- Uvicorn
- Pillow

**Frontend**
- HTML5 Canvas
- CSS
- JavaScript

**Deployment**
- Render
- GitHub Pages

---

## ▶️ Run Locally

### Clone the repository
```bash
git clone https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
cd <REPO_NAME>
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python -m src.train
```

### Start the API
```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

### Open the UI
Open `docs/index.html` in your browser.

---

## 📌 Key Learning Outcomes

- End-to-end ML pipeline design
- CNN training and evaluation
- Model serving via REST API
- Frontend ↔ backend integration
- Real-world deployment workflow

---

## 📄 License

MIT License
