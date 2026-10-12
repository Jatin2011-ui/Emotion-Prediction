# 🧠 Emotional Prediction API

> **Detect the emotion behind any sentence using a Bidirectional GRU deep learning model — served via a clean FastAPI backend and an interactive web UI.**

---

## ✨ Overview

This project classifies the **emotion** expressed in a given text into one of six categories using a trained **BiGRU (Bidirectional Gated Recurrent Unit)** neural network. It exposes a REST API via **FastAPI** and ships with a minimal, dark-themed **web interface** for interactive testing.

### 🎭 Supported Emotions

| Emotion   | Emoji |
|-----------|-------|
| Sadness   | 😔    |
| Joy       | 😊    |
| Love      | ❤️    |
| Anger     | 😠    |
| Fear      | 😨    |
| Surprise  | 😮    |

---

## 🗂️ Project Structure

```
emotional-prediction/
│
├── main.py               # FastAPI app — API endpoints, model loading, inference
├── convert_model.py      # Utility: converts Keras 3 .keras model → Keras 2 .h5
├── requirements.txt      # Python dependencies
│
├── Artifacts/
│   ├── BiGRU_model.keras # Original Keras 3 saved model
│   ├── BiGRU_model_v2.h5 # Converted Keras 2.15-compatible model (used at runtime)
│   └── tokenizer.pkl     # Fitted Keras Tokenizer (vocabulary)
│
└── static/
    ├── index.html        # Web UI entry point
    ├── style.css         # Dark-themed UI styles
    └── script.js         # Frontend logic (fetch API calls, result rendering)
```

---

## 🏗️ Model Architecture

The model is a **Stacked Bidirectional GRU** trained on the [Emotions Dataset](https://huggingface.co/datasets/dair-ai/emotion):

```
Input (sequence length: 50)
    │
    ▼
Embedding Layer  (vocab: 10,000 × dim: 300)
    │
    ▼
Bidirectional GRU  (128 units, return_sequences=True)
    │
    ▼
Dropout (0.3)
    │
    ▼
Bidirectional GRU  (64 units)
    │
    ▼
Dropout (0.3)
    │
    ▼
Dense (6 units, softmax)  → Emotion probabilities
```

**Training configuration:**
- Tokenizer vocabulary: 10,000 most frequent words
- Max sequence length: 50 tokens
- Padding: post-padding, post-truncation

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- `pip` or a virtual environment manager

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd emotional-prediction
```

### 2. Create & Activate a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** This project requires `tensorflow==2.15.0` and `numpy<2.0` for compatibility with the saved model format.

### 4. (One-time) Convert the Model

If `Artifacts/BiGRU_model_v2.h5` does not exist yet, run the conversion script to transform the Keras 3 `.keras` model into a Keras 2.15-compatible `.h5` file:

```bash
python convert_model.py
```

This script:
1. Reads weights from `BiGRU_model.keras` (a zip containing `model.weights.h5`)
2. Rebuilds the exact architecture in Keras 2.15
3. Transfers weights layer-by-layer
4. Saves the result to `Artifacts/BiGRU_model_v2.h5`

### 5. Run the Server

```bash
python main.py
```

The server starts at **`http://localhost:8000`** with hot-reload enabled.

---

## 🌐 Web Interface

Open your browser and navigate to:

```
http://localhost:8000
```

Type any sentence and hit **Analyze** to see the predicted emotion along with the full probability breakdown for all six classes.

---

## 📡 API Reference

### `GET /health`

Returns the server and model status.

**Response:**
```json
{
  "status": "server is running",
  "model_loaded": true
}
```

---

### `POST /predict`

Predicts the emotion of a given text.

**Request Body:**
```json
{
  "text": "I am so happy and excited about today!"
}
```

| Field  | Type   | Constraints         | Description             |
|--------|--------|---------------------|-------------------------|
| `text` | string | 1–2000 characters   | The sentence to analyze |

**Response:**
```json
{
  "text": "I am so happy and excited about today!",
  "predicted_emotion": "joy",
  "confidence": 0.892,
  "all_probabilities": {
    "sadness":  0.021,
    "joy":      0.892,
    "love":     0.043,
    "anger":    0.012,
    "fear":     0.018,
    "surprise": 0.014
  }
}
```

---

### `GET /docs`

Interactive Swagger UI for exploring and testing all API endpoints.

```
http://localhost:8000/docs
```

---

## 🧪 Example cURL Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love you so much, you mean everything to me"}'
```

---

## 📦 Dependencies

| Package              | Version     | Purpose                          |
|----------------------|-------------|----------------------------------|
| `fastapi`            | ≥ 0.110.0   | Web framework & REST API         |
| `uvicorn[standard]`  | ≥ 0.29.0    | ASGI server with hot-reload      |
| `pydantic`           | ≥ 2.6.0     | Request/response validation      |
| `tensorflow`         | == 2.15.0   | Deep learning inference (Keras)  |
| `numpy`              | < 2.0       | Numerical operations             |

---

## ⚠️ Known Notes

- **TensorFlow deprecation warnings** on startup are harmless — they relate to TF 2.15 using legacy Keras APIs internally and do not affect functionality.
- **"No training configuration"** warning when loading the `.h5` model is expected — the model is used for inference only and does not need to be compiled.
- The **`love`** and **`surprise`** classes can occasionally be confused with `sadness` and `anger` respectively, depending on phrasing — this is a known trait of the training data distribution.

---

## 📄 License

This project is intended for educational and internship purposes.
