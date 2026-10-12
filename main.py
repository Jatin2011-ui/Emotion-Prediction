import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hides TF info and warning logs



from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from keras.models import load_model
import numpy as np
import pickle
import re

# 1. Constants
model_path = "Artifacts/BiGRU_model_v2.h5"
tokenizer_path = "Artifacts/tokenizer.pkl"
max_seq_length = 50
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

EMOTION_EMOJIS = {
    "sadness": "😔",
    "joy": "😊",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😮",
}


# 2. Text preprocessing (must match training)
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 3. Schemas
class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I feel so happy and excited"},
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# 4. Model loading and lifespan
d1_model = {}  # {"BiGRU": model, "Tokenizer": tokenizer}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the model and tokenizer...")
    d1_model["BiGRU"] = load_model(model_path)
    # The tokenizer.pkl was saved with Keras 3 which uses the module path
    # keras.src.legacy.preprocessing.text for the Tokenizer class.
    # Keras 2.15 has it at keras.src.preprocessing.text instead.
    # Injecting a sys.modules alias lets pickle find the right class.
    import sys
    import keras.src.preprocessing.text as _tok_module
    sys.modules.setdefault("keras.src.legacy", type(sys)("keras.src.legacy"))
    sys.modules.setdefault("keras.src.legacy.preprocessing", type(sys)("keras.src.legacy.preprocessing"))
    sys.modules["keras.src.legacy.preprocessing.text"] = _tok_module
    with open(tokenizer_path, "rb") as file:
        d1_model["Tokenizer"] = pickle.load(file)
    print("Model loaded successfully.")

    yield  # server runs and waits for requests here

    d1_model.clear()


# 5. App, CORS and static files
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# 6. Endpoints
@app.get("/", include_in_schema=False)
def server_ui():
    return FileResponse("static/index.html")


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="server is running", model_loaded=bool(d1_model))


@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    bigru_model = d1_model.get("BiGRU")
    tokenizer = d1_model.get("Tokenizer")

    if bigru_model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please try again later.",
        )

    # 1. clean
    cleaned_text = preprocess_text(text_input.text)

    # 2. & 3. tokenize and pad
    tokenized_text = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_seq_length,
        padding="post",
        truncating="post",
    )

    # 4. predict
    probabilities = bigru_model.predict(padded_sequence, verbose=0)[0]
    top_index = int(np.argmax(probabilities))

    all_probabilities = {
        label: float(prob) for label, prob in zip(emotion_labels, probabilities)
    }

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=emotion_labels[top_index],
        confidence=float(probabilities[top_index]),
        all_probabilities=all_probabilities,
    )


# 7. Entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)