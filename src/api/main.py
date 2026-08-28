from pathlib import Path
import json

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIRECTORY / "category_classifier.onnx"
LABELS_PATH = BASE_DIRECTORY / "labels.json"

MODEL_INPUT_NAME = "text"
PROBABILITIES_OUTPUT_NAME = "probabilities"


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


class PredictionResponse(BaseModel):
    predictedCategory: str
    confidence: float
    confidences: dict[str, float]


app = FastAPI(
    title="Screenshot Category Classification API",
    version="1.0.0",
)

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
)

labels: list[str] = json.loads(
    LABELS_PATH.read_text(encoding="utf-8")
)


@app.get("/")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "screenshot-category-classifier",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text must not be blank.",
        )

    input_tensor = np.asarray([[text]], dtype=object)

    probabilities = session.run(
        [PROBABILITIES_OUTPUT_NAME],
        {MODEL_INPUT_NAME: input_tensor},
    )[0][0]

    if len(probabilities) != len(labels):
        raise HTTPException(
            status_code=500,
            detail="The model output does not match the label list.",
        )

    predicted_index = int(np.argmax(probabilities))
    confidence_values = {
        label: float(probabilities[index])
        for index, label in enumerate(labels)
    }

    return PredictionResponse(
        predictedCategory=labels[predicted_index],
        confidence=float(probabilities[predicted_index]),
        confidences=confidence_values,
    )
