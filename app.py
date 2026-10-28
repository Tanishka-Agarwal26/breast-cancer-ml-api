
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load the saved model and feature names
saved_data = joblib.load("model.joblib")

model = saved_data["model"]
feature_names = saved_data["feature_names"]

app = FastAPI(
    title="Breast Cancer Classification API",
    description="Educational ML prediction service",
    version="1.0.0"
)


class PredictionInput(BaseModel):
    features: list[float] = Field(
        ...,
        description="Numerical features in the exact training order"
    )


@app.get("/")
def home():
    return {"message": "Breast Cancer ML API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_count": len(feature_names)
    }


@app.post("/predict")
def predict(data: PredictionInput):

    if len(data.features) != len(feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_names)} features"
        )

    prediction = model.predict([data.features])[0]

    probability = model.predict_proba([data.features])[0]
    classes = list(model.classes_)

    confidence = float(
        probability[classes.index(prediction)]
    )

    diagnosis = (
        "Malignant" if prediction == "M" else "Benign"
    )

    return {
        "prediction": str(prediction),
        "diagnosis": diagnosis,
        "confidence": round(confidence, 4),
        "model": "Logistic Regression"
    }
