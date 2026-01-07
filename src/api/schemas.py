from pydantic import BaseModel

# This is clean separation:
# app.py defines behavior
# schemas.py defines the shape of responses
# It ensures your API always responds with consistent fields:
# { "prediction": 7, "confidence": 0.93 }

class PredictResponse(BaseModel):
    prediction: int
    confidence: float
