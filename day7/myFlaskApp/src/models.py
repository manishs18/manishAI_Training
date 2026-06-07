from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_items=1, description="List of numerical features for prediction")

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    status: str = "success"