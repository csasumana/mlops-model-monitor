from typing import Optional

from pydantic import BaseModel,ConfigDict


class PredictionRequest(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prediction: int
    probability: float
    churn_label: str
    model_source: str
    registered_model_name: str
    registered_model_version: Optional[str] = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    model_source: str
    registered_model_name: str | None = None
    registered_model_version: str | None = None