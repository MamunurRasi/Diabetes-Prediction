from pydantic import BaseModel

class InputFeatures(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

class OutputPrediction(BaseModel):
    prediction: int
    probability: float
