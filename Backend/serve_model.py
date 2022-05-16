from fastapi import FastAPI
from pydantic import BaseModel
import json
import pickle
import numpy as np


app = FastAPI()

class ClassifyInput(BaseModel):
    model_id: str
    age         : int
    job         : str
    marital     : str
    education   : str
    default     : str
    housing     : str
    loan        : str
    contact     : str
    month       : str
    day_of_week : str
    duration    : int
    campaign    : int
    pdays       : int
    previous    : int
    poutcome    : str
    emp.var.rate  : float
    cons.price.idx: float
    cons.conf.idx : float
    euribor3m     : float
    nr.employed   : float


@app.get('/classify')
async def classify(x_in: ClassifyInput):
    
    data = x_in.dict()
    X_infer = np.array([x for key,x in data.items() if key != 'model_id'])
    X_infer = X_infer.reshape(-1,1)
    loaded_model = pickle.load(open(data['model_id'], 'rb'))
    classification = loaded_model.predict(X_infer)

    classification = {str(x) for x in enumerate(classification[0])}
    classification = json.dumps(classification)
    return classification
