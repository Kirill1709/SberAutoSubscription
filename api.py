import dill
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()
with open('car_subscription.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    session_id : str
    client_id: str
    visit_date: str = Field(pattern='\d{4}-\d{2}-\d{2}')
    visit_time: str = Field(pattern='\d{2}:\d{2}:\d{2}')
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str = Field(pattern='\d{3,4}x\d{3,4}')
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    client_id: str
    pred: int

@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)
    return {
        'client_id': form.client_id,
        'pred': y[0],
    }



