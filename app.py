from fastapi import FastAPI
from pydantic import BaseModel
import pickle

from main import vectorizer

app = FastAPI()

with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)


class Query(BaseModel):
    text: str


@app.post("/classify")
async def classify_text(query: Query):
    text_vectorized = vectorizer.transform([query.text])
    prediction = model.predict(text_vectorized)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]
    return {"predicted_label": predicted_label}

