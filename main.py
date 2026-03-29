from fastapi import FastAPI, UploadFile
import pickle
from feature_extractor import extract_features

app = FastAPI()

model = pickle.load(open("svm_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

@app.post("/predict")
async def predict(file: UploadFile):

    with open("temp.wav","wb") as f:
        f.write(await file.read())

    features = extract_features("temp.wav")

    features = scaler.transform([features])

    prediction = model.predict(features)

    result = "Parkinson Detected" if prediction[0]==1 else "Healthy"

    return {"prediction": result}