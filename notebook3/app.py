from fastapi import FastAPI, HTTPException
from app_ml import train_ml, predict_next_close

app = FastAPI(title="Stock ML Service", version="1.0")


@app.get("/predict/{ticker}")
def predict_endpoint(ticker: str):
    try:
        result = predict_next_close(ticker)
        return {"status": "ok", **result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/train/{ticker}")
def train_endpoint(ticker: str, start: str = "2015-01-01"):
    try:
        _, meta = train_ml(ticker, start=start)
        return {"status": "ok", "meta": meta}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
