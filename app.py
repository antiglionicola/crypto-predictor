from fastapi import FastAPI
import requests
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Crypto Predictor API is running"}

@app.get("/predict/{coin_id}")
def predict(coin_id: str):
    try:
        # Ottiene i dati reali dal mercato (ultimi 7 giorni)
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=7"
        data = requests.get(url).json()

        prices = [p[1] for p in data["prices"]]
        X = np.arange(len(prices)).reshape(-1, 1)
        y = np.array(prices)

        # Modello semplice di regressione lineare per predizione
        model = LinearRegression().fit(X, y)
        next_day = model.predict([[len(prices)]])[0]

        return {
            "coin": coin_id,
            "last_price": round(float(prices[-1]), 2),
            "predicted_next_price": round(float(next_day), 2)
        }

    except Exception as e:
        return {"error": str(e)}

