from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = FastAPI()

class StockData(BaseModel):
    symbol: str
    start_date: str
    end_date: str

@app.post("/predict/")
async def predict_stock_price(stock_data: StockData):
    symbol = stock_data.symbol
    start_date = stock_data.start_date
    end_date = stock_data.end_date
    
    # Load the trained model
    model = load_model('my_model.h5')  # Load the model inside the endpoint function
    
    # Download historical data
    data = yf.download(symbol, start=start_date, end=end_date)
    data = data.reset_index()
    data = data.drop(['Date','Adj Close'], axis = 1)    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Prepare input data
    x_test = []
    for i in range(100, scaled_data.shape[0]):
        x_test.append(scaled_data[i - 100:i, -1:])  # Using only the last column (closing price)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Reshape for LSTM input

    print(x_test.shape)

    # Make predictions
    predictions = model.predict(x_test)
    
    # Inverse scaling
    scale_factor = 1 / 0.00682769
    predictions = predictions * scale_factor
    
    return predictions.tolist()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
