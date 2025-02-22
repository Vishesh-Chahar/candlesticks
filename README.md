

# 📈 Stock Price Predictor

A deep learning-powered web application that predicts stock prices using LSTM models. Built with **Streamlit**, **PyTorch**, and **Plotly**, this project allows users to train models with historical stock data and make future price predictions.

---

## 🚀 Features

- **Train LSTM Models**: Train a stock price prediction model using a custom CSV file or a Kaggle dataset.
- **Interactive Candlestick Charts**: Visualize stock price movements with dynamic **Plotly** charts.
- **Predict Future Prices**: Load a trained model and forecast the next closing price.
- **Data Validation**: Ensures CSV files contain the necessary stock price columns before training or prediction.

---

## 🛠️ Technologies Used

- **Streamlit**: For the web-based user interface.
- **PyTorch**: For deep learning model training and inference.
- **Pandas & NumPy**: For data processing.
- **Plotly**: For interactive stock charts.
- **Kaggle API**: To fetch the latest stock price datasets.

---

## 📂 Project Structure

```
📦 stock-price-predictor
├── 📜 app.py              # Streamlit web app
├── 📜 candle.py           # Model training & data preprocessing
├── 📜 requirements.txt    # Python dependencies
├── 📜 stock_predictor.pth # Trained model (optional)
├── 📜 README.md           # Project documentation
```

---

## ⚡ Getting Started

### 1️⃣ Install Dependencies

Ensure you have Python **3.8+** installed. Then, run:

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 📊 Usage

### 📌 Training a Model

1. Choose between **uploading a custom CSV** or **using a Kaggle dataset**.
2. Ensure the dataset contains:
   - `Open`, `High`, `Low`, `Close`, `Volume` columns.
3. Set training parameters (learning rate, batch size, epochs).
4. Click **"Train Model"** and monitor progress.
5. Once training is complete, the model is saved as `stock_predictor.pth`.

### 📌 Making Predictions

1. Upload a CSV file with historical stock data.
2. The trained LSTM model predicts the **next closing price**.
3. The prediction is displayed alongside a **candlestick chart**.

---

## 📈 Model Architecture

The model is a **bidirectional LSTM** with:
- **2 LSTM layers**
- **64 hidden units**
- **LeakyReLU activation**
- **Fully connected output layer**

---

## 📦 Dataset

The model supports any stock dataset with:
- `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- CSV format
- Time series ordered by date

To use the Kaggle dataset:
1. Ensure `kagglehub` is installed.
2. It automatically downloads `"World-Stock-Prices-Dataset.csv"`.

---

## 🔥 Future Enhancements

- 📊 Add more technical indicators (MACD, Bollinger Bands).
- 🚀 Hyperparameter tuning with **Optuna**.
- 🤖 Deploy with **Docker** and **Heroku**.

---

## 🤝 Contributing

Pull requests are welcome! To contribute:

1. Fork the repo
2. Create a branch (`feature-xyz`)
3. Commit your changes
4. Open a PR 🎉

---

## 🛡️ License

MIT License © 2025 Stock Price Predictor

---

🚀 Happy Trading! 📈


This `README.md` covers everything from setup to model details. Let me know if you'd like any tweaks! 🚀
