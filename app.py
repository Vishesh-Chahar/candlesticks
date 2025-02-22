import streamlit as st
import torch
import pandas as pd
import plotly.graph_objects as go
from candle import load_kaggle_data, CandlestickDataset, LSTMPredictor, device, train_model, save_model
from torch.utils.data import DataLoader
import torch.nn as nn

def create_candlestick_chart(df, timeframe):
    """Create an interactive candlestick chart using plotly"""
    df_subset = df.tail(timeframe)
    
    fig = go.Figure(data=[go.Candlestick(x=df_subset.index,
                open=df_subset['Open'],
                high=df_subset['High'],
                low=df_subset['Low'],
                close=df_subset['Close'])])
    
    fig.update_layout(
        title='Stock Price Candlestick Chart',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark'
    )
    
    return fig

def load_model(path='stock_predictor.pth'):
    """Load the trained model."""
    try:
        checkpoint = torch.load(path, map_location=device)
        model_architecture = checkpoint['model_architecture']
        
        model = LSTMPredictor(
            input_size=model_architecture['input_size'],
            hidden_size=model_architecture['hidden_size'],
            num_layers=model_architecture['num_layers']
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict(model, dataset, last_sequence):
    """Make prediction using the trained model."""
    model.eval()
    with torch.no_grad():
        sequence_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
        prediction = model(sequence_tensor)
        # Move prediction to CPU before converting to numpy
        prediction = prediction.cpu()
        # Inverse transform the prediction
        prediction = dataset.scaler.inverse_transform(
            [[0] * 3 + [prediction.item()] + [0] * 7]
        )[0, 3]  # Get the Close price
        return prediction

def validate_csv_columns(df):
    """Validate that the CSV has all required columns."""
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    return missing_columns

def main():
    st.title('Stock Price Predictor')
    
    # Initialize session state for DataFrame if it doesn't exist
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    # Create tabs for training and inference
    tab1, tab2 = st.tabs(["Train Model", "Make Predictions"])
    
    with tab1:
        st.header("Train Model")
        
        # Add data source selection
        data_source = st.radio(
            "Select Data Source",
            ["Upload Custom CSV", "Use Kaggle Dataset"]
        )
        
        # Display CSV requirements
        st.info("""
        📊 Required CSV columns:
        - Open: Opening price
        - High: Highest price
        - Low: Lowest price
        - Close: Closing price
        - Volume: Trading volume
        
        Make sure your CSV file includes these columns with proper numerical values.
        """)
        
        # Sidebar controls
        st.sidebar.header('Training Parameters')
        learning_rate = st.sidebar.slider('Learning Rate', 1, 100, 4)
        learning_rate = learning_rate/10000
        batch_size = st.sidebar.slider('Batch Size', 8, 64, 32)
        num_epochs = st.sidebar.slider('Number of Epochs', 100, 2000, 1000)
        
        # Visualization controls
        st.sidebar.header('Visualization Parameters')
        candlestick_timeframe = st.sidebar.slider('Candlestick Timeframe', 10, 200, 50)
        
        # Handle data loading based on source selection
        if data_source == "Upload Custom CSV":
            uploaded_file = st.file_uploader("Upload training data CSV", type=['csv'])
            if uploaded_file is not None:
                try:
                    st.session_state.df = pd.read_csv(uploaded_file)
                    missing_columns = validate_csv_columns(st.session_state.df)
                    
                    if missing_columns:
                        st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                        st.session_state.df = None
                    else:
                        st.success("✅ CSV file validated successfully!")
                        st.write("Preview of uploaded data:")
                        st.dataframe(st.session_state.df.head())
                    
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                    st.session_state.df = None
        else:  # Kaggle Dataset
            if st.button("Load Kaggle Dataset"):
                with st.spinner('Loading Kaggle dataset...'):
                    try:
                        st.session_state.df = load_kaggle_data()
                        st.success("✅ Kaggle dataset loaded successfully!")
                        st.write("Preview of Kaggle data:")
                        st.dataframe(st.session_state.df.head())
                    except Exception as e:
                        st.error(f"Error loading Kaggle dataset: {str(e)}")
                        st.session_state.df = None
        
        # Only show training button if we have data
        if st.session_state.df is not None:
            if st.button('Train Model'):
                with st.spinner('Training model...'):
                    try:
                        # Display candlestick chart
                        st.subheader('Stock Price Chart')
                        fig = create_candlestick_chart(st.session_state.df, candlestick_timeframe)
                        st.plotly_chart(fig)
                        
                        # Create and train model
                        dataset = CandlestickDataset(st.session_state.df)
                        
                        # Split data
                        train_size = int(len(dataset) * 0.8)
                        test_size = len(dataset) - train_size
                        train_dataset, test_dataset = torch.utils.data.random_split(dataset, 
                                                                                  [train_size, test_size])
                        
                        # Create data loaders
                        train_loader = DataLoader(train_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=True)
                        test_loader = DataLoader(test_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=False)
                        
                        # Initialize model
                        input_size = 11  # Number of features
                        hidden_size = 64
                        num_layers = 2
                        
                        model = LSTMPredictor(input_size, hidden_size, num_layers).to(device)
                        
                        # Loss and optimizer
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                        
                        # Training progress bar and status for Streamlit
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Custom progress callback for Streamlit
                        def progress_callback(epoch, num_epochs, loss):
                            progress = (epoch + 1) / num_epochs
                            progress_bar.progress(progress)
                            status_text.text(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}')

                        # Train the model using the function from candle.py
                        training_losses = train_model(
                            model=model,
                            train_loader=train_loader,
                            criterion=criterion,
                            optimizer=optimizer,
                            num_epochs=num_epochs,
                            progress_callback=progress_callback
                        )
                        
                        st.success('✅ Training completed successfully!')
                        
                        # Save the model
                        save_model(model)
                        st.success('✅ Model saved successfully!')
                        
                        # Plot training loss
                        st.subheader('Training Loss Over Time')
                        st.line_chart(pd.DataFrame(training_losses, columns=['Loss']))
                        
                    except Exception as e:
                        st.error(f'❌ An error occurred during training: {str(e)}')
        else:
            if data_source == "Upload Custom CSV":
                st.warning("Please upload a CSV file to proceed with training")
            else:
                st.info("Click the button to load the Kaggle dataset")

    with tab2:
        st.header("Make Predictions")
        
        # Initialize prediction session state if needed
        if 'prediction_df' not in st.session_state:
            st.session_state.prediction_df = None
        if 'prediction_result' not in st.session_state:
            st.session_state.prediction_result = None
        
        uploaded_file = st.file_uploader("Upload CSV file for prediction", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Load the uploaded data
                st.session_state.prediction_df = pd.read_csv(uploaded_file)
                
                # Check required columns
                missing_columns = validate_csv_columns(st.session_state.prediction_df)
                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                    st.session_state.prediction_df = None
                else:
                    st.success("✅ CSV file validated successfully!")
                    st.write("Preview of prediction data:")
                    st.dataframe(st.session_state.prediction_df.head())
                    
                    # Create dataset from uploaded data
                    dataset = CandlestickDataset(st.session_state.prediction_df)
                    
                    # Load the trained model
                    model = load_model()
                    if model is None:
                        st.error("❌ Please train a model first!")
                        return
                    
                    # Get the last sequence for prediction
                    last_sequence = dataset.X[-1].cpu().numpy()
                    
                    # Make prediction
                    st.session_state.prediction_result = predict(model, dataset, last_sequence)
                    
                    # Display results
                    st.subheader("Prediction Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Last Close Price",
                            f"${st.session_state.prediction_df['Close'].iloc[-1]:.2f}",
                            delta=None
                        )
                    
                    with col2:
                        delta = st.session_state.prediction_result - st.session_state.prediction_df['Close'].iloc[-1]
                        delta_percent = (delta / st.session_state.prediction_df['Close'].iloc[-1]) * 100
                        st.metric(
                            "Predicted Next Close",
                            f"${st.session_state.prediction_result:.2f}",
                            f"{delta_percent:+.2f}%"
                        )
                    
                    # Display candlestick chart with prediction
                    st.subheader("Price Chart with Prediction")
                    fig = create_candlestick_chart(st.session_state.prediction_df, 30)  # Show last 30 days
                    # Add prediction point
                    fig.add_scatter(
                        x=[st.session_state.prediction_df.index[-1] + 1],
                        y=[st.session_state.prediction_result],
                        mode='markers',
                        marker=dict(size=10, color='yellow'),
                        name='Prediction'
                    )
                    st.plotly_chart(fig)
                    
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.session_state.prediction_df = None
                st.session_state.prediction_result = None

if __name__ == '__main__':
    main() 