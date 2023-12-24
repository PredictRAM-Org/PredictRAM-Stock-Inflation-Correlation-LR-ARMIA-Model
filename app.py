import os
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import seaborn as sns

# Load CPI data
cpi_data = pd.read_excel("CPI.xlsx")
cpi_data['Date'] = pd.to_datetime(cpi_data['Date'])
cpi_data.set_index('Date', inplace=True)

# Load stock data
stock_folder = "stock_folder"
stock_files = [f for f in os.listdir(stock_folder) if f.endswith(".xlsx")]

# Function to calculate correlation and build models
def analyze_stock(stock_data, cpi_data, expected_inflation):
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    
    # Merge stock and CPI data on Date
    merged_data = pd.merge(stock_data, cpi_data, left_index=True, right_index=True, how='inner')
    
    # Handle NaN values in CPI column
    if merged_data['CPI'].isnull().any():
        st.write(f"Warning: NaN values found in 'CPI' column for {stock_data.name}. Dropping NaN values.")
        merged_data = merged_data.dropna(subset=['CPI'])

    # Calculate CPI change
    merged_data['CPI Change'] = merged_data['CPI'].pct_change()

    # Drop NaN values after calculating percentage change
    merged_data = merged_data.dropna()

    # Show correlation between 'Close' column and 'CPI Change'
    correlation_close_cpi = merged_data['Close'].corr(merged_data['CPI Change'])
    st.write(f"Correlation between 'Close' and 'CPI Change' for {stock_data.name}: {correlation_close_cpi}")

    # Calculate the adjusted correlation based on expected inflation impact
    adjusted_correlation = correlation_close_cpi + 0.1 * expected_inflation  # You can adjust the multiplier as needed
    st.write(f"Adjusted Correlation with Expected Inflation ({expected_inflation}): {adjusted_correlation}")

    # Train Linear Regression model
    model_lr = LinearRegression()
    X_lr = merged_data[['CPI']]
    y_lr = merged_data['Close']
    model_lr.fit(X_lr, y_lr)

    # Train ARIMA model using auto_arima
    model_arima = auto_arima(y_lr, seasonal=False, suppress_warnings=True)
    
    # Predict future prices based on Linear Regression
    future_prices_lr = model_lr.predict([[expected_inflation]])
    st.write(f"Predicted Price Change for Future Inflation (Linear Regression): {future_prices_lr[0]}")

    # Predict future prices based on ARIMA
    future_prices_arima = model_arima.predict(1).iloc[0]  # 1 is the number of steps to forecast
    st.write(f"Predicted Price Change for Future Inflation (ARIMA): {future_prices_arima}")

    # Display the latest actual price
    latest_actual_price = merged_data['Close'].iloc[-1]
    st.write(f"Latest Actual Price for {stock_data.name}: {latest_actual_price}")

    return correlation_close_cpi, adjusted_correlation, future_prices_lr[0], future_prices_arima, latest_actual_price

# Streamlit UI
st.title("Stock-CPI Correlation Analysis with Expected Inflation and Price Prediction")
expected_inflation = st.number_input("Enter Expected Upcoming Inflation:", min_value=0.0, step=0.01)

# Select tenure for training the model
tenure_options = ['1 month', '3 months', '6 months', '1 year', '3 years', '5 years']
selected_tenure = st.selectbox("Select Tenure for Training Model:", tenure_options)

# Convert tenure to timedelta for filtering data
tenure_mapping = {'1 month': pd.DateOffset(months=1),
                  '3 months': pd.DateOffset(months=3),
                  '6 months': pd.DateOffset(months=6),
                  '1 year': pd.DateOffset(years=1),
                  '3 years': pd.DateOffset(years=3),
                  '5 years': pd.DateOffset(years=5)}

selected_tenure_offset = tenure_mapping[selected_tenure]
end_date = pd.to_datetime("2023-11-30")  # Training till November 2023
start_date = end_date - selected_tenure_offset

train_model_button = st.button("Train Model")

if train_model_button:
    st.write(f"Training model with Expected Inflation: {expected_inflation} and Tenure: {selected_tenure}...")
    
    actual_correlations = []
    expected_correlations = []
    future_prices_lr_list = []
    future_prices_arima_list = []
    latest_actual_prices = []
    stock_names = []

    for stock_file in stock_files:
        st.write(f"\nTraining for {stock_file}...")
        selected_stock_data = pd.read_excel(os.path.join(stock_folder, stock_file))
        selected_stock_data.name = stock_file  # Assign a name to the stock_data for reference
        
        # Filter stock data based on selected tenure
        selected_stock_data = selected_stock_data[(selected_stock_data['Date'] >= start_date) & (selected_stock_data['Date'] <= end_date)]
        
        actual_corr, expected_corr, future_price_lr, future_price_arima, latest_actual_price = analyze_stock(selected_stock_data, cpi_data, expected_inflation)
        
        actual_correlations.append(actual_corr)
        expected_correlations.append(expected_corr)
        future_prices_lr_list.append(future_price_lr)
        future_prices_arima_list.append(future_price_arima)
        latest_actual_prices.append(latest_actual_price)
        stock_names.append(stock_file)

    # Display overall summary in a table
    summary_data = {
        'Stock': stock_names,
        'Actual Correlation': actual_correlations,
        f'Expected Correlation (Inflation={expected_inflation})': expected_correlations,
        'Predicted Price Change (Linear Regression)': future_prices_lr_list,
        'Predicted Price Change (ARIMA)': future_prices_arima_list,
        'Latest Actual Price': latest_actual_prices
    }
    summary_df = pd.DataFrame(summary_data)
    st.write("\nCorrelation and Price Prediction Summary:")
    st.table(summary_df)
