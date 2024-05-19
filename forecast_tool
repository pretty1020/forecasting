import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
import plotly.graph_objects as go
import base64


# Function to perform moving average forecasting
def moving_average_forecast(data, value_column, horizon=12):
    forecast = data[value_column].rolling(window=horizon, min_periods=1).mean().iloc[-horizon:]
    return forecast


# Function to perform exponential smoothing forecasting (Simple Exponential Smoothing)
def simple_exponential_smoothing_forecast(data, value_column, horizon=12):
    model = ExponentialSmoothing(data[value_column], trend=None, seasonal=None)
    forecast = model.fit(smoothing_level=0.2).forecast(steps=horizon)
    return forecast


# Function to perform exponential smoothing forecasting (Holt-Winters)
def exponential_smoothing_forecast(data, value_column, horizon=12):
    model = ExponentialSmoothing(data[value_column], trend="add", seasonal="add", seasonal_periods=12)
    forecast = model.fit().forecast(steps=horizon)
    return forecast


# Function to perform ARIMA forecasting
def arima_forecast(data, value_column, horizon=12):
    model = ARIMA(data[value_column], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=horizon)
    return forecast


# Function to perform SARIMA forecasting
def sarima_forecast(data, value_column, horizon=12):
    model = SARIMAX(data[value_column], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=horizon)
    return forecast


# Function to perform Seasonal Decomposition of Time Series (STL) forecasting
def stl_forecast(data, value_column, horizon=12):
    model = STL(data[value_column], seasonal=13)
    result = model.fit()
    forecast = result.forecast(steps=horizon)
    return forecast


# Function to calculate error measures (MAE and RMSE)
def calculate_error(actual, forecast):
    mae = np.mean(np.abs(actual - forecast))
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    return mae, rmse


# Function to plot historical and forecasted values
def plot_forecast(data, forecast, method):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data[method], mode="lines", name="Historical"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode="lines", name="Forecasted"))
    fig.update_layout(title=f"{method} Forecast", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig)


def main():
    st.set_page_config(
        page_title="Simple Time Series Forecasting Tool",
        page_icon="📈",
        layout="centered",
    )

    # Custom CSS to set background color
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f0f0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Simple Time Series Forecasting Tool")

    # User guide attachment
    with open("User_Guide.pdf", "rb") as f:

        st.markdown(
            f'<a href="data:application/pdf;base64,{base64.b64encode(f.read()).decode()}" download="user_guide.pdf">Download User Guide</a>',
            unsafe_allow_html=True)

    # Sample data attachment
    with open("sample_data.csv", "rb") as f:

        st.markdown(
            f'<a href="data:file/csv;base64,{base64.b64encode(f.read()).decode()}" download="sample_data.csv">Download Sample Data</a>',
            unsafe_allow_html=True)

    st.write("Upload a CSV file with time series data.")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.subheader("**Data Preview**")
        st.write(data.head())

        date_column = st.selectbox("Select date column", data.columns)
        value_column = st.selectbox("Select value column", data.columns)

        data[date_column] = pd.to_datetime(data[date_column])
        data.set_index(date_column, inplace=True)

        st.subheader("**Choose Forecasting Method**")
        forecasting_methods = ["Simple Exponential Smoothing", "Exponential Smoothing", "ARIMA",
                               "SARIMA", "Moving Average"]
        selected_method = st.selectbox("Select forecasting method", forecasting_methods)

        horizon = st.number_input("Enter the number of steps ahead to forecast", min_value=1, max_value=120, value=12)

        if st.button("Perform Forecasting"):

            if selected_method == "Simple Exponential Smoothing":
                forecast = simple_exponential_smoothing_forecast(data, value_column, horizon)
            elif selected_method == "Exponential Smoothing":
                forecast = exponential_smoothing_forecast(data, value_column, horizon)
            elif selected_method == "ARIMA":
                forecast = arima_forecast(data, value_column, horizon)
            elif selected_method == "SARIMA":
                forecast = sarima_forecast(data, value_column, horizon)
            elif selected_method == "STL":
                forecast = stl_forecast(data, value_column, horizon)
            elif selected_method == "Moving Average":
                forecast = moving_average_forecast(data, value_column, horizon)

            st.subheader("**Forecasted Values**")
            st.write(forecast)

            actual_values = data[value_column].values[-horizon:]
            mae, rmse = calculate_error(actual_values, forecast)

            st.subheader("**Error Measures**")
            st.write(f"Mean Absolute Error (MAE): {mae}")
            st.write(f"Root Mean Squared Error (RMSE): {rmse}")

            plot_forecast(data, forecast, value_column)

            # Download forecasted values
            csv = forecast.to_csv(index=True)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="forecast.csv">Download Forecasted Values CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
