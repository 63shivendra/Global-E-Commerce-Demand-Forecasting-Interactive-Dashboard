import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

# Configure Streamlit page
st.set_page_config(page_title="Sales Forecasting App", page_icon=":bar_chart:", layout="wide")

# Page title with emoji styling
st.markdown("<h1 style='text-align: center;'>📊 Multi-level Sales Forecasting Dashboard 📈</h1>", unsafe_allow_html=True)

# Sidebar for filters
st.sidebar.header("🔧 Filters")

@st.cache_data
def load_data():
    with st.spinner("Loading data..."):
        df = pd.read_csv("dataset.csv", encoding='latin1')
        df["Order Date"] = pd.to_datetime(df["Order Date"], infer_datetime_format=True, errors='coerce')
    return df

df = load_data()

# Debugging: Check date range and invalid dates
st.write(f"Min Order Date: {df['Order Date'].min()}")
st.write(f"Max Order Date: {df['Order Date'].max()}")
na_dates = df["Order Date"].isna().sum()
st.write(f"Number of rows with invalid Order Date: {na_dates}")

# Date range selection
min_date = df["Order Date"].min()
max_date = df["Order Date"].max()
start_date, end_date = st.sidebar.date_input(
    "Select date range:",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)
# Ensure we have two dates
if not isinstance(start_date, (list, tuple)):
    start_date = [start_date, end_date]
start_date, end_date = start_date[0], start_date[-1]

# Region selection
regions = df["Region"].dropna().unique().tolist()
selected_regions = st.sidebar.multiselect("Select region(s):", options=regions, default=regions)

# Forecast horizon (months)
forecast_months = st.sidebar.number_input("Forecast horizon (months):", min_value=1, max_value=24, value=3)

# Apply filters to data
df_filtered = df.copy()
df_filtered = df_filtered[(df_filtered["Order Date"] >= pd.to_datetime(start_date)) & 
                          (df_filtered["Order Date"] <= pd.to_datetime(end_date))]
if selected_regions:
    df_filtered = df_filtered[df_filtered["Region"].isin(selected_regions)]

if df_filtered.empty:
    st.warning("No data available for selected filters. Please adjust the filters.")
    st.stop()

# Get unique categories and sub-categories after filtering
categories = sorted(df_filtered["Category"].unique())
subcats_by_cat = {
    cat: sorted(df_filtered[df_filtered["Category"] == cat]["Sub-Category"].unique())
    for cat in categories
}

# Forecast function using a simple LSTM model
def forecast_series(data, n_input=3, n_out=1):
    series = np.array(data).astype('float32')
    if len(series) < n_input + 1:
        return None, None, None  # not enough data
    scaler = MinMaxScaler(feature_range=(0,1))
    series_scaled = scaler.fit_transform(series.reshape(-1,1))
    X, y = [], []
    for i in range(len(series_scaled) - n_input):
        X.append(series_scaled[i:i+n_input])
        y.append(series_scaled[i+n_input])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_input, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    y_pred_test = model.predict(X_test, verbose=0)
    y_test_orig = scaler.inverse_transform(y_test)
    y_pred_orig = scaler.inverse_transform(y_pred_test)
    rmse = math.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    forecast = []
    input_seq = series_scaled[-n_input:].reshape(1, n_input, 1)
    for _ in range(forecast_months):
        yhat = model.predict(input_seq, verbose=0)
        yhat_orig = scaler.inverse_transform(yhat)[0][0]
        forecast.append(yhat_orig)
        input_seq = np.concatenate([input_seq[:,1:,:], yhat.reshape(1,1,1)], axis=1)
    return forecast, model, rmse

# Create two main tabs: Categories and Sub-Categories
tab1, tab2 = st.tabs(["📂 Category Level Forecast", "📑 Sub-Category Level Forecast"])

with tab1:
    st.header("Category-level Sales Forecast by Category")
    for cat in categories:
        with st.expander(f"📋 Category: {cat}", expanded=False):
            df_cat = df_filtered[df_filtered["Category"] == cat]
            if df_cat.empty:
                st.info(f"No data for category {cat}.")
                continue
            df_cat = df_cat.set_index("Order Date").resample("MS")["Sales"].sum().reset_index()
            df_cat.rename(columns={"Order Date": "Month", "Sales": "Monthly_Sales"}, inplace=True)
            if df_cat.empty:
                st.info(f"No monthly sales data available for category {cat} in the selected period.")
                continue
            fig_hist = px.line(df_cat, x="Month", y="Monthly_Sales", title=f"Historical Sales - {cat}", markers=True)
            fig_hist.update_traces(mode="lines+markers", hovertemplate="Month: %{x}<br>Sales: %{y}")
            st.plotly_chart(fig_hist, use_container_width=True)
            series = df_cat["Monthly_Sales"].values
            forecast_values, model, rmse = forecast_series(series, n_input=3, n_out=1)
            if forecast_values is None:
                st.warning(f"Not enough data to forecast for category {cat}.")
                continue
            last_date = df_cat["Month"].max()
            future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(),
                                         periods=len(forecast_values), freq='MS')
            df_forecast = pd.DataFrame({"Month": future_dates, "Forecast_Sales": forecast_values})
            fig_fore = go.Figure()
            fig_fore.add_trace(go.Scatter(x=df_cat["Month"], y=df_cat["Monthly_Sales"],
                                          mode='lines+markers', name='Actual'))
            fig_fore.add_trace(go.Scatter(x=df_forecast["Month"], y=df_forecast["Forecast_Sales"],
                                          mode='lines+markers', name='Forecast'))
            fig_fore.update_layout(
                title=f"Forecast vs Actual - {cat}", xaxis_title="Month", yaxis_title="Sales"
            )
            st.plotly_chart(fig_fore, use_container_width=True)
            last_sale = df_cat["Monthly_Sales"].iloc[-1]
            next_fore = forecast_values[0]
            delta_pct = (next_fore - last_sale) / last_sale * 100 if last_sale else None
            col1, col2, col3 = st.columns(3)
            col1.metric("Last Month Sales", f"${last_sale:,.0f}")
            col2.metric(f"Next {forecast_months}mo Forecast", f"${next_fore:,.0f}",
                        f"{delta_pct:.1f}%", delta_color="inverse")
            col3.metric("Model RMSE", f"{rmse:.2f}")
            csv = df_forecast.to_csv(index=False)
            st.download_button(
                label="Download Forecast (CSV)",
                data=csv,
                file_name=f"{cat}_forecast.csv",
                mime="text/csv"
            )

with tab2:
    st.header("Sales Forecast by Sub-Category")
    selected_cat = st.selectbox("Select category:", options=categories)
    subcats = subcats_by_cat.get(selected_cat, [])
    if subcats:
        sub_tabs = st.tabs(subcats)
        for idx, sub in enumerate(subcats):
            with sub_tabs[idx]:
                st.subheader(f"Sub-Category: {sub}")
                df_sub = df_filtered[
                    (df_filtered["Category"] == selected_cat) &
                    (df_filtered["Sub-Category"] == sub)
                ]
                if df_sub.empty:
                    st.info(f"No data for sub-category {sub}.")
                    continue
                df_sub = df_sub.set_index("Order Date").resample("MS")["Sales"].sum().reset_index()
                df_sub.rename(columns={"Order Date": "Month", "Sales": "Monthly_Sales"}, inplace=True)
                if df_sub.empty:
                    st.info(f"No monthly sales data available for sub-category {sub} in the selected period.")
                    continue
                fig_hist = px.line(df_sub, x="Month", y="Monthly_Sales",
                                   title=f"Historical Sales - {selected_cat} > {sub}", markers=True)
                fig_hist.update_traces(mode="lines+markers", hovertemplate="Month: %{x}<br>Sales: %{y}")
                st.plotly_chart(fig_hist, use_container_width=True)
                series = df_sub["Monthly_Sales"].values
                forecast_values, model, rmse = forecast_series(series, n_input=3, n_out=1)
                if forecast_values is None:
                    st.warning(f"Not enough data to forecast for sub-category {sub}.")
                    continue
                last_date = df_sub["Month"].max()
                future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(),
                                             periods=len(forecast_values), freq='MS')
                df_forecast = pd.DataFrame({"Month": future_dates, "Forecast_Sales": forecast_values})
                fig_fore = go.Figure()
                fig_fore.add_trace(go.Scatter(x=df_sub["Month"], y=df_sub["Monthly_Sales"],
                                              mode='lines+markers', name='Actual'))
                fig_fore.add_trace(go.Scatter(x=df_forecast["Month"], y=df_forecast["Forecast_Sales"],
                                              mode='lines+markers', name='Forecast'))
                fig_fore.update_layout(
                    title=f"Forecast vs Actual - {selected_cat} > {sub}",
                    xaxis_title="Month", yaxis_title="Sales"
                )
                st.plotly_chart(fig_fore, use_container_width=True)
                last_sale = df_sub["Monthly_Sales"].iloc[-1]
                next_fore = forecast_values[0]
                delta_pct = (next_fore - last_sale) / last_sale * 100 if last_sale else None
                col1, col2, col3 = st.columns(3)
                col1.metric("Last Month Sales", f"${last_sale:,.0f}")
                col2.metric(f"Next {forecast_months}mo Forecast", f"${next_fore:,.0f}",
                            f"{delta_pct:.1f}%", delta_color="inverse")
                col3.metric("Model RMSE", f"{rmse:.2f}")
                csv = df_forecast.to_csv(index=False)
                st.download_button(
                    label="Download Forecast (CSV)",
                    data=csv,
                    file_name=f"{selected_cat}_{sub}_forecast.csv",
                    mime="text/csv"
                )