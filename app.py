import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv("synthetic_seasonal_product_sales.csv")
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    data.set_index('date', inplace=True)
    return data

data = load_data()

# Feature Engineering
def feature_engineering(data):
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['day_of_week'] = data.index.dayofweek
    data['lag_1'] = data['units_sold'].shift(1)
    data['lag_7'] = data['units_sold'].shift(7)
    data['rolling_mean_7'] = data['units_sold'].rolling(window=7).mean()
    data['rolling_mean_30'] = data['units_sold'].rolling(window=30).mean()
    data.dropna(inplace=True)
    return data

data = feature_engineering(data)

# Train Model
def train_model(data):
    X = data.drop(columns=['units_sold'])
    y = data['units_sold']
    X = pd.get_dummies(X, columns=['store_location', 'season'], drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(random_state=42)
    
    param_dist = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                   n_iter=50, cv=5, verbose=2, n_jobs=-1, random_state=42)
    
    rf_random.fit(X_train, y_train)
    y_pred = rf_random.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    return rf_random, X_train.columns, rmse, y_test, X_test

rf_random, features, rmse, y_test, X_test = train_model(data)

# Function for recommending price
def recommend_price(product_id, store_location, competitor_price, promotion, stock_level, season, rf_model, original_features):
    input_data = pd.DataFrame({
        'product_id': [product_id],
        'competitor_price': [competitor_price],
        'promotion': [promotion],
        'stock_level': [stock_level],
        'store_location': [store_location],
        'season': [season],
        'month': [1],
        'day': [1],
        'day_of_week': [1],
        'lag_1': [0],
        'lag_7': [0],
        'rolling_mean_7': [0],
        'rolling_mean_30': [0],
    })
    
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=original_features, fill_value=0)
    
    predicted_units_sold = rf_model.predict(input_data)[0]
    
    initial_price = competitor_price + np.random.uniform(-2, 2)
    recommended_price = initial_price
    
    if predicted_units_sold > 50:
        recommended_price *= 1.1
    elif predicted_units_sold < 20:
        recommended_price *= 0.9
    
    if competitor_price < recommended_price:
        recommended_price = competitor_price * 0.98
    
    if promotion == 1:
        recommended_price *= 0.85
    
    estimated_revenue = predicted_units_sold * recommended_price
    
    return {
        'predicted_units_sold': predicted_units_sold,
        'recommended_price': round(recommended_price, 2),
        'estimated_revenue': round(estimated_revenue, 2)
    }

# Streamlit App Interface
st.title("Product Pricing Strategy Recommendation")

st.write(f"Model RMSE: {rmse:.2f}")

# Raw Data Display
st.subheader("Raw Data")
st.write(data.head())

# Units Sold Over Time
st.subheader("Units Sold Over Time")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data.index, data['units_sold'], label='Units Sold')
ax.set_xlabel('Date')
ax.set_ylabel('Units Sold')
ax.set_title('Units Sold Over Time')
st.pyplot(fig)

# Feature Distributions
st.subheader("Feature Distributions")
features_to_plot = ['competitor_price', 'promotion', 'stock_level']
fig, axs = plt.subplots(1, len(features_to_plot), figsize=(15, 5))

for i, feature in enumerate(features_to_plot):
    sns.histplot(data[feature], ax=axs[i])
    axs[i].set_title(f'Distribution of {feature}')

st.pyplot(fig)

# Feature Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
corr_matrix = data.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Model Predictions vs Actuals
st.subheader("Model Predictions vs Actuals")
y_pred = rf_random.predict(X_test)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Units Sold')
ax.set_ylabel('Predicted Units Sold')
ax.set_title('Model Predictions vs Actuals')
st.pyplot(fig)

# User Inputs for Price Recommendation
product_id = st.number_input("Product ID", min_value=1001, max_value=1050, step=1)
store_location = st.selectbox("Store Location", ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"])
competitor_price = st.number_input("Competitor Price", min_value=0.0, step=0.01)
promotion = st.selectbox("Promotion", [0, 1])
stock_level = st.number_input("Stock Level", min_value=0, step=1)
season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])

if st.button("Recommend Price"):
    result = recommend_price(
        product_id=product_id,
        store_location=store_location,
        competitor_price=competitor_price,
        promotion=promotion,
        stock_level=stock_level,
        season=season,
        rf_model=rf_random,
        original_features=features
    )
    
    st.write(f"Predicted Units Sold: {result['predicted_units_sold']}")
    st.write(f"Recommended Price: ${result['recommended_price']}")
    st.write(f"Estimated Revenue: ${result['estimated_revenue']}")
