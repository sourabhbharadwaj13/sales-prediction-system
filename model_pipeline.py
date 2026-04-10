import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def load_and_clean_data(filepath="sales_data.csv"):
    df = pd.read_csv(filepath)
    
    # 1. Remove duplicates
    df = df.drop_duplicates()
    
    # 2. Handle missing values
    # Fill missing footfall with median per store
    df['Customer_Footfall'] = df['Customer_Footfall'].fillna(df.groupby('Store_ID')['Customer_Footfall'].transform('median'))
    
    # 3. Convert 'Date' to datetime and extract features
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Year'] = df['Date'].dt.year
    df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Season (1: Winter, 2: Spring, 3: Summer, 4: Fall)
    def get_season(month):
        if month in [12, 1, 2]: return 1
        elif month in [3, 4, 5]: return 2
        elif month in [6, 7, 8]: return 3
        else: return 4
    df['Season'] = df['Month'].apply(get_season)
    
    # Sort by Date and Store and Category to create lag features properly
    df = df.sort_values(by=['Date', 'Store_ID', 'Product_Category']).reset_index(drop=True)
    
    return df

def perform_feature_engineering(df):
    df_fe = df.copy()
    
    # Group by Store and Category to calculate lag and rolling features
    # Note: For strict time series, we need to ensure no data leakage and handle grouped lags
    
    # 1-day/7-day lag of Sales_Amount
    df_fe['Sales_Lag_1'] = df_fe.groupby(['Store_ID', 'Product_Category'])['Sales_Amount'].shift(1)
    df_fe['Sales_Lag_7'] = df_fe.groupby(['Store_ID', 'Product_Category'])['Sales_Amount'].shift(7)
    
    # 7-day rolling average
    df_fe['Sales_Roll_Mean_7'] = df_fe.groupby(['Store_ID', 'Product_Category'])['Sales_Amount'].transform(lambda x: x.shift(1).rolling(window=7).mean())
    
    # Drop rows with NaNs introduced by shift/rolling
    df_fe = df_fe.dropna().reset_index(drop=True)
    
    # Encode Categorical variables (Product_Category)
    df_fe = pd.get_dummies(df_fe, columns=['Product_Category'], drop_first=True)
    
    return df_fe

def train_and_evaluate_models(df_fe):
    features = [col for col in df_fe.columns if col not in ['Date', 'Sales_Amount']]
    X = df_fe[features]
    y = df_fe['Sales_Amount']
    
    # Train-test split (time-based split preferred for time series)
    train_size = int(len(df_fe) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        trained_models[name] = model

    # Select the overall best model by R2
    best_model_name = max(results, key=lambda k: results[k]['R2'])
    best_model = trained_models[best_model_name]
    
    # Feature importance for Best Model (if tree-based)
    feature_importance = None
    if best_model_name in ['Random Forest', 'XGBoost']:
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': best_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
    return results, best_model_name, best_model, feature_importance, features

def generate_future_predictions(model, latest_data, features, days=7):
    """
    Generate naive future predictions using the trained model.
    In a real scenario, future feature values (like promotions, true holiday schedules) 
    must be known or forecasted. Here we use naive continuation.
    """
    # Assuming latest_data contains the last row per store/category
    future_rows = []
    
    current_date = latest_data['Date'].max()
    
    # For simplicity, let's just predict for one store & one category based on its last known state
    # We will simulate a forward stepping loop
    
    for i in range(1, days + 1):
        next_date = current_date + pd.Timedelta(days=i)
        
        # Prepare a synthetic row for prediction
        # Copy last known state and update time features
        row = latest_data.copy().iloc[-1:] 
        row['Month'] = next_date.month
        row['Day'] = next_date.day
        row['DayOfWeek'] = next_date.dayofweek
        row['Year'] = next_date.year
        row['Is_Weekend'] = 1 if next_date.dayofweek >= 5 else 0
        
        # Predict
        pred_sales = model.predict(row[features])[0]
        future_rows.append({'Date': next_date, 'Predicted_Sales': pred_sales})
        
    return pd.DataFrame(future_rows)
