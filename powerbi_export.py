import pandas as pd
import numpy as np

def generate_powerbi_dataset(df_clean, df_fe, best_model, features, days_to_predict=30, output_file="powerbi_dataset.csv"):
    """
    Generate a consolidated dataset for Power BI, combining historical actuals
    with future forecasts for all stores and all categories.
    """
    print("Generating Power BI base dataset... This might take a moment.")
    
    # 1. Historical Data Portion (Actuals)
    historical_df = df_clean.copy()
    historical_df = historical_df[['Date', 'Store_ID', 'Product_Category', 'Sales_Amount']]
    historical_df['Status'] = 'Actual'
    
    # 2. Forecasting Portion (Predictions)
    stores = df_clean['Store_ID'].unique()
    categories = df_clean['Product_Category'].unique()
    
    future_rows = []
    
    for store_id in stores:
        for category in categories:
            # Locate last known feature state for this store/category combo
            store_cat_fe = df_fe[(df_fe['Store_ID'] == store_id)]
            if f'Product_Category_{category}' in df_fe.columns:
                store_cat_fe = store_cat_fe[store_cat_fe[f'Product_Category_{category}'] == 1]
                
            if store_cat_fe.empty:
                continue
                
            last_row = store_cat_fe.iloc[-1:].copy()
            current_date = last_row['Date'].max()
            
            for i in range(1, days_to_predict + 1):
                next_date = current_date + pd.Timedelta(days=i)
                
                pred_row = last_row.copy()
                pred_row['Date'] = next_date
                pred_row['Month'] = next_date.month
                pred_row['Day'] = next_date.day
                pred_row['DayOfWeek'] = next_date.dayofweek
                pred_row['Year'] = next_date.year
                pred_row['Is_Weekend'] = 1 if next_date.dayofweek >= 5 else 0
                
                pred_val = best_model.predict(pred_row[features])[0]
                
                future_rows.append({
                    'Date': next_date,
                    'Store_ID': store_id,
                    'Product_Category': category,
                    'Sales_Amount': pred_val,
                    'Status': 'Forecast'
                })
    
    forecast_df = pd.DataFrame(future_rows)
    
    # Concatenate and save
    final_df = pd.concat([historical_df, forecast_df], ignore_index=True)
    
    # Ensure Date format is clean for PowerBI
    final_df['Date'] = pd.to_datetime(final_df['Date']).dt.date
    
    final_df.to_csv(output_file, index=False)
    print(f"Dataset successfully exported to {output_file}")
    return output_file
