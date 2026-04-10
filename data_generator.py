import pandas as pd
import numpy as np
from datetime import timedelta

def generate_sales_data(filepath="sales_data.csv", num_days=1095, num_stores=5):
    """
    Generate synthetic sales data for num_days (default 3 years) across num_stores.
    """
    np.random.seed(42)
    end_date = pd.to_datetime("today").normalize()
    start_date = end_date - timedelta(days=num_days)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    categories = ["Electronics", "Clothing", "Home & Kitchen", "Groceries", "Toys"]
    
    data = []
    
    for date in dates:
        # Determine if it's a holiday (roughly 3% chance)
        is_holiday = np.random.choice([0, 1], p=[0.96, 0.04])
        
        # Seasonality factors: More sales in winter/summer
        month = date.month
        season_multiplier = 1.0
        if month in [11, 12]:
            season_multiplier = 1.3 # Holiday peak
        elif month in [6, 7]:
            season_multiplier = 1.1 # Summer
            
        # Weekend factor
        weekend_multiplier = 1.2 if date.weekday() >= 5 else 1.0

        for store_id in range(1, num_stores + 1):
            # Store-specific footfall baseline
            base_footfall = np.random.randint(200, 1000)
            
            for category in categories:
                # Promotion chance (15% chance)
                has_promotion = np.random.choice([0, 1], p=[0.85, 0.15])
                
                # Promotion multiplier
                promo_multiplier = 1.5 if has_promotion else 1.0
                holiday_multiplier = 1.4 if is_holiday else 1.0
                
                # Base sales for category
                if category == "Electronics":
                    base_sales = np.random.normal(5000, 1000)
                elif category == "Groceries":
                    base_sales = np.random.normal(8000, 500)
                else:
                    base_sales = np.random.normal(3000, 800)
                
                # Calculate final sales
                sales_amount = base_sales * season_multiplier * weekend_multiplier * promo_multiplier * holiday_multiplier
                noise = np.random.normal(0, max(200, sales_amount * 0.05))
                sales_amount = max(50, sales_amount + noise) # Ensure positive
                
                # Customer footfall
                footfall = int(base_footfall * weekend_multiplier * (1.2 if has_promotion else 1.0) * holiday_multiplier)
                
                # Introduce some missing values (e.g., 2% of footfall data missing)
                if np.random.uniform() < 0.02:
                    footfall_val = np.nan
                else:
                    footfall_val = footfall

                data.append({
                    "Date": date,
                    "Store_ID": store_id,
                    "Product_Category": category,
                    "Sales_Amount": round(sales_amount, 2),
                    "Is_Holiday": is_holiday,
                    "Has_Promotion": has_promotion,
                    "Customer_Footfall": footfall_val
                })

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Introduce random duplicate rows (around 1%)
    duplicates = df.sample(frac=0.01, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Shuffle slightly so duplicates aren't directly at the end
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Generated synthetic data and saved to {filepath} ({len(df)} rows)")

if __name__ == "__main__":
    generate_sales_data()
