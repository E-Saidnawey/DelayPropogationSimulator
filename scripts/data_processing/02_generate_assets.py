# # 1. Load your flight data
# flights_df = pd.read_csv('your_flights.csv')

# # 2. Load the OpenSky data 
# # We'll only load the useful columns to save memory
# useful_columns = ['registration', 'manufacturername', 'model', 'typecode', 'operator']
# opensky_df = pd.read_csv('aircraftDatabase.csv', usecols=useful_columns)

# # 3. Clean the Tail Numbers (Crucial Step!)
# # Tail numbers can sometimes have hidden spaces or case differences
# flights_df['TAIL_NUM'] = flights_df['TAIL_NUM'].astype(str).str.strip().str.upper()
# opensky_df['registration'] = opensky_df['registration'].astype(str).str.strip().str.upper()

# # 4. Perform the Merge
# # This says: "Take flights_df, and add columns from opensky_df where TAIL_NUM == registration"
# merged_df = pd.merge(
#     flights_df, 
#     opensky_df, 
#     left_on='TAIL_NUM', 
#     right_on='registration', 
#     how='left'
# )

# # 5. Clean up (Optional)
# # Since we have TAIL_NUM, we don't need the 'registration' column anymore
# merged_df.drop(columns=['registration'], inplace=True)

# # 6. Save your new enriched CSV
# merged_df.to_csv('enriched_flight_data.csv', index=False)


import pandas as pd
import copy
from datetime import datetime, timedelta

def generate_assets_table(cleaned_data):
    assert isinstance(cleaned_data, pd.DataFrame), "Input must be pandas DataFrame"
    
    date_format = '%m/%d/%Y %I:%M:%S %p'
    utc_format = '%Y-%m-%d %H:%M:%S%z'

    assets = cleaned_data.copy()
    assets['FL_DATE'] = pd.to_datetime(assets['FL_DATE'], format=date_format)
    assets['actual_arr_utc'] = pd.to_datetime(assets['actual_arr_utc'], format=utc_format)
    assets['actual_dep_utc'] = pd.to_datetime(assets['actual_dep_utc'], format=utc_format)
    assets['flight_time'] = assets['actual_arr_utc'] - assets['actual_dep_utc']
    
    grouped_assets = assets.groupby(['TAIL_NUM'])
    origin_airport = grouped_assets['ORIGIN_AIRPORT_ID'].agg(lambda x: x.mode().iloc[0]).reset_index()
    date_range = grouped_assets['FL_DATE'].max() - grouped_assets['FL_DATE'].min()
    total_flights = grouped_assets.size()
    flight_time   = grouped_assets['flight_time'].sum()
    avg_flights_per_day = total_flights / date_range.dt.days
    avg_hours_per_day = flight_time / date_range.dt.days
    carrier = grouped_assets['OP_UNIQUE_CARRIER'].agg(lambda x: x.mode().iloc[0]).reset_index()

    assets_final = pd.concat([
        origin_airport.set_index('TAIL_NUM'),
        carrier.set_index('TAIL_NUM'),
        total_flights.rename('total_flights'),
        date_range.rename('date_range'),
        flight_time.rename('total_flight_time'),
        avg_flights_per_day.rename('avg_flights_per_day'),
        avg_hours_per_day.rename('avg_hours_per_day')
    ], axis=1).reset_index()
    
    aircraft_mdl_df = pd.read_csv('data/raw/MASTER.txt')
    aircraft_seat_df = pd.read_csv('data/raw/ACFTREF.txt')

    aircraft_mdl_df['N-NUMBER'] = 'N' + aircraft_mdl_df['N-NUMBER']

    assets_final['TAIL_NUM'] = assets_final['TAIL_NUM'].astype(str).str.strip().str.upper()
    aircraft_mdl_df['N-NUMBER'] = aircraft_mdl_df['N-NUMBER'].astype(str).str.strip().str.upper()

    merged_df = pd.merge(
        assets_final, 
        aircraft_mdl_df[['N-NUMBER', 'MFR MDL CODE']], 
        left_on='TAIL_NUM', 
        right_on='N-NUMBER', 
        how='left'
    )

    merged_df = merged_df.dropna(subset=['N-NUMBER'], inplace=False)

    final_df = pd.merge(
            merged_df,
            aircraft_seat_df[['CODE', 'NO-SEATS']],
            left_on='MFR MDL CODE', 
            right_on='CODE',
            how='left'
        )

    return final_df

if __name__ == '__main__':
    # File paths - UPDATE THESE
    input_file = "data/processed/flights_cleaned_us_only.csv"
    output_file = "data/processed/flight_assets_table.csv"
    
    # Load flight data
    print(f"Loading flight data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows")
    # Clean and convert
    assets_df = generate_assets_table(df)
    
    if len(assets_df) > 0:

        print(f"\nSaving cleaned data to {output_file}...")
        assets_df.to_csv(output_file, index=False)

        print(f"\n✅ Done! Cleaned data saved to {output_file}")
        
        print("\nUpdating Flights Cleaned Data")
        processed_tailnum = set(assets_df['TAIL_NUM'])
        
        df_cleaned = pd.read_csv('data/processed/flights_cleaned_us_only.csv')
        tail_mask = df_cleaned['TAIL_NUM'].isin(processed_tailnum)

        df_cleaned = df_cleaned[tail_mask].copy()
        df_cleaned.to_csv('data/processed/flights_cleaned_us_only.csv')

        print("\nSample of cleaned data:")
        print(assets_df.head(10))


    else:
        print(f"\n❌ No data to save. Check your input files and filters.")