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
    
    return assets_final

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
        
        # Optional: Show a sample
        print("\nSample of cleaned data:")
        print(assets_df.head(10))
    else:
        print(f"\n❌ No data to save. Check your input files and filters.")