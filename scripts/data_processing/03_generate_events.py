import pandas as pd
from datetime import datetime

def generate_events_table(cleaned_data):
    # --- 1. PRE-FILTERING (Sanity Checks) ---
    print(f"Initial rows: {len(cleaned_data)}")
    
    # Convert to datetime for comparison
    cleaned_data['scheduled_dep_utc'] = pd.to_datetime(cleaned_data['scheduled_dep_utc'])
    cleaned_data['actual_dep_utc'] = pd.to_datetime(cleaned_data['actual_dep_utc'])
    
    # A. Remove "Impossible Delays" (e.g., actual time is > 12 hours from scheduled)
    # This removes the "20 hours ahead" noise you mentioned.
    delay_diff_hours = (cleaned_data['actual_dep_utc'] - cleaned_data['scheduled_dep_utc']).dt.total_seconds() / 3600
    cleaned_data = cleaned_data[delay_diff_hours.abs() < 12].copy()
    print(f"After removing >12h outliers: {len(cleaned_data)}")

    # B. Remove Scheduled Overlaps (Same plane, same scheduled time)
    # Sanity check, since we removed them in clean_data.py.
    cleaned_data = cleaned_data.drop_duplicates(subset=['TAIL_NUM', 'scheduled_dep_utc'], keep='first')
    print(f"After removing scheduled overlaps: {len(cleaned_data)}")

    # --- 2. GENERATE EVENTS ---
    
    # Create departure events
    departures = pd.DataFrame({
        'event_id': ['EVT_' + str(i).zfill(8) for i in range(1, len(cleaned_data)*2, 2)],
        'asset_id': cleaned_data['TAIL_NUM'],
        'scheduled_time': cleaned_data['scheduled_dep_utc'],
        'actual_time': cleaned_data['actual_dep_utc'],
        'event_type': 'DEPARTURE',
        'flight_date': cleaned_data['FL_DATE'],
        'carrier': cleaned_data['OP_UNIQUE_CARRIER'],
        'flight_number': cleaned_data['MKT_CARRIER_FL_NUM'],
        'origin': cleaned_data['ORIGIN_AIRPORT_ID'],
        'destination': cleaned_data['DEST_AIRPORT_ID'],
    })

    # Create arrival events
    arrivals = pd.DataFrame({
        'event_id': ['EVT_' + str(i).zfill(8) for i in range(2, len(cleaned_data)*2+1, 2)],
        'asset_id': cleaned_data['TAIL_NUM'],
        'scheduled_time': pd.to_datetime(cleaned_data['scheduled_arr_utc']),
        'actual_time': pd.to_datetime(cleaned_data['actual_arr_utc']),
        'event_type': 'ARRIVAL',
        'flight_date': cleaned_data['FL_DATE'],
        'carrier': cleaned_data['OP_UNIQUE_CARRIER'],
        'flight_number': cleaned_data['MKT_CARRIER_FL_NUM'],
        'origin': cleaned_data['ORIGIN_AIRPORT_ID'],
        'destination': cleaned_data['DEST_AIRPORT_ID']
    })
    
    # Concatenate
    events = pd.concat([departures, arrivals], ignore_index=True)
    
    # Sort by asset and scheduled time to ensure chronological sequence
    events = events.sort_values(['asset_id', 'scheduled_time']).reset_index(drop=True)
    
    return events

if __name__ == '__main__':
    # File paths - UPDATE THESE
    input_file = "data/processed/flights_cleaned_us_only.csv"
    output_file = "data/processed/flight_events_table.csv"
    
    # Load flight data
    print(f"Loading flight data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows")
    # Clean and convert

    events_df = generate_events_table(df)
    
    if len(events_df) > 0:

        print(f"\nSaving cleaned data to {output_file}...")
        events_df.to_csv(output_file, index=False)

        print(f"\n✅ Done! Cleaned data saved to {output_file}")
        
        # Optional: Show a sample
        print("\nSample of cleaned data:")
        print(events_df[['event_id',
                         'asset_id',
                         'scheduled_time',
                         'actual_time',
                         'event_type',
                         'flight_date',
                         'carrier',
                         'flight_number',
                         'origin',
                         'destination']].head(10)
                        )
    else:
        print(f"\n❌ No data to save. Check your input files and filters.")