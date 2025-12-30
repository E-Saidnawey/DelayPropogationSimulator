import pandas as pd


def generate_dependencies_table(events_table):
    events_table['scheduled_time'] = pd.to_datetime(events_table['scheduled_time'])
    sorted_events = events_table.sort_values(['asset_id', 'scheduled_time'])
    
    # Get the previous event info
    sorted_events['upstream_event_id'] = sorted_events.groupby('asset_id')['event_id'].shift(1)
    sorted_events['upstream_event_type'] = sorted_events.groupby('asset_id')['event_type'].shift(1)
    
    # 1. Capture ALL sequential dependencies for the same asset
    dependencies = sorted_events.dropna(subset=['upstream_event_id']).copy()
    
    # 2. Categorize them so the simulator knows how to handle the time
    def categorize(row):
        if row['upstream_event_type'] == 'DEPARTURE' and row['event_type'] == 'ARRIVAL':
            return 'FLIGHT_LEG'
        if row['upstream_event_type'] == 'ARRIVAL' and row['event_type'] == 'DEPARTURE':
            return 'AIRCRAFT_TURNAROUND'
        return 'OTHER'

    dependencies['dependency_type'] = dependencies.apply(categorize, axis=1)
    
    # 3. Cleanup and format
    dependencies = dependencies.rename(columns={'event_id': 'downstream_event_id'})
    dependencies['dependency_id'] = [f"DEP_{str(i+1).zfill(8)}" for i in range(len(dependencies))]
    
    return dependencies[[
        'dependency_id', 'upstream_event_id', 'downstream_event_id', 
        'asset_id', 'dependency_type'
    ]]

if __name__ == '__main__':
    # File paths - UPDATE THESE
    input_file = "data/processed/flight_events_table.csv"
    output_file = "data/processed/flights_dependency_table.csv"
    
    # Load flight data
    print(f"Loading flight data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows")
    # Clean and convert
    dependency_df = generate_dependencies_table(df)
    
    if len(dependency_df) > 0:

        print(f"\nSaving cleaned data to {output_file}...")
        dependency_df.to_csv(output_file, index=False)

        print(f"\n✅ Done! Cleaned data saved to {output_file}")
        
        # Optional: Show a sample
        print("\nSample of cleaned data:")
        print(dependency_df.head(10))
    else:
        print(f"\n❌ No data to save. Check your input files and filters.")