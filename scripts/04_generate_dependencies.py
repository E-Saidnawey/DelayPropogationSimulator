import pandas as pd


def generate_dependencies_table(events_table):
    dependencies = []
    
    # Sort events by asset and scheduled time
    sorted_events = events_table.sort_values(['asset_id', 'scheduled_time'])
    
    # Group by asset
    for asset_id, asset_events in sorted_events.groupby('asset_id'):
        # Separate arrivals and departures
        arrivals = asset_events[asset_events['event_type'] == 'ARRIVAL'].copy()
        departures = asset_events[asset_events['event_type'] == 'DEPARTURE'].copy()
        
        if arrivals.empty or departures.empty:
            continue
        

        arrivals = arrivals.rename(columns={'scheduled_time': 'scheduled_time_arrival'})
        departures = departures.rename(columns={'scheduled_time': 'scheduled_time_departure'})
        
        # Use merge_asof to match each arrival with the next departure
        matched = pd.merge_asof(
            arrivals,
            departures,
            left_on='scheduled_time_arrival',
            right_on='scheduled_time_departure',
            by='asset_id',
            direction='forward',
            suffixes=('_arrival', '_departure')
        )

        matched['scheduled_time_departure'] = pd.to_datetime(matched['scheduled_time_departure'])
        matched['scheduled_time_arrival'] = pd.to_datetime(matched['scheduled_time_arrival'])
        

        # Filter out matches where departure is too far in the future (>24 hours)
        matched['time_diff_hours'] = (
            matched['scheduled_time_departure'] - 
            matched['scheduled_time_arrival']
        ).dt.total_seconds() / 3600
        
        valid_matches = matched[matched['time_diff_hours'] < 24].copy()
        
        # Calculate scheduled separation in minutes
        valid_matches['scheduled_separation_minutes'] = (
            valid_matches['scheduled_time_departure'] - 
            valid_matches['scheduled_time_arrival']
        ).dt.total_seconds() / 60
        
        # Add other fields
        valid_matches['min_separation_minutes'] = 45
        valid_matches['dependency_type'] = 'AIRCRAFT_TURNAROUND'
        
        dependencies.append(valid_matches)
    
    # Combine all dependencies
    if dependencies:
        all_deps = pd.concat(dependencies, ignore_index=True)
        
        # Create dependency IDs
        all_deps['dependency_id'] = [
            f"DEP_{str(i+1).zfill(8)}" 
            for i in range(len(all_deps))
        ]
        
        # Select and rename columns as needed
        result = all_deps[[
            'dependency_id',
            'event_id_arrival',  # rename to upstream_event_id
            'event_id_departure',  # rename to downstream_event_id
            # ... other columns
        ]].rename(columns={
            'event_id_arrival': 'upstream_event_id',
            'event_id_departure': 'downstream_event_id',
            # ...
        })
        
        return result
    else:
        return pd.DataFrame()

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