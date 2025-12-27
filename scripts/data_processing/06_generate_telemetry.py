import pandas as pd
from datetime import datetime

def create_telemetry_stream(flight_events_df):
    """
    Convert flight events table into a time-ordered telemetry stream.
    
    Parameters:
    -----------
    flight_events_df : pandas.DataFrame
        Input table with columns: event_id, asset_id, scheduled_time, 
        actual_time, event_type, flight_date, carrier, flight_number, 
        origin, destination
    
    Returns:
    --------
    list of dict : Telemetry stream sorted by timestamp
    """
    
    telemetry_emissions = []
    
    # Iterate through each event
    for idx, row in flight_events_df.iterrows():
        
        # Create scheduled emission
        scheduled_emission = {
            'timestamp': row['scheduled_time'],
            'event_id': row['event_id'],
            'asset_id': row['asset_id'],
            'state': 'scheduled',
            'event_type': row['event_type'],
            # Optional: include reference data
            'flight_date': row['flight_date'],
            'carrier': row['carrier'],
            'flight_number': row['flight_number'],
            'origin': row['origin'],
            'destination': row['destination']
        }
        
        # Create actual emission
        actual_emission = {
            'timestamp': row['actual_time'],
            'event_id': row['event_id'],
            'asset_id': row['asset_id'],
            'state': 'actual',
            'event_type': row['event_type'],
            # Optional: include reference data
            'flight_date': row['flight_date'],
            'carrier': row['carrier'],
            'flight_number': row['flight_number'],
            'origin': row['origin'],
            'destination': row['destination']
        }
        
        telemetry_emissions.append(scheduled_emission)
        telemetry_emissions.append(actual_emission)
    
    # Sort globally by timestamp
    telemetry_stream = sorted(telemetry_emissions, key=lambda x: x['timestamp'])
    
    return telemetry_stream


if __name__ == "__main__":
    # Load from CSV
    input_file = "data/processed/flight_events_table.csv"
    output_file = "data/processed/telemetry_stream.csv"
    flight_events_df = pd.read_csv(input_file)
        
    # Ensure timestamps are in datetime format
    flight_events_df['scheduled_time'] = pd.to_datetime(flight_events_df['scheduled_time'])
    flight_events_df['actual_time'] = pd.to_datetime(flight_events_df['actual_time'])
    
    # Create telemetry stream
    telemetry_stream = create_telemetry_stream(flight_events_df)
    
    # Output options:
    
    # Save as CSV
    telemetry_df = pd.DataFrame(telemetry_stream)
    telemetry_df.to_csv(output_file, index=False)