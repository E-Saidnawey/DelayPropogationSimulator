import pandas as pd
from datetime import datetime, timedelta

def load_and_prepare_data(assets_path, telemetry_stream):
    """
    Load data and ensure correct data types.
    """
    
    # Load assets
    df_assets = pd.read_csv(assets_path)
    
    # Convert numeric columns to float
    numeric_cols = ['avg_flights_per_day', 'avg_hours_per_day', 'total_flight_time']
    for col in numeric_cols:
        if col in df_assets.columns:
            df_assets[col] = pd.to_numeric(df_assets[col], errors='coerce').fillna(0.0)
    
    # Load events
    df_events = pd.read_csv(telemetry_stream)
    
    # Convert datetime columns
    date_format = '%m/%d/%Y %I:%M:%S %p'
    datetime_cols = ['timestamp', 'flight_date']
    for col in datetime_cols:
        if col in df_events.columns:
            df_events[col] = pd.to_datetime(df_events[col], format=)
    
    return df_assets, df_events


def initialize_asset_states(flight_assets_df, flight_events_df):
    """
    Initialize asset state tracking for simulation.
    
    Parameters:
    -----------
    flight_assets_df : DataFrame with columns [TAIL_NUM, avg_flights_per_day, 
                       avg_hours_per_day, total_flight_time]
    flight_events_df : DataFrame with flight events to find first scheduled time
    
    Returns:
    --------
    dict : asset_state dictionary ready for simulation
    """
    
    asset_state = {}
    
    # For each aircraft in the fleet
    for _, row in flight_assets_df.iterrows():
        tail_num = row['TAIL_NUM']
        
        # Find the first scheduled event for this aircraft
        asset_events = flight_events_df[flight_events_df['asset_id'] == tail_num]
        
        if len(asset_events) > 0:
            # Get the earliest scheduled time for this asset
            first_scheduled = asset_events['scheduled_time'].min()
        else:
            # Default to earliest time in entire dataset if no events found
            first_scheduled = flight_events_df['scheduled_time'].min()
        
        # Initialize state tracking
        asset_state[tail_num] = {
            "available_at": first_scheduled,  # When aircraft becomes available
            "cumulative_delay": 0,             # Total delay accumulated (minutes)
            "utilization": 0,                  # Total flight time used (hours)
            
            # Additional metrics for Step 4 simulation
            "current_flight": None,            # Track current flight assignment
            "delay_events": [],                # History of delay events
            "baseline_utilization": row['avg_hours_per_day'],  # Expected daily hours
            "total_flight_time": row['total_flight_time']      # Historical total
        }
    
    return asset_state


def get_asset_availability_window(asset_state, asset_id, scheduled_time, 
                                   turnaround_time=timedelta(minutes=30)):
    """
    Helper function for Step 4: Check if asset is available at scheduled time.
    
    Parameters:
    -----------
    asset_state : dict
    asset_id : str (TAIL_NUM)
    scheduled_time : datetime
    turnaround_time : timedelta, minimum ground time between flights
    
    Returns:
    --------
    tuple : (is_available: bool, delay_needed: timedelta)
    """
    
    available_at = asset_state[asset_id]["available_at"]
    required_ready = scheduled_time - turnaround_time
    
    if available_at <= required_ready:
        return True, timedelta(0)
    else:
        delay_needed = available_at - scheduled_time
        return False, delay_needed

def initialize_simulation(flight_assets_df, flight_events_df):
    """
    Efficient initialization without building full dependency graph.
    
    """
    
    # 1. Initialize asset states (this loop is fine - only dozens/hundreds of aircraft)
    asset_state = initialize_asset_states(flight_assets_df, flight_events_df)
    
    # 2. Create sorted event queue
    event_queue = flight_events_df.sort_values('scheduled_time').copy()
    event_queue['processed'] = False
    event_queue['actual_delay'] = 0
    event_queue['propagated_delay'] = 0  # Track delays from upstream events
    
    
    return asset_state, event_queue

# Preparation for Simulation 
def prepare_for_simulation(flight_assets_table, flight_events_table):
    """
    Complete Initialization Setup.
    """
    
    print("=== Initializing Asset States ===")
    
    # Initialize
    asset_state, event_queue = initialize_simulation(
        flight_assets_table, 
        flight_events_table
    )
    
    # Validation
    print(f"✓ Initialized {len(asset_state)} aircraft")
    print(f"✓ Created event queue with {len(event_queue)} events")
    
    # Summary statistics
    print("\nAsset State Summary:")
    for tail_num, state in list(asset_state.items())[:3]:  # Show first 3
        print(f"  {tail_num}:")
        print(f"    Available at: {state['available_at']}")
        print(f"    Baseline utilization: {state['baseline_utilization']:.2f} hrs/day")
    
    return asset_state, event_queue




if __name__ == '__main__':
    flights_assets_table = 'data/processed/flight_assets_table.csv'
    flights_events_table = 'data/processed/flight_events_table.csv'

    # Usage
    df_assets, df_events = load_and_prepare_data(
        flights_assets_table,
        flights_events_table
    )

    asset_state, event_queue = prepare_for_simulation(
        df_assets, df_events
    )