import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def load_and_prepare_data(assets_path, telemetry_path, dependencies_path):
    """
    Load data, ensure types, and PRE-CALCULATE flight durations.
    """
    # 1. Load raw data
    df_assets = pd.read_csv(assets_path)
    df_events = pd.read_csv(telemetry_path)
    df_deps = pd.read_csv(dependencies_path)

    df_events=df_events.drop(columns=['CARRIER_DELAY',
                                      'WEATHER_DELAY',
                                      'NAS_DELAY',
                                      'SECURITY_DELAY',
                                      'LATE_AIRCRAFT_DELAY']) # remove columns not needed for simulation. These are used as binary in the ML

    # 2. Type conversions
    numeric_cols = ['avg_flights_per_day', 'avg_hours_per_day', 'total_flight_time', 'total_flights']
    for col in numeric_cols:
        if col in df_assets.columns:
            df_assets[col] = pd.to_numeric(df_assets[col], errors='coerce').fillna(0.0)

    datetime_cols = ['scheduled_time', 'timestamp']
    for col in datetime_cols:
        if col in df_events.columns:
            df_events[col] = pd.to_datetime(df_events[col], utc=True)

    # 3. ENRICHMENT: Calculate Scheduled Flight Durations
    # We merge the events table with itself to find the Arrival time for every Departure
    # This assumes 'event_id' and 'downstream_event_id' are reliable from your dependency table
    print("Enriching telemetry with flight durations...")
    
    # Map upstream (Dep) -> downstream (Arr)
    flight_legs = df_deps[df_deps['dependency_type'] == 'FLIGHT_LEG'][['upstream_event_id', 'downstream_event_id']]
    
    # Join to get the Scheduled Time of the downstream Arrival
    events_sched = df_events[['event_id', 'scheduled_time']].set_index('event_id')
    
    flight_durations = flight_legs.merge(
        events_sched, 
        left_on='downstream_event_id', 
        right_index=True,
        suffixes=('', '_arr')
    ).merge(
        events_sched,
        left_on='upstream_event_id',
        right_index=True,
        suffixes=('_arr', '_dep')
    )
    
    # Calculate duration
    flight_durations['scheduled_duration'] = flight_durations['scheduled_time_arr'] - flight_durations['scheduled_time_dep']
    
    # Map back to the main events dataframe
    duration_map = flight_durations.set_index('upstream_event_id')['scheduled_duration']
    df_events['scheduled_duration'] = df_events['event_id'].map(duration_map)

    # Fill NaT for Arrivals (they don't have a flight duration, they have a turnaround)
    df_events['scheduled_duration'] = df_events['scheduled_duration'].fillna(pd.Timedelta(seconds=0))

    return df_assets, df_events, df_deps


def initialize_asset_states(flight_assets_df, flight_events_df):
    """
    Fastest version using dictionary comprehension.
    """
    print("Initializing asset states...")
    
    # Pre-calculate first scheduled times
    first_times = flight_events_df.groupby('asset_id')['scheduled_time'].min().to_dict()
    global_min_time = flight_events_df['scheduled_time'].min()
    
    assets_dict = flight_assets_df.set_index('TAIL_NUM').to_dict('index')
    
    # Initialize state dict
    asset_state = {
        tail_num: {
            "available_at": first_times.get(tail_num, global_min_time),
            "cumulative_delay": 0.0,
            "current_flight": None,
            "delay_events": [],
            "number_of_flights": 0,
            "last_departure_time": first_times.get(tail_num, global_min_time)
        }
        for tail_num in assets_dict.keys()
    }
    
    print(f"Initialized {len(asset_state)} assets")
    return asset_state