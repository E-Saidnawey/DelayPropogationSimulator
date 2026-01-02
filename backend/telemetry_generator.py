"""
Telemetry Stream Generator
Converts historical flight data into realistic real-time telemetry updates
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
from pathlib import Path


def load_and_prepare_flights(flights_csv: str) -> pd.DataFrame:
    """
    Load flight data from CSV and prepare it for telemetry generation
    
    Args:
        flights_csv: Path to cleaned flights data
        
    Returns:
        Prepared DataFrame with datetime columns and filled delays
    """
    flights = pd.read_csv(flights_csv)
    
    # Convert to datetime
    flights['scheduled_dep_utc'] = pd.to_datetime(flights['scheduled_dep_utc'])
    flights['actual_dep_utc'] = pd.to_datetime(flights['actual_dep_utc'])
    flights['scheduled_arr_utc'] = pd.to_datetime(flights['scheduled_arr_utc'])
    flights['actual_arr_utc'] = pd.to_datetime(flights['actual_arr_utc'])
    
    # Fill NaN delays with 0
    delay_cols = ['dep_delay_minutes', 'arr_delay_minutes', 
                 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 
                 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
    for col in delay_cols:
        if col in flights.columns:
            flights[col] = flights[col].fillna(0)
    
    return flights


def generate_flight_telemetry_updates(flight_row: pd.Series) -> List[Dict[str, Any]]:
    """
    Generate incremental telemetry updates for a single flight
    Shows delay growing from scheduled time to actual departure AND arrival
    
    Args:
        flight_row: Single row from flights DataFrame
        
    Returns:
        List of telemetry update dictionaries
    """
    updates = []
    
    scheduled_dep = flight_row['scheduled_dep_utc']
    actual_dep = flight_row['actual_dep_utc']
    scheduled_arr = flight_row['scheduled_arr_utc']
    actual_arr = flight_row['actual_arr_utc']
    
    final_dep_delay = flight_row.get('dep_delay_minutes', 0)
    final_arr_delay = flight_row.get('arr_delay_minutes', 0)
    
    # Get delay reasons for departure (only include if > 20 min)
    dep_delay_reasons = {}
    if final_dep_delay > 20:
        if flight_row.get('CARRIER_DELAY', 0) > 0:
            dep_delay_reasons['carrier_delay'] = float(flight_row['CARRIER_DELAY'])
        if flight_row.get('WEATHER_DELAY', 0) > 0:
            dep_delay_reasons['weather_delay'] = float(flight_row['WEATHER_DELAY'])
        if flight_row.get('NAS_DELAY', 0) > 0:
            dep_delay_reasons['nas_delay'] = float(flight_row['NAS_DELAY'])
        if flight_row.get('SECURITY_DELAY', 0) > 0:
            dep_delay_reasons['security_delay'] = float(flight_row['SECURITY_DELAY'])
        if flight_row.get('LATE_AIRCRAFT_DELAY', 0) > 0:
            dep_delay_reasons['late_aircraft_delay'] = float(flight_row['LATE_AIRCRAFT_DELAY'])
    
    # Get delay reasons for arrival (only include if > 20 min)
    arr_delay_reasons = {}
    if final_arr_delay > 20:
        if flight_row.get('CARRIER_DELAY', 0) > 0:
            arr_delay_reasons['carrier_delay'] = float(flight_row['CARRIER_DELAY'])
        if flight_row.get('WEATHER_DELAY', 0) > 0:
            arr_delay_reasons['weather_delay'] = float(flight_row['WEATHER_DELAY'])
        if flight_row.get('NAS_DELAY', 0) > 0:
            arr_delay_reasons['nas_delay'] = float(flight_row['NAS_DELAY'])
        if flight_row.get('SECURITY_DELAY', 0) > 0:
            arr_delay_reasons['security_delay'] = float(flight_row['SECURITY_DELAY'])
        if flight_row.get('LATE_AIRCRAFT_DELAY', 0) > 0:
            arr_delay_reasons['late_aircraft_delay'] = float(flight_row['LATE_AIRCRAFT_DELAY'])
    
    # === DEPARTURE EVENTS ===
    
    # Generate incremental departure delay updates
    if final_dep_delay > 0:
        # Number of updates depends on delay magnitude
        if final_dep_delay < 15:
            num_updates = 1
        elif final_dep_delay < 30:
            num_updates = 2
        elif final_dep_delay < 60:
            num_updates = 3
        else:
            num_updates = 4
        
        # Generate incremental delays
        delay_increments = np.linspace(0, final_dep_delay, num_updates + 1)[1:]
        
        for i, delay in enumerate(delay_increments):
            # Time of this update
            update_time = scheduled_dep + timedelta(minutes=delay * 0.7)
            
            update = {
                'flight_number': str(flight_row.get('MKT_CARRIER_FL_NUM', 'UNKNOWN')),
                'tail_number': str(flight_row.get('TAIL_NUM', 'UNKNOWN')),
                'carrier': str(flight_row.get('MKT_UNIQUE_CARRIER', 'UNKNOWN')),
                'origin': str(flight_row.get('ORIGIN_AIRPORT_ID', 'UNKNOWN')),
                'destination': str(flight_row.get('DEST_AIRPORT_ID', 'UNKNOWN')),
                'scheduled_time': scheduled_dep.isoformat(),
                'current_delay_minutes': float(delay),
                'update_time': update_time.isoformat(),
                'status': 'delayed' if delay > 15 else 'on-time',
                'event_type': 'departure'
            }
            
            # Add delay reasons only on final update if > 20 min
            if i == len(delay_increments) - 1 and delay > 20:
                update['delay_reasons'] = dep_delay_reasons
            
            updates.append(update)
    
    # Final departure event
    departure_update = {
        'flight_number': str(flight_row.get('MKT_CARRIER_FL_NUM', 'UNKNOWN')),
        'tail_number': str(flight_row.get('TAIL_NUM', 'UNKNOWN')),
        'carrier': str(flight_row.get('MKT_UNIQUE_CARRIER', 'UNKNOWN')),
        'origin': str(flight_row.get('ORIGIN_AIRPORT_ID', 'UNKNOWN')),
        'destination': str(flight_row.get('DEST_AIRPORT_ID', 'UNKNOWN')),
        'scheduled_time': scheduled_dep.isoformat(),
        'actual_time': actual_dep.isoformat(),
        'current_delay_minutes': float(final_dep_delay),
        'update_time': actual_dep.isoformat(),
        'status': 'departed',
        'event_type': 'departure'
    }
    
    if final_dep_delay > 20:
        departure_update['delay_reasons'] = dep_delay_reasons
    
    updates.append(departure_update)
    
    # === ARRIVAL EVENTS ===
    
    # Generate incremental arrival delay updates (if different from departure)
    if final_arr_delay > 0:
        # Number of updates depends on delay magnitude
        if final_arr_delay < 15:
            num_updates = 1
        elif final_arr_delay < 30:
            num_updates = 2
        elif final_arr_delay < 60:
            num_updates = 3
        else:
            num_updates = 4
        
        # Generate incremental delays
        delay_increments = np.linspace(0, final_arr_delay, num_updates + 1)[1:]
        
        for i, delay in enumerate(delay_increments):
            # Time of this update (relative to scheduled arrival)
            update_time = scheduled_arr + timedelta(minutes=delay * 0.7)
            
            update = {
                'flight_number': str(flight_row.get('MKT_CARRIER_FL_NUM', 'UNKNOWN')),
                'tail_number': str(flight_row.get('TAIL_NUM', 'UNKNOWN')),
                'carrier': str(flight_row.get('MKT_UNIQUE_CARRIER', 'UNKNOWN')),
                'origin': str(flight_row.get('ORIGIN_AIRPORT_ID', 'UNKNOWN')),
                'destination': str(flight_row.get('DEST_AIRPORT_ID', 'UNKNOWN')),
                'scheduled_time': scheduled_arr.isoformat(),
                'current_delay_minutes': float(delay),
                'update_time': update_time.isoformat(),
                'status': 'delayed' if delay > 15 else 'on-time',
                'event_type': 'arrival'
            }
            
            # Add delay reasons only on final update if > 20 min
            if i == len(delay_increments) - 1 and delay > 20:
                update['delay_reasons'] = arr_delay_reasons
            
            updates.append(update)
    
    # Final arrival event
    arrival_update = {
        'flight_number': str(flight_row.get('MKT_CARRIER_FL_NUM', 'UNKNOWN')),
        'tail_number': str(flight_row.get('TAIL_NUM', 'UNKNOWN')),
        'carrier': str(flight_row.get('MKT_UNIQUE_CARRIER', 'UNKNOWN')),
        'origin': str(flight_row.get('ORIGIN_AIRPORT_ID', 'UNKNOWN')),
        'destination': str(flight_row.get('DEST_AIRPORT_ID', 'UNKNOWN')),
        'scheduled_time': scheduled_arr.isoformat(),
        'actual_time': actual_arr.isoformat(),
        'current_delay_minutes': float(final_arr_delay),
        'update_time': actual_arr.isoformat(),
        'status': 'arrived',
        'event_type': 'arrival'
    }
    
    if final_arr_delay > 20:
        arrival_update['delay_reasons'] = arr_delay_reasons
    
    updates.append(arrival_update)
    if final_arr_delay > 20:
        arrival_update['delay_reasons'] = arr_delay_reasons
    
    updates.append(arrival_update)
    
    return updates


def generate_telemetry_stream(flights_df: pd.DataFrame, start_date: str, hours: int | None = None) -> List[Dict[str, Any]]:
    """
    Generate telemetry updates for all flights in a time range
    
    Args:
        flights_df: Prepared DataFrame with flight data
        start_date: ISO format date string (e.g., "2024-01-01")
        hours: Number of hours of flights to include
    
    Returns:
        List of telemetry updates sorted by update_time
    """
    start = pd.to_datetime(start_date).tz_localize('UTC')

    if hours:
        end = start + timedelta(hours=hours)
    else:
        end = flights_df['scheduled_dep_utc'].max() + timedelta(hours=1)

    # Filter flights in this time range
    mask = (flights_df['scheduled_dep_utc'] >= start) & (flights_df['scheduled_dep_utc'] < end)
    flights_in_range = flights_df[mask].copy()
    
    print(f"Found {len(flights_in_range)} flights between {start} and {end}")
    
    # Generate telemetry for each flight
    all_updates = []
    total_flights = len(flights_in_range)
    
    for idx, (_, flight) in enumerate(flights_in_range.iterrows()):
        if idx > 0 and idx % 1000 == 0:
            print(f"  Processed {idx}/{total_flights} flights ({idx/total_flights*100:.1f}%)")
        
        try:
            updates = generate_flight_telemetry_updates(flight)
            all_updates.extend(updates)
        except Exception as e:
            print(f"Error processing flight {flight.get('FL_NUM', 'UNKNOWN')}: {e}")
            continue
    
    # Sort by update time
    print(f"Sorting {len(all_updates)} telemetry updates by time...")
    all_updates.sort(key=lambda x: x['update_time'])
    
    print(f"✓ Generated {len(all_updates)} telemetry updates")
    
    return all_updates


def save_telemetry_stream(updates: List[Dict[str, Any]], output_path: str):
    """
    Save telemetry stream to JSON file
    
    Args:
        updates: List of telemetry update dictionaries
        output_path: Path to output JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(updates, f, indent=2)
    print(f"Saved telemetry stream to {output_path}")


def main():
    """Generate sample telemetry stream"""
    
    flights_csv = 'data/processed/flights_cleaned_us_only.csv'
    start_date = '2024-1-1'
    
    print(f"Generating telemetry stream from {flights_csv}")
    
    # Load and prepare flight data
    flights_df = load_and_prepare_flights(flights_csv)
    
    # Generate telemetry stream
    updates = generate_telemetry_stream(flights_df, start_date)
    
    # Save to file
    output_path = Path('telemetry-dashboard/backend/telemetry_stream.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_telemetry_stream(updates, str(output_path))
    
    # Print summary
    print(f"\nSample telemetry update:")
    print(json.dumps(updates[0], indent=2))
    
    print(f"\nUnique carriers: {len(set(u['carrier'] for u in updates))}")
    print(f"Unique flights: {len(set(u['flight_number'] for u in updates))}")
    print(f"Delayed flights: {len([u for u in updates if u.get('current_delay_minutes', 0) > 15])}")


if __name__ == "__main__":
    main()