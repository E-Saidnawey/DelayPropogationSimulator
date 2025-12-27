"""
Simple Environment Table Generator - Using Existing Flight Delay Data
No external weather API needed! Uses WeatherDelay column you already have.
"""

import pandas as pd
import numpy as np

def generate_environment_from_delays(flight_data_csv, output_file=None):
    """
    Generate environment table directly from flight delay columns
        
    Args:
        flight_data_csv: Path to your flight CSV with delay columns
        output_file: Output environment CSV file
        
    Returns:
        DataFrame with environment table
    """
    
    if output_file is None:
        raise ValueError('Implement output file location')

    print("="*70)
    print("GENERATING ENVIRONMENT TABLE FROM FLIGHT DELAYS")
    print("="*70)
    
    # Load flight data
    print(f"\nLoading flight data from {flight_data_csv}...")
    flights = pd.read_csv(flight_data_csv, parse_dates=['FL_DATE'])
    
    print(f"✓ Loaded {len(flights):,} flights")
    
    # Check for required columns
    required_cols = ['FL_DATE', 'ORIGIN_AIRPORT_ID', 'WEATHER_DELAY']
    missing_cols = [col for col in required_cols if col not in flights.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Optional columns for richer analysis
    optional_cols = {
        'CANCELLATION_CODE': False,
        'DIVERTED': False,
        'DEP_DELAY': False,
        'CarrierDelay': False,
        'NASDelay': False,
        'SecurityDelay': False,
        'LateAircraftDelay': False
    }
    
    for col in optional_cols:
        optional_cols[col] = col in flights.columns
    
    print(f"\nAvailable delay columns:")
    print(f"  ✓ WeatherDelay (required)")
    for col, available in optional_cols.items():
        status = "✓" if available else "✗"
        print(f"  {status} {col}")
    
    # Group by date and airport
    print(f"\nAggregating by date and airport...")
    grouped = flights.groupby(['FL_DATE', 'ORIGIN_AIRPORT_ID'])
    
    environment_records = []
    
    for (date, airport), group in grouped:
        total_flights = len(group)
        
        # Core weather metrics from WeatherDelay
        weather_delays = group['WEATHER_DELAY'].fillna(0)
        avg_weather_delay = weather_delays.mean()
        max_weather_delay = weather_delays.max()
        flights_with_weather_delay = (weather_delays > 0).sum()
        
        # Additional metrics if available
        weather_cancellations = 0
        if optional_cols['CANCELLATION_CODE']:
            weather_cancellations = len(group[group['CANCELLATION_CODE'] == 'B'])
        
        diversions = 0
        if optional_cols['DIVERTED']:
            diversions = len(group[group['DIVERTED'] == 1])
        
        # Calculate weather severity index (0-10 scale)
        severity = calculate_weather_severity(
            avg_weather_delay=avg_weather_delay,
            max_weather_delay=max_weather_delay,
            pct_flights_delayed=flights_with_weather_delay / total_flights,
            weather_cancellations=weather_cancellations,
            total_flights=total_flights,
            diversions=diversions
        )
        
        # Build record
        record = {
            'timestamp': date,
            'location': airport,
            'weather_severity_index': severity,
            'avg_weather_delay_min': avg_weather_delay,
            'max_weather_delay_min': max_weather_delay,
            'flights_affected': total_flights,
            'flights_with_weather_delay': flights_with_weather_delay,
            'pct_flights_delayed': flights_with_weather_delay / total_flights * 100,
            'weather_cancellations': weather_cancellations,
            'diversions': diversions,
            'data_source': 'FLIGHT_WEATHER_DELAY'
        }
        
        # Add optional delay breakdown
        if optional_cols['CarrierDelay']:
            record['avg_carrier_delay_min'] = group['CarrierDelay'].fillna(0).mean()
        if optional_cols['NASDelay']:
            record['avg_nas_delay_min'] = group['NASDelay'].fillna(0).mean()
        if optional_cols['SecurityDelay']:
            record['avg_security_delay_min'] = group['SecurityDelay'].fillna(0).mean()
        if optional_cols['LateAircraftDelay']:
            record['avg_late_aircraft_delay_min'] = group['LateAircraftDelay'].fillna(0).mean()
        
        environment_records.append(record)
    
    # Create DataFrame
    environment = pd.DataFrame(environment_records)
    
    # Sort by date and airport
    environment = environment.sort_values(['timestamp', 'location'])
    
    # Save to CSV
    environment.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Total environment records: {len(environment):,}")
    print(f"Date range: {environment['timestamp'].min()} to {environment['timestamp'].max()}")
    print(f"Airports: {environment['location'].nunique()}")
    print(f"Output saved to: {output_file}")
    
    print(f"\nWeather Severity Statistics:")
    print(environment['weather_severity_index'].describe())
    
    print(f"\nSample records:")
    print(environment.head(10))
    
    return environment


def calculate_weather_severity(
    avg_weather_delay,
    max_weather_delay,
    pct_flights_delayed,
    weather_cancellations,
    total_flights,
    diversions
):
    """
    Calculate weather severity index (0-10 scale) from delay data
    
    Components:
    - Average weather delay (0-4 points)
    - Percentage of flights delayed (0-3 points)
    - Cancellations (0-2 points)
    - Extreme delays (0-1 point)
    """
    severity = 0
    
    # Component 1: Average weather delay (0-4 points)
    # 0 min = 0 pts, 15 min = 1 pt, 30 min = 2 pts, 60 min = 3 pts, 90+ min = 4 pts
    if avg_weather_delay > 0:
        if avg_weather_delay >= 90:
            severity += 4
        elif avg_weather_delay >= 60:
            severity += 3
        elif avg_weather_delay >= 30:
            severity += 2
        elif avg_weather_delay >= 15:
            severity += 1
    
    # Component 2: Percentage of flights with weather delays (0-3 points)
    # 0% = 0 pts, 10% = 1 pt, 25% = 2 pts, 50%+ = 3 pts
    if pct_flights_delayed >= 0.50:
        severity += 3
    elif pct_flights_delayed >= 0.25:
        severity += 2
    elif pct_flights_delayed >= 0.10:
        severity += 1
    
    # Component 3: Weather cancellations (0-2 points)
    # Based on cancellation rate
    if total_flights > 0:
        cancel_rate = weather_cancellations / total_flights
        if cancel_rate >= 0.10:  # 10%+ cancellations
            severity += 2
        elif cancel_rate >= 0.05:  # 5%+ cancellations
            severity += 1
    
    # Component 4: Extreme individual delays (0-1 point)
    # If any flight had 120+ min weather delay
    if max_weather_delay >= 120:
        severity += 1
    
    return min(severity, 10)  # Cap at 10


def compare_delay_types(flight_data_csv):
    """
    Analyze and compare different delay types
    Helps understand what's driving delays
    """
    print("\n" + "="*70)
    print("DELAY TYPE COMPARISON")
    print("="*70)
    
    flights = pd.read_csv(flight_data_csv)
    
    delay_columns = ['WEATHER_DELAY', 'CARRIER_DELAY', 'NAS_DELAY', 
                     'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']

    available_delays = [col for col in delay_columns if col in flights.columns]
    
    print(f"\nAverage delay by type (minutes):")
    for col in available_delays:
        avg_delay = flights[col].fillna(0).mean()
        flights_delayed = (flights[col] > 0).sum()
        pct_delayed = flights_delayed / len(flights) * 100
        print(f"  {col:20s}: {avg_delay:6.2f} min avg, {pct_delayed:5.1f}% of flights")
    
    print(f"\nTotal delay minutes by type:")
    for col in available_delays:
        total = flights[col].fillna(0).sum()
        print(f"  {col:20s}: {total:,.0f} min total")
    
    # Weather vs non-weather
    if all(col in flights.columns for col in available_delays):
        weather_total = flights['WEATHER_DELAY'].fillna(0).sum()
        other_total = sum(flights[col].fillna(0).sum() 
                         for col in ['CARRIER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'])
        
        print(f"\nWeather vs Other Delays:")
        print(f"  Weather:  {weather_total:,.0f} min ({weather_total/(weather_total+other_total)*100:.1f}%)")
        print(f"  Other:    {other_total:,.0f} min ({other_total/(weather_total+other_total)*100:.1f}%)")


if __name__ == "__main__":
    
    flight_file = 'data/processed/flights_cleaned_us_only.csv'
    output_file = 'data/processed/flights_weather_table.csv'

    # Generate environment table
    environment = generate_environment_from_delays(flight_file, output_file)
    
    # Show delay comparison
    compare_delay_types(flight_file)
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)