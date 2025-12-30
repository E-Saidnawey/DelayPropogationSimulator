import pandas as pd
import pytz
from datetime import datetime

# US state to timezone mapping
STATE_TIMEZONES = {
    'AL': 'America/Chicago', 'AK': 'America/Anchorage', 'AZ': 'America/Phoenix',
    'AR': 'America/Chicago', 'CA': 'America/Los_Angeles', 'CO': 'America/Denver',
    'CT': 'America/New_York', 'DE': 'America/New_York', 'FL': 'America/New_York',
    'GA': 'America/New_York', 'HI': 'Pacific/Honolulu', 'ID': 'America/Denver',
    'IL': 'America/Chicago', 'IN': 'America/New_York', 'IA': 'America/Chicago',
    'KS': 'America/Chicago', 'KY': 'America/New_York', 'LA': 'America/Chicago',
    'ME': 'America/New_York', 'MD': 'America/New_York', 'MA': 'America/New_York',
    'MI': 'America/New_York', 'MN': 'America/Chicago', 'MS': 'America/Chicago',
    'MO': 'America/Chicago', 'MT': 'America/Denver', 'NE': 'America/Chicago',
    'NV': 'America/Los_Angeles', 'NH': 'America/New_York', 'NJ': 'America/New_York',
    'NM': 'America/Denver', 'NY': 'America/New_York', 'NC': 'America/New_York',
    'ND': 'America/Chicago', 'OH': 'America/New_York', 'OK': 'America/Chicago',
    'OR': 'America/Los_Angeles', 'PA': 'America/New_York', 'RI': 'America/New_York',
    'SC': 'America/New_York', 'SD': 'America/Chicago', 'TN': 'America/Chicago',
    'TX': 'America/Chicago', 'UT': 'America/Denver', 'VT': 'America/New_York',
    'VA': 'America/New_York', 'WA': 'America/Los_Angeles', 'WV': 'America/New_York',
    'WI': 'America/Chicago', 'WY': 'America/Denver'
}

# US states and territories list (for validation)
US_STATES = set(STATE_TIMEZONES.keys())

# Special cases for states with multiple timezones
SPECIAL_TIMEZONE_OVERRIDES = {
    'FL': {
        'Pensacola': 'America/Chicago',
        'Panama City': 'America/Chicago',
    },
    'TX': {
        'El Paso': 'America/Denver',
    },
    'KY': {
        'Louisville': 'America/New_York',
        'Lexington': 'America/New_York',
    },
    'TN': {
        'Nashville': 'America/Chicago',
        'Memphis': 'America/Chicago',
        'Knoxville': 'America/New_York',
    },
    'KS': {
        'Dodge City': 'America/Denver',
    },
    'NE': {
        'Scottsbluff': 'America/Denver',
    },
    'SD': {
        'Rapid City': 'America/Denver',
    },
    'ND': {
        'Dickinson': 'America/Denver',
    },
    'OR': {
        'Ontario': 'America/Denver',
    },
    'ID': {
        'Coeur d\'Alene': 'America/Los_Angeles',
    }
}


def load_airport_lookup(lookup_file):
    """
    Load and parse the airport lookup table, identifying US airports
    """
    # Read the lookup table
    airports = pd.read_csv(lookup_file)
    
    # Parse the Description to extract city and state
    # Format: "City, ST: Airport Name"
    airports[['location', 'airport_name']] = airports['Description'].str.split(':', n=1, expand=True)
    airports[['city', 'state']] = airports['location'].str.split(',', n=1, expand=True)
    
    # Clean up whitespace
    airports['city'] = airports['city'].str.strip()
    airports['state'] = airports['state'].str.strip()
    airports['airport_name'] = airports['airport_name'].str.strip()
    
    # Flag US airports
    airports['is_us'] = airports['state'].isin(US_STATES)
    
    # Map state to timezone (only for US airports)
    airports['timezone'] = airports.apply(
        lambda row: STATE_TIMEZONES.get(row['state']) if row['is_us'] else None,
        axis=1
    )
    
    # Handle special cases (cities with different timezone than state default)
    for state, city_overrides in SPECIAL_TIMEZONE_OVERRIDES.items():
        for city, tz in city_overrides.items():
            mask = (airports['state'] == state) & (airports['city'].str.contains(city, case=False, na=False))
            airports.loc[mask, 'timezone'] = tz
    
    # Create lookup dictionaries
    airport_tz_dict = dict(zip(airports['Code'].astype(str), airports['timezone']))
    us_airports_set = set(airports[airports['is_us']]['Code'].astype(str))
    
    return airports, airport_tz_dict, us_airports_set


def clean_and_convert_times(df, airport_lookup_file, output_file):
    """
    Complete cleaning script with timezone conversion and US-only filtering
    
    Parameters:
    - df: Input dataframe with flight data
    - airport_lookup_file: Path to airport lookup CSV
    - output_file: Path to save cleaned data
    """
    
    print("Loading airport timezone data...")
    airports_df, airport_tz_dict, us_airports_set = load_airport_lookup(airport_lookup_file)
    
    print(f"Loaded {len(airport_tz_dict)} total airports")
    print(f"Identified {len(us_airports_set)} US airports")
    
    # Step 1: Filter for US-only flights FIRST (before other filtering)
    print("\nFiltering for US-only flights...")
    initial_rows = len(df)
    
    # Convert airport IDs to string for matching
    df['ORIGIN_AIRPORT_ID'] = df['ORIGIN_AIRPORT_ID'].astype(str)
    df['DEST_AIRPORT_ID'] = df['DEST_AIRPORT_ID'].astype(str)
    
    # Filter: both origin AND destination must be in US
    us_flight_mask = (
        df['ORIGIN_AIRPORT_ID'].isin(us_airports_set) &
        df['DEST_AIRPORT_ID'].isin(us_airports_set)
    )
    
    df = df[us_flight_mask].copy()
    
    print(f"Filtered from {initial_rows:,} to {len(df):,} rows")
    print(f"Removed {initial_rows - len(df):,} international/non-US flights ({(initial_rows - len(df))/initial_rows*100:.1f}%)")
    
    # Show what was filtered out
    non_us_origins = set(df[~df['ORIGIN_AIRPORT_ID'].isin(us_airports_set)]['ORIGIN_AIRPORT_ID'].unique()) if not us_flight_mask.all() else set()
    non_us_dests = set(df[~df['DEST_AIRPORT_ID'].isin(us_airports_set)]['DEST_AIRPORT_ID'].unique()) if not us_flight_mask.all() else set()
    
    if non_us_origins or non_us_dests:
        print(f"\nNon-US airports excluded:")
        if non_us_origins:
            sample_origins = list(non_us_origins)[:5]
            print(f"  Origins: {sample_origins} {'...' if len(non_us_origins) > 5 else ''} ({len(non_us_origins)} total)")
        if non_us_dests:
            sample_dests = list(non_us_dests)[:5]
            print(f"  Destinations: {sample_dests} {'...' if len(non_us_dests) > 5 else ''} ({len(non_us_dests)} total)")
    
    # Step 2: Filter out cancelled flights and missing data
    print("\nFiltering cancelled and incomplete flights...")
    pre_filter_rows = len(df)
    
    df = df[
        (df['CANCELLED'] == 0) & 
        (df['TAIL_NUM'].notna()) & 
        (df['CRS_DEP_TIME'].notna()) & 
        (df['CRS_ARR_TIME'].notna()) &
        (df['ORIGIN_AIRPORT_ID'].notna()) &
        (df['DEST_AIRPORT_ID'].notna())
    ].copy()
    
    print(f"Filtered from {pre_filter_rows:,} to {len(df):,} rows ({len(df)/pre_filter_rows*100:.1f}% retained)")
    
    if len(df) == 0:
        print("\n⚠️  WARNING: No data remaining after filtering! Check your input data.")
        return df
    
    # Step 3: Parse base dates
    print("\nParsing dates...")
    date_format = '%m/%d/%Y %I:%M:%S %p'
    df['base_date'] = pd.to_datetime(df['FL_DATE'], format=date_format).dt.normalize()
    
    # Step 4: Create naive datetimes (local time, no timezone yet)
    print("Creating naive datetimes...")
    
    # Scheduled times
    df['scheduled_dep_local'] = df['base_date'] + \
        pd.to_timedelta(df['CRS_DEP_TIME'] // 100, unit='h') + \
        pd.to_timedelta(df['CRS_DEP_TIME'] % 100, unit='m')
    
    df['scheduled_arr_local'] = df['base_date'] + \
        pd.to_timedelta(df['CRS_ARR_TIME'] // 100, unit='h') + \
        pd.to_timedelta(df['CRS_ARR_TIME'] % 100, unit='m')
    
    # Actual times (use scheduled if actual is missing)
    df['actual_dep_local'] = df['scheduled_dep_local'].copy()
    dep_mask = df['DEP_TIME'].notna()
    df.loc[dep_mask, 'actual_dep_local'] = df.loc[dep_mask, 'base_date'] + \
        pd.to_timedelta(df.loc[dep_mask, 'DEP_TIME'] // 100, unit='h') + \
        pd.to_timedelta(df.loc[dep_mask, 'DEP_TIME'] % 100, unit='m')
    
    df['actual_arr_local'] = df['scheduled_arr_local'].copy()
    arr_mask = df['ARR_TIME'].notna()
    df.loc[arr_mask, 'actual_arr_local'] = df.loc[arr_mask, 'base_date'] + \
        pd.to_timedelta(df.loc[arr_mask, 'ARR_TIME'] // 100, unit='h') + \
        pd.to_timedelta(df.loc[arr_mask, 'ARR_TIME'] % 100, unit='m')
    
    # Step 5: Handle midnight crossings (before timezone conversion)
    print("Handling midnight crossings...")
    
    crosses_midnight = df['scheduled_arr_local'] < df['scheduled_dep_local']
    df.loc[crosses_midnight, 'scheduled_arr_local'] += pd.Timedelta(days=1)
    print(f"  Scheduled: {crosses_midnight.sum():,} flights cross midnight")
    
    crosses_midnight = df['actual_arr_local'] < df['actual_dep_local']
    df.loc[crosses_midnight, 'actual_arr_local'] += pd.Timedelta(days=1)
    print(f"  Actual: {crosses_midnight.sum():,} flights cross midnight")
    
    # Step 6: Convert to UTC
    print("\nConverting to UTC...")
    
    df['scheduled_dep_utc'] = convert_to_utc(
        df['scheduled_dep_local'],
        df['ORIGIN_AIRPORT_ID'],
        airport_tz_dict,
        'departure'
    )
    
    df['scheduled_arr_utc'] = convert_to_utc(
        df['scheduled_arr_local'],
        df['DEST_AIRPORT_ID'],
        airport_tz_dict,
        'arrival'
    )
    
    df['actual_dep_utc'] = convert_to_utc(
        df['actual_dep_local'],
        df['ORIGIN_AIRPORT_ID'],
        airport_tz_dict,
        'departure'
    )
    
    df['actual_arr_utc'] = convert_to_utc(
        df['actual_arr_local'],
        df['DEST_AIRPORT_ID'],
        airport_tz_dict,
        'arrival'
    )
    # remove instances where flights are rebooked and show up twice in our dataset
    df = remove_impossible_overlaps(df)

    # Step 7: Calculate delay columns
    print("\nCalculating delays...")
    df['dep_delay_minutes'] = (df['actual_dep_utc'] - df['scheduled_dep_utc']).dt.total_seconds() / 60
    df['arr_delay_minutes'] = (df['actual_arr_utc'] - df['scheduled_arr_utc']).dt.total_seconds() / 60
    
    # Step 8: Drop intermediate columns
    columns_to_drop = ['base_date', 'scheduled_dep_local', 'scheduled_arr_local', 
                       'actual_dep_local', 'actual_arr_local']
    df = df.drop(columns=columns_to_drop)
    
    # Step 9: Save to file
    print(f"\nSaving cleaned data to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("CLEANING SUMMARY")
    print("="*60)
    print(f"Total US domestic flights: {len(df):,}")
    print(f"Date range: {df['FL_DATE'].min()} to {df['FL_DATE'].max()}")
    print(f"Unique aircraft: {df['TAIL_NUM'].nunique():,}")
    print(f"Unique origin airports: {df['ORIGIN_AIRPORT_ID'].nunique():,}")
    print(f"Unique destination airports: {df['DEST_AIRPORT_ID'].nunique():,}")
    print(f"\nTop 10 Origin Airports:")
    print(df['ORIGIN_AIRPORT_ID'].value_counts().head(10))
    print(f"\nTop 10 Destination Airports:")
    print(df['DEST_AIRPORT_ID'].value_counts().head(10))
    print(f"\nDelay Statistics:")
    print(f"  Mean departure delay: {df['dep_delay_minutes'].mean():.1f} minutes")
    print(f"  Mean arrival delay: {df['arr_delay_minutes'].mean():.1f} minutes")
    print(f"  Median departure delay: {df['dep_delay_minutes'].median():.1f} minutes")
    print(f"  Median arrival delay: {df['arr_delay_minutes'].median():.1f} minutes")
    print(f"  Flights with departure delay > 15 min: {(df['dep_delay_minutes'] > 15).sum():,}")
    print(f"  Flights with arrival delay > 15 min: {(df['arr_delay_minutes'] > 15).sum():,}")
    print(f"  Flights with departure delay > 60 min: {(df['dep_delay_minutes'] > 60).sum():,}")
    print("="*60)
    
    return df

def remove_impossible_overlaps(df):
    # Sort by asset and time
    df = df.sort_values(['TAIL_NUM', 'scheduled_dep_utc'])
    
    # 1. Drop exact duplicates (same plane, same time, same flight)
    df = df.drop_duplicates(subset=['TAIL_NUM', 'scheduled_dep_utc'])
    
    # 2. Drop "Impossible" overlaps 
    # (Where a plane's next departure is before its previous arrival)
    # We use shift to compare the previous arrival time to the current departure
    df['prev_arr_utc'] = df.groupby('TAIL_NUM')['scheduled_arr_utc'].shift(1)
    
    # A flight is 'impossible' if it starts before the last one ended
    # We give it a 10-minute buffer for safety
    is_impossible = (df['scheduled_dep_utc'] < df['prev_arr_utc'])
    
    print(f"Removing {is_impossible.sum()} impossible overlapping flights...")
    return df[~is_impossible].drop(columns=['prev_arr_utc'])

def convert_to_utc(datetime_series, airport_id_series, airport_tz_dict, event_type):
    """
    Convert local datetime to UTC based on airport timezone
    
    Parameters:
    - datetime_series: Series of naive datetimes in local time
    - airport_id_series: Series of airport IDs
    - airport_tz_dict: Dictionary mapping airport_id -> timezone name
    - event_type: 'departure' or 'arrival' (for logging)
    """
    result = pd.Series(index=datetime_series.index, dtype='datetime64[ns, UTC]')
    
    unknown_airports = set()
    
    # Process by unique airport for efficiency
    for airport_id in airport_id_series.unique():
        if pd.isna(airport_id):
            continue
        
        mask = airport_id_series == airport_id
        
        # Get timezone for this airport
        tz_name = airport_tz_dict.get(str(airport_id))
        
        if tz_name is None or pd.isna(tz_name):
            unknown_airports.add(airport_id)
            # This shouldn't happen for US airports, but handle it
            print(f"  ⚠️  WARNING: No timezone found for US airport {airport_id}")
            tz = pytz.UTC
            result[mask] = datetime_series[mask].dt.tz_localize(tz)
        else:
            try:
                tz = pytz.timezone(tz_name)
                # Localize to airport timezone, then convert to UTC
                # ambiguous='NaT' handles DST transitions
                # nonexistent='NaT' handles invalid times during DST
                localized = datetime_series[mask].dt.tz_localize(
                    tz, 
                    ambiguous='NaT', 
                    nonexistent='NaT'
                )
                result[mask] = localized.dt.tz_convert('UTC')
            except Exception as e:
                print(f"  ⚠️  Error converting timezone for airport {airport_id}: {e}")
                result[mask] = datetime_series[mask].dt.tz_localize(pytz.UTC)
    
    if unknown_airports:
        print(f"  Warning: {len(unknown_airports)} US airports with unknown timezones in {event_type}")
        print(f"    Unknown airports: {list(unknown_airports)[:10]}")
    
    return result


# Main execution
if __name__ == "__main__":
    # File paths - UPDATE THESE
    input_file = "data/raw/flight_data_raw.csv"
    airport_lookup_file = "data/raw/airport_lookup.csv"
    output_file = "data/processed/flights_cleaned_us_only.csv"
    
    # Load flight data
    print(f"Loading flight data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows")
    
    # Clean and convert
    cleaned_df = clean_and_convert_times(df, airport_lookup_file, output_file)
    
    if len(cleaned_df) > 0:
        print(f"\n✅ Done! Cleaned data saved to {output_file}")
        
        # Optional: Show a sample
        print("\nSample of cleaned data:")
        print(cleaned_df[['TAIL_NUM', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 
                          'scheduled_dep_utc', 'scheduled_arr_utc', 
                          'dep_delay_minutes', 'arr_delay_minutes']].head(10))
    else:
        print(f"\n❌ No data to save. Check your input files and filters.")