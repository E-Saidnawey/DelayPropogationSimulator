import copy
import pandas as pd
import networkx as nx
from datetime import timedelta
from typing import Dict, Set, List, Optional, Tuple
import numpy as np
from scripts.simulation.initialize import initialize_asset_states, load_and_prepare_data

DELAY_THRESHOLD_MIN = 20 # minutes
PREDICTION_HORIZON_HOURS = 8 # hours
MAX_DEPENDENCY_DEPTH = 5 # flights


def build_dependency_graph(flights_dependency_table: pd.DataFrame) -> nx.DiGraph:
    """
    Build directed graph from flight dependencies.
    Optimized for large datasets (490k+ rows).
    """
    edges = list(flights_dependency_table[['upstream_event_id', 'downstream_event_id']].itertuples(index=False, name=None))
    
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    return G


def detect_delay(event: pd.Series, scheduled_col: str = 'scheduled_time') -> Optional[float]:
    """Detect if event delay exceeds threshold."""
    delay = (event['timestamp'] - event[scheduled_col]).total_seconds() / 60
    return delay if delay >= DELAY_THRESHOLD_MIN else None


def clone_simulation_state(asset_state: Dict, telemetry_index: int) -> Dict:
    """Create deep copy of simulation state."""
    return {
        'asset_state': copy.deepcopy(asset_state),
        'telemetry_index': telemetry_index
    }


def get_affected_event_ids(
    event_id: str,
    dependency_graph: nx.DiGraph,
    max_depth: int = MAX_DEPENDENCY_DEPTH
) -> Set[str]:
    """Get downstream events affected by delay using BFS with depth limit."""
    affected = set()
    frontier = [(event_id, 0)]
    
    while frontier:
        current, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        
        if current in dependency_graph:
            for downstream in dependency_graph.successors(current):
                if downstream not in affected:
                    affected.add(downstream)
                    frontier.append((downstream, depth + 1))
    
    return affected


def fast_forward_simulation(
    telemetry_df: pd.DataFrame,
    cloned_state: Dict,
    affected_event_ids: Set[str],
    horizon_hours: float = PREDICTION_HORIZON_HOURS
) -> Dict:
    """Simulate future events to predict downstream delay impact."""
    asset_state = cloned_state['asset_state']
    start_idx = cloned_state['telemetry_index']
    start_time = telemetry_df.iloc[start_idx]['timestamp']
    horizon_end = start_time + timedelta(hours=horizon_hours)
    
    total_future_delay = 0.0
    impacted_events = []
    event_delays = {}
    
    # Pre-filter telemetry for affected events within horizon
    future_slice = telemetry_df.iloc[start_idx + 1:]
    mask = (future_slice['timestamp'] <= horizon_end) & (future_slice['event_id'].isin(affected_event_ids))
    relevant_events = future_slice[mask]
    
    for idx, event in relevant_events.iterrows():
        asset_id = event['asset_id']
        asset = asset_state[asset_id]
        
        # Calculate effective start time considering asset availability
        effective_start = max(event['scheduled_time'], asset['available_at'])
        delay_min = (effective_start - event['scheduled_time']).total_seconds() / 60
        
        if event['event_type'] == 'DEPARTURE':
            # If it's a Departure, it's starting a FLIGHT.
            # It won't be 'available' at the next airport until it flies there.
            # Duration = Original Scheduled Arrival - Original Scheduled Departure
            # You should have 'scheduled_arr_utc' or a 'duration' available in your telemetry
            flight_duration = event['scheduled_arrival_time'] - event['scheduled_time']
            
            # The plane is available at the destination after the flight
            asset['available_at'] = effective_start + flight_duration
            
        elif event['event_type'] == 'ARRIVAL':
            # If it's an Arrival, it's starting a TURNAROUND.
            # It won't be 'available' for the next flight until serviced.
            min_turnaround = timedelta(minutes=30) 
            
            # The plane is available for the next departure after the 45-min buffer
            asset['available_at'] = effective_start + min_turnaround

        if delay_min > 0:
            total_future_delay += delay_min
            impacted_events.append(event['event_id'])
            event_delays[event['event_id']] = delay_min
            
            asset['cumulative_delay'] += delay_min
            asset['delay_events'].append(event['event_id'])
        
        # Update asset availability based on event duration
        # Assuming event ends at timestamp (can be adjusted if duration column exists)
        asset['available_at'] = effective_start
        asset['last_event_id'] = event['event_id']
    
    return {
        'total_delay': total_future_delay,
        'impacted_events': impacted_events,
        'event_delays': event_delays,
        'num_impacted': len(impacted_events)
    }


def predict_baseline_impact(
    telemetry_df: pd.DataFrame,
    asset_state: Dict,
    dependency_graph: nx.DiGraph,
    trigger_event: pd.Series,
    telemetry_index: int
) -> Optional[Dict]:
    """Predict downstream impact of a delay trigger event."""
    delay = detect_delay(trigger_event)
    if delay is None:
        return None
    
    # Clone current state
    cloned_state = clone_simulation_state(asset_state, telemetry_index)
    
    # Get affected downstream events
    affected_event_ids = get_affected_event_ids(
        trigger_event['event_id'],
        dependency_graph,
        max_depth=MAX_DEPENDENCY_DEPTH
    )
    
    # Fast-forward simulation on cloned state
    baseline_result = fast_forward_simulation(
        telemetry_df,
        cloned_state,
        affected_event_ids,
        horizon_hours=PREDICTION_HORIZON_HOURS
    )
    
    # Add trigger information
    baseline_result['trigger_event_id'] = trigger_event['event_id']
    baseline_result['trigger_delay'] = delay
    baseline_result['trigger_timestamp'] = trigger_event['timestamp']
    baseline_result['affected_event_count'] = len(affected_event_ids)
    
    return baseline_result


def run_simulation(
    telemetry_df: pd.DataFrame,
    asset_df: pd.DataFrame,
    flights_dependency_table: pd.DataFrame,
    delay_threshold_min: float = DELAY_THRESHOLD_MIN,
    prediction_horizon_hours: float = PREDICTION_HORIZON_HOURS,
    max_dependency_depth: int = MAX_DEPENDENCY_DEPTH
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run full simulation with delay detection and impact prediction.
    
    Returns:
        - predictions_df: DataFrame with all delay predictions
        - final_asset_state: Final state of all assets
    """
    global DELAY_THRESHOLD_MIN, PREDICTION_HORIZON_HOURS, MAX_DEPENDENCY_DEPTH
    DELAY_THRESHOLD_MIN = delay_threshold_min
    PREDICTION_HORIZON_HOURS = prediction_horizon_hours
    MAX_DEPENDENCY_DEPTH = max_dependency_depth
    TURNAROUND_MINUTES = 30
    # Sort telemetry by timestamp
    telemetry_df = telemetry_df.sort_values('timestamp').reset_index(drop=True)
    
    # Initialize
    asset_state = initialize_asset_states(asset_df, telemetry_df)
    dependency_graph = build_dependency_graph(flights_dependency_table)
    
    predictions = []
    
    # Process telemetry stream
    for idx, event in telemetry_df.iterrows():
        asset_id = event['asset_id']
        asset = asset_state[asset_id]
        
        event_type = event['event_type'].upper()
    
        if event_type == 'DEPARTURE':
            asset['current_flight'] = event['flight_number']
            asset['last_departure_time'] = event['timestamp']
            
        elif event_type == 'ARRIVAL':
            # Aircraft arrives - becomes available after turnaround
            turnaround_delta = timedelta(minutes=TURNAROUND_MINUTES)
            asset['available_at'] = event['timestamp'] + turnaround_delta
            asset['current_flight'] = None  # Flight completed
            asset['total_flight_time'] = asset['total_flight_time'] + (event['timestamp'] - asset['last_departure_time'])
            asset['number_of_flights'] += 1

        else:
            raise ValueError('Unknown Event Type in Telemetry Data')
        
        # Check for delay trigger
        prediction = predict_baseline_impact(
            telemetry_df,
            asset_state,
            dependency_graph,
            event,
            idx
        )
        
        if prediction is not None:
            predictions.append(prediction)
    
    # Convert predictions to DataFrame
    if predictions:
        predictions_df = pd.DataFrame(predictions)
    else:
        predictions_df = pd.DataFrame(columns=[
            'trigger_event_id', 'trigger_delay', 'trigger_timestamp',
            'total_delay', 'impacted_events', 'num_impacted',
            'affected_event_count', 'event_delays'
        ])
    
    return predictions_df, asset_state


def main():
    """Main execution function."""
    # Load data
    print("Loading data...")
    telemetry_file = 'data/processed/telemetry_stream.csv'
    dependency_graph = 'data/processed/flights_dependency_table.csv'
    flights_assets_file = 'data/processed/flight_assets_table.csv'
    flights_events_file = 'data/processed/'
    
    # Convert timestamp columns to datetime
    df_assets, df_telemetry, df_dependency = load_and_prepare_data(
        flights_assets_file,
        telemetry_file,
        dependency_graph
    )
    
    # Run simulation
    print("Running simulation...")
    predictions_df, final_asset_state = run_simulation(
        df_telemetry,
        df_assets,
        df_dependency,
        delay_threshold_min=20,
        prediction_horizon_hours=3,
        max_dependency_depth=3
    )
    
    # Save results
    print(f"Simulation complete. Found {len(predictions_df)} delay triggers.")
    predictions_df.to_csv('delay_predictions.csv', index=False)
    
    # Summary statistics
    if len(predictions_df) > 0:
        print(f"\nSummary Statistics:")
        print(f"Total delay triggers: {len(predictions_df)}")
        print(f"Average downstream delay per trigger: {predictions_df['total_delay'].mean():.2f} minutes")
        print(f"Average events impacted per trigger: {predictions_df['num_impacted'].mean():.2f}")
        print(f"Total predicted downstream delay: {predictions_df['total_delay'].sum():.2f} minutes")
    
    return predictions_df, final_asset_state


if __name__ == "__main__":
    predictions_df, final_asset_state = main()