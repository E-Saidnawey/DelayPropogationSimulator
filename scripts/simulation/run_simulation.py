import copy
import pandas as pd
import networkx as nx
from datetime import timedelta
from tqdm import tqdm
from typing import Dict, Set, List, Optional, Tuple, Any
from scripts.simulation.initialize import initialize_asset_states, load_and_prepare_data
import time

# --- CONFIGURATION ---
TURNAROUND_MINUTES = 45 
MIN_TURNAROUND_DELTA = timedelta(minutes=TURNAROUND_MINUTES)

def build_dependency_graph(flights_dependency_table: pd.DataFrame) -> nx.DiGraph:
    """Build directed graph from flight dependencies."""
    G = nx.DiGraph()
    edges = list(flights_dependency_table[['upstream_event_id', 'downstream_event_id']].itertuples(index=False, name=None))
    G.add_edges_from(edges)
    return G

# --- CORE LOGIC (USED BY BOTH LOOPS) ---

def process_asset_state_update(asset: Dict[str, Any], event: Any, actual_timestamp: pd.Timestamp = None) -> float:
    """
    Updates a single asset's state based on an event.
    
    Args:
        event: Can be either a pandas Series or a namedtuple from itertuples
        actual_timestamp: If provided, forces the event to start at this time (Reality).
    """
    # Handle both Series and namedtuple access
    if hasattr(event, 'scheduled_time'):
        # namedtuple from itertuples - use dot notation
        scheduled_time = event.scheduled_time
        event_type = event.event_type
        event_scheduled_duration = getattr(event, 'scheduled_duration', timedelta(0))
        event_flight_number = getattr(event, 'flight_number', None)
    else:
        # pandas Series - use bracket notation
        scheduled_time = event['scheduled_time']
        event_type = event['event_type']
        event_scheduled_duration = event.get('scheduled_duration', timedelta(0))
        event_flight_number = event.get('flight_number', None)
    
    # --- 1. DETERMINE START TIME ---
    if actual_timestamp is not None:
        # REALITY: The event happened when it happened.
        effective_start = actual_timestamp
    else:
        # PREDICTION: The event starts when the schedule says, OR when the plane arrives (whichever is later).
        effective_start = max(scheduled_time, asset['available_at'])
    
    # --- 2. CALCULATE DELAY ---
    # How late is this start compared to the schedule?
    delay_minutes = (effective_start - scheduled_time).total_seconds() / 60.0
    
    # --- 3. DETERMINE DURATION & NEXT AVAILABILITY ---
    if event_type == 'DEPARTURE':
        # BUSY = Flight Duration (Pre-calculated)
        duration = event_scheduled_duration
        
        # Metadata updates
        asset['current_flight'] = event_flight_number
        asset['last_departure_time'] = effective_start
        
    elif event_type == 'ARRIVAL':
        # BUSY = Turnaround Time (e.g., 45 mins)
        duration = MIN_TURNAROUND_DELTA
        
        # Metadata updates
        asset['current_flight'] = None
        asset['number_of_flights'] += 1
    else:
        duration = timedelta(0)

    # --- 4. UPDATE CLOCK ---
    # The asset is now busy until Start + Duration
    asset['available_at'] = effective_start + duration
    
    # --- 5. STATS ---
    if delay_minutes > 0:
        asset['cumulative_delay'] += delay_minutes
        
    return max(0.0, delay_minutes)

# --- SIMULATION FUNCTIONS ---

def get_affected_event_ids(event_id: str, dependency_graph: nx.DiGraph, max_depth: int) -> Set[str]:
    """BFS to find downstream events."""
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
    horizon_hours: float,
    affected_indices: Optional[List[int]] = None
) -> Dict:
    """
    Predictive simulation. 
    Uses the EXACT SAME update logic as the main loop via process_asset_state_update.
    """
    asset_state = cloned_state['asset_state']
    start_idx = cloned_state['telemetry_index']
    
    # Define Horizon
    start_time = telemetry_df.iloc[start_idx]['timestamp']
    horizon_end = start_time + timedelta(hours=horizon_hours)
    
    if affected_indices is not None:
        relevant_events = telemetry_df.iloc[affected_indices]
        relevant_events = relevant_events[relevant_events['timestamp'] <= horizon_end]
    else:
        # Filter for relevant future events
        # We slice first for speed, then filter by ID and Time
        future_slice = telemetry_df.iloc[start_idx + 1:]
        relevant_mask = (
            (future_slice['timestamp'] <= horizon_end) & 
            (future_slice['event_id'].isin(affected_event_ids))
        )
        relevant_events = future_slice[relevant_mask]
    
    total_future_delay = 0.0
    impacted_events = []
    event_delays = {}
    
    for _, event in relevant_events.iterrows():
        asset = asset_state[event['asset_id']]
        
        # --- SHARED LOGIC CALL ---
        delay = process_asset_state_update(asset, event)
        # -------------------------
        
        if delay > 0 and event['event_type'] == 'DEPARTURE':
            total_future_delay += delay
            impacted_events.append(event['event_id'])
            event_delays[event['event_id']] = delay
            
            # For prediction, we track specific delay events
            if 'delay_events' in asset:
                asset['delay_events'].append(event['event_id'])
    
    return {
        'total_delay': total_future_delay,
        'impacted_events': impacted_events,
        'event_delays': event_delays,
        'num_impacted': len(impacted_events)
    }


def predict_baseline_impact(
    telemetry_df: pd.DataFrame,
    asset_state: Dict,
    trigger_event: Any,  # Can be Series or namedtuple
    telemetry_index: int,
    event_df: Dict,
    config: Dict,
    event_index_map: Optional[Dict[str, List[int]]] = None
) -> Optional[Dict]:
    """Checks for delay and runs prediction if needed."""
    
    # Handle both Series and namedtuple access
    if hasattr(trigger_event, 'timestamp'):
        # namedtuple from itertuples
        trigger_timestamp = trigger_event.timestamp
        trigger_scheduled_time = trigger_event.scheduled_time
        trigger_event_id = trigger_event.event_id
        trigger_asset_id = trigger_event.asset_id
        trigger_event_carrier = trigger_event.carrier
        trigger_event_flight_num = str(trigger_event.flight_number)

    else:
        # pandas Series
        trigger_timestamp = trigger_event['timestamp']
        trigger_scheduled_time = trigger_event['scheduled_time']
        trigger_event_id = trigger_event['event_id']
        trigger_asset_id = trigger_event['asset_id']
        trigger_event_carrier = trigger_event['carrier']
        trigger_event_flight_num = str(trigger_event['flight_number'])
    
    # 1. Detect Delay using the raw timestamp
    raw_delay = (trigger_timestamp - trigger_scheduled_time).total_seconds() / 60
    
    if raw_delay < config['delay_threshold']:
        return None
        
    # 2. Clone State
    cloned_state = {
        'asset_state': {k: v.copy() for k, v in asset_state.items()},
        'telemetry_index': telemetry_index
    }
    
    # --- 3. PRIME THE SIMULATION (The Fix) ---
    # We must apply the TRIGGER event to the cloned state using 'actual_timestamp'.
    # This pushes the 'available_at' clock forward by the delay amount.
    trigger_asset = cloned_state['asset_state'][trigger_asset_id]
    process_asset_state_update(trigger_asset, trigger_event, actual_timestamp=trigger_timestamp)
    
    # 4. Get Dependencies (Successors of the trigger)
    affected_ids = event_df.get(trigger_event_id, set())
    
    if not affected_ids:
        return None
        
    affected_indices = None
    if event_index_map and trigger_event_id in event_index_map:
        affected_indices = event_index_map[trigger_event_id]

    # 5. Run Prediction on the DOWNSTREAM events
    result = fast_forward_simulation(
        telemetry_df,
        cloned_state,
        affected_ids,
        horizon_hours=config['horizon_hours'],
        affected_indices=affected_indices
    )
    
    result.update({
        'trigger_event_num': trigger_event_id,
        'trigger_carrier': trigger_event_carrier,
        'trigger_flight_number': trigger_event_flight_num,
        'trigger_delay': raw_delay,
        'trigger_timestamp': trigger_timestamp,
        'affected_event_count': len(affected_ids)
    })
    
    return result

def precompute_event_indices(telemetry_df: pd.DataFrame, dependency_df: Dict) -> Dict[str, List[int]]:
    """
    OPTIMIZATION: Pre-compute indices for each event's downstream events.
    This avoids repeated .isin() filtering in the main loop.
    """
    print("Pre-computing event indices for faster lookups...")
    event_index_map = {}
    
    # Create a mapping of event_id to its index in telemetry_df
    event_id_to_idx = {event_id: idx for idx, event_id in enumerate(telemetry_df['event_id'])}
    
    # For each event with dependencies, store the indices of affected events
    for event_id, affected_ids in tqdm(dependency_df.items(), desc="Building index map"):
        if affected_ids:
            # Convert event IDs to indices
            indices = [event_id_to_idx[aid] for aid in affected_ids if aid in event_id_to_idx]
            if indices:
                event_index_map[event_id] = sorted(indices)
    
    print(f"Indexed {len(event_index_map)} events with dependencies")
    return event_index_map

def run_simulation(
    telemetry_df: pd.DataFrame,
    asset_df: pd.DataFrame,
    flights_dependency_table: pd.DataFrame,
    delay_threshold_min: float = 20,
    prediction_horizon_hours: float = 8,
    max_dependency_depth: int = 5
) -> Tuple[pd.DataFrame, Dict]:
    
    print("\n" + "="*50)
    print("INITIALIZING SIMULATION")
    print("="*50)

    # Initialize
    print("Initializing asset states...")
    asset_state = initialize_asset_states(asset_df, telemetry_df)
    print(f"  ✓ {len(asset_state)} assets initialized")
    
    print("Building dependency graph...")
    dependency_graph = build_dependency_graph(flights_dependency_table)
    print(f"  ✓ Graph built with {len(dependency_graph.nodes())} nodes and {len(dependency_graph.edges())} edges")
    
    print("Computing affected events for each node...")
    dependency_df = {}
    for node in tqdm(dependency_graph.nodes(), desc="Computing dependencies"):
        dependency_df[node] = get_affected_event_ids(node, dependency_graph, max_dependency_depth)
    print(f"  ✓ Dependencies computed for {len(dependency_df)} events")

    config = {
        'delay_threshold': delay_threshold_min,
        'horizon_hours': prediction_horizon_hours,
        'max_depth': max_dependency_depth
    }
    
    predictions = []
    

    event_index_map = precompute_event_indices(telemetry_df, dependency_df)
    
    predictions = []
    
    # Performance tracking
    prediction_times = []
    state_update_times = []
    
    print("\n" + "="*50)
    print("RUNNING SIMULATION")
    print("="*50)
    print(f"Processing {len(telemetry_df)} events...")
    print(f"Delay threshold: {delay_threshold_min} minutes")
    print(f"Prediction horizon: {prediction_horizon_hours} hours")
    print(f"Max dependency depth: {max_dependency_depth}")
    print()
    
    with tqdm(total=len(telemetry_df), desc="Simulating events", unit="event") as pbar:
        for idx, row in enumerate(telemetry_df.itertuples(index=False)):
            # Convert row to series-like object for compatibility
            event = row
            asset = asset_state[event.asset_id]
            
            # 1. Run Prediction (Check if this event triggers a future problem)
            pred_start = time.time()
            prediction = predict_baseline_impact(
                telemetry_df, asset_state, event, idx, dependency_df, config, event_index_map
            )
            prediction_times.append(time.time() - pred_start)
            
            if prediction:
                predictions.append(prediction)
                # Update progress bar description when we find a delay trigger
                pbar.set_postfix({
                    'delays_found': len(predictions),
                    'last_delay': f"{prediction['trigger_delay']:.0f}m"
                })
                
            # 2. Update REAL State
            state_start = time.time()
            process_asset_state_update(asset, event, actual_timestamp=event.timestamp)
            state_update_times.append(time.time() - state_start)
            
            pbar.update(1)
            
            # Periodic detailed updates every 10%
            if (idx + 1) % (len(telemetry_df) // 10) == 0:
                pbar.write(f"  [{idx+1:,}/{len(telemetry_df):,}] Predictions so far: {len(predictions)}")

    # Performance statistics
    perf_stats = {
        'total_prediction_time': sum(prediction_times),
        'total_state_update_time': sum(state_update_times),
        'avg_prediction_time': sum(prediction_times) / len(prediction_times) if prediction_times else 0,
        'avg_state_update_time': sum(state_update_times) / len(state_update_times) if state_update_times else 0,
        'max_prediction_time': max(prediction_times) if prediction_times else 0,
        'total_events_processed': len(telemetry_df),
        'predictions_generated': len(predictions)
    }

    predictions_df = pd.DataFrame(predictions) if predictions else pd.DataFrame()
    
    print("\n" + "="*50)
    print("SIMULATION COMPLETE")
    print("="*50)
    print(f"✓ Processed {len(telemetry_df):,} events")
    print(f"✓ Found {len(predictions_df):,} delay triggers")
    print(f"✓ Total prediction time: {perf_stats['total_prediction_time']:.2f}s")
    print(f"✓ Total state update time: {perf_stats['total_state_update_time']:.2f}s")
    print(f"✓ Average time per event: {(perf_stats['total_prediction_time'] + perf_stats['total_state_update_time']) / len(telemetry_df) * 1000:.2f}ms")
    print()
    
    return predictions_df, asset_state, perf_stats

def analyze_simulation_results(predictions_df: pd.DataFrame):
    """
    Analyzes the simulation output to find high-impact events.
    """
    if predictions_df.empty:
        print("No delays predicted.")
        return

    print("\n" + "="*40)
    print("SIMULATION ANALYSIS REPORT")
    print("="*40)
    
    # 1. Overall Impact
    total_triggers = len(predictions_df)
    total_downstream_min = predictions_df['total_delay'].sum()
    avg_amplification = total_downstream_min / predictions_df['trigger_delay'].sum()
    
    print(f"Total Trigger Events: {total_triggers}")
    print(f"Total Downstream Delay Generated: {total_downstream_min:,.0f} minutes")
    print(f"Avg 'Ripple Effect': For every 1 min of trigger delay, we see {avg_amplification:.2f} mins downstream.")

    # 2. Worst Triggers (Top 5 events that caused the most chaos)
    print("\n--- Top 5 Most Disruptive Flights ---")
    worst_offenders = predictions_df.sort_values('total_delay', ascending=False).head(5)
    for _, row in worst_offenders.iterrows():
        print(f"Event {row['trigger_flight_number']} (Delay: {row['trigger_delay']:.0f}m)"
              f"-> Caused {row['total_delay']:.0f}m downstream delay "
              f"across {row['num_impacted']} future flights.")

    # 3. High Impact Ratio
    # Find events where a small delay caused a HUGE impact
    predictions_df['impact_ratio'] = predictions_df['total_delay'] / predictions_df['trigger_delay']
    scary_ripples = predictions_df[predictions_df['trigger_delay'] > 10].sort_values('impact_ratio', ascending=False).head(3)
    
    print("\n--- Highest Multiplier Events (Butterfly Effect) ---")
    for _, row in scary_ripples.iterrows():
        print(f"Flight {row['trigger_flight_number']}: {row['trigger_delay']:.0f}m delay "
              f"multiplied {row['impact_ratio']:.1f}x ({row['total_delay']:.0f}m total)")

def main():
    """Main execution function."""
    start_time = time.time()

    # Load data
    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    telemetry_file = 'data/processed/telemetry_stream.csv'
    dependency_graph = 'data/processed/flights_dependency_table.csv'
    flights_assets_file = 'data/processed/flight_assets_table.csv'
    
    # Convert timestamp columns to datetime
    df_assets, df_telemetry, df_dependency = load_and_prepare_data(
        flights_assets_file,
        telemetry_file,
        dependency_graph
    )
    
    print(f"  ✓ Loaded {len(df_assets):,} assets")
    print(f"  ✓ Loaded {len(df_telemetry):,} telemetry events")
    print(f"  ✓ Loaded {len(df_dependency):,} dependencies")
    
    # Run simulation
    predictions_df, final_asset_state, perf_stats = run_simulation(
        df_telemetry,
        df_assets,
        df_dependency,
        delay_threshold_min=20,
        prediction_horizon_hours=48,
        max_dependency_depth=5
    )
    
    # Save results
    if len(predictions_df) > 0:
        print("\n" + "="*50)
        print("SAVING RESULTS")
        print("="*50)
        predictions_df.to_csv('results/delay_predictions.csv', index=False)
        print(f"  ✓ Saved predictions to results/delay_predictions.csv")
        
        # Save performance stats
        perf_df = pd.DataFrame([perf_stats])
        perf_df.to_csv('results/performance_stats.csv', index=False)
        print(f"  ✓ Saved performance stats to results/performance_stats.csv")
    
    # Analyze
    analyze_simulation_results(predictions_df)

    # Summary statistics
    if len(predictions_df) > 0:
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"  • Total delay triggers: {len(predictions_df):,}")
        print(f"  • Average downstream delay per trigger: {predictions_df['total_delay'].mean():.2f} minutes")
        print(f"  • Average events impacted per trigger: {predictions_df['num_impacted'].mean():.2f}")
        print(f"  • Total predicted downstream delay: {predictions_df['total_delay'].sum():.2f} minutes")
        print(f"  • Total execution time: {time.time() - start_time:.2f} seconds")
        print()

    return predictions_df, final_asset_state


if __name__ == "__main__":
    predictions_df, final_asset_state = main()