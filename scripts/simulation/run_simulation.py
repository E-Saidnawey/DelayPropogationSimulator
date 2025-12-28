import pandas as pd
from datetime import datetime, timedelta
from scripts.simulation.initialize import load_and_prepare_data, initialize_simulation

def run_event_simulation(asset_state, telemetry_stream, turnaround_time_minutes=30):
    """
    STEP 4: Event-driven simulation loop
    
    Processes events chronologically, tracks delays, and propagates impacts.
    
    Parameters:
    -----------
    
    
    turnaround_time_minutes : int - minimum time between flights for same aircraft
    
    Returns:
    --------
    tuple : (updated_event_queue, updated_asset_state, delay_summary)
    """
    
    print("Starting simulation...")
    print(f"Processing {len(event_queue)} events")
    
    turnaround = timedelta(minutes=turnaround_time_minutes)
    delay_summary = []
    
    # Process events in chronological order
    for idx, event in telemetry_stream.iterrows():
        event_id = event['event_id']
        scheduled_time = event['scheduled_time']
        actual_time = event['actual_time']
        asset_id = event['asset_id']
        event_type = event['event_type']
        
        # ============================================================
        # STEP 4.1: Check Asset Availability
        # ============================================================
        
        # When does this aircraft become available?
        asset_available_at = asset_state[asset_id]['available_at']
        
        # For departures, we need the aircraft ready BEFORE departure
        # For arrivals, we just need to track when it lands
        if event_type == 'departure':
            required_ready_time = scheduled_time
            
            # Is the aircraft available in time?
            if asset_available_at > required_ready_time:
                # Aircraft not ready → DELAY
                asset_induced_delay = (asset_available_at - required_ready_time).total_seconds() / 60
            else:
                # Aircraft ready on time
                asset_induced_delay = 0
        else:
            # Arrivals don't need availability check
            asset_induced_delay = 0
        
        # ============================================================
        # STEP 4.2: Calculate Total Delay
        # ============================================================
        
        # Compare actual vs scheduled (this is your historical data)
        historical_delay = (actual_time - scheduled_time).total_seconds() / 60
        
        # Get any propagated delay from upstream events
        propagated_delay = event_queue.loc[idx, 'actual_delay']
        
        # Total delay is the maximum of all delay sources
        total_delay = max(historical_delay, asset_induced_delay, propagated_delay)
        
        # Update the event queue with actual delay
        event_queue.loc[idx, 'actual_delay'] = total_delay
        
        # Calculate actual event time
        actual_event_time = scheduled_time + timedelta(minutes=total_delay)
        
        # ============================================================
        # STEP 4.3: Update Asset State
        # ============================================================
        
        if event_type == 'departure':
            # Aircraft is now in flight, not available
            # We'll update availability when it arrives
            asset_state[asset_id]['current_flight'] = event_id
            
        elif event_type == 'arrival':
            # Aircraft has landed, will be available after turnaround
            asset_state[asset_id]['available_at'] = actual_event_time + turnaround
            asset_state[asset_id]['current_flight'] = None
            
            # Track utilization (flight time)
            # Find corresponding departure to calculate flight duration
            departure_event = event_queue[
                (event_queue['asset_id'] == asset_id) & 
                (event_queue['event_type'] == 'departure') &
                (event_queue['scheduled_time'] < scheduled_time)
            ].iloc[-1]  # Get most recent departure
            
            flight_duration = (actual_event_time - departure_event['scheduled_time']).total_seconds() / 3600
            asset_state[asset_id]['utilization'] += flight_duration
        
        # Track cumulative delay for this aircraft
        if total_delay > 0:
            asset_state[asset_id]['cumulative_delay'] += total_delay
            asset_state[asset_id]['delay_events'].append({
                'event_id': event_id,
                'delay_minutes': total_delay,
                'delay_type': 'asset' if asset_induced_delay > 0 else 'historical',
                'timestamp': actual_event_time
            })
        
        # ============================================================
        # STEP 4.4: Propagate Delay to Downstream Events
        # ============================================================
        
        if total_delay > 0:
            # Find all events that depend on this one
            downstream_events = dependencies_df[
                dependencies_df['upstream_event_id'] == event_id
            ]
            
            if len(downstream_events) > 0:
                # Get the IDs of affected events
                affected_ids = downstream_events['downstream_event_id'].values
                
                # Propagate delay to all downstream events (vectorized operation)
                # Only update events that haven't been processed yet
                mask = (
                    event_queue['event_id'].isin(affected_ids) & 
                    (event_queue['scheduled_time'] > actual_event_time)
                )
                
                # Add the delay to downstream events
                # Use max to ensure we don't reduce existing delays
                current_delays = event_queue.loc[mask, 'actual_delay']
                event_queue.loc[mask, 'actual_delay'] = pd.DataFrame({
                    'current': current_delays,
                    'new': total_delay
                }).max(axis=1).values
        
        # ============================================================
        # Record delay summary for analysis
        # ============================================================
        
        if total_delay > 5:  # Only track significant delays
            delay_summary.append({
                'event_id': event_id,
                'asset_id': asset_id,
                'scheduled_time': scheduled_time,
                'actual_time': actual_event_time,
                'delay_minutes': total_delay,
                'asset_induced': asset_induced_delay,
                'propagated': propagated_delay,
                'historical': historical_delay,
                'downstream_affected': len(downstream_events) if total_delay > 0 else 0
            })
    
    # Create summary DataFrame
    delay_summary_df = pd.DataFrame(delay_summary)
    
    print(f"\nSimulation complete!")
    print(f"Total delays tracked: {len(delay_summary_df)}")
    print(f"Total delay minutes: {delay_summary_df['delay_minutes'].sum():.0f}")
    
    return event_queue, asset_state, delay_summary_df


# ============================================================
# Analysis Functions
# ============================================================

def analyze_simulation_results(event_queue, asset_state, delay_summary_df):
    """
    Analyze the simulation results for insights.
    """
    
    print("\n" + "="*60)
    print("SIMULATION ANALYSIS")
    print("="*60)
    
    # 1. Overall delay statistics
    total_events = len(event_queue)
    delayed_events = len(event_queue[event_queue['actual_delay'] > 0])
    
    print(f"\nOverall Statistics:")
    print(f"  Total events: {total_events:,}")
    print(f"  Delayed events: {delayed_events:,} ({delayed_events/total_events*100:.1f}%)")
    print(f"  Average delay: {event_queue['actual_delay'].mean():.1f} minutes")
    print(f"  Max delay: {event_queue['actual_delay'].max():.1f} minutes")
    
    # 2. Asset utilization analysis
    print(f"\nAsset Utilization:")
    for asset_id, state in asset_state.items():
        utilization_loss = (state['cumulative_delay'] / 60) / state.get('baseline_utilization', 1)
        print(f"  {asset_id}:")
        print(f"    Total delay: {state['cumulative_delay']:.0f} minutes")
        print(f"    Utilization: {state['utilization']:.1f} hours")
        print(f"    Delay events: {len(state['delay_events'])}")
    
    # 3. Delay propagation analysis
    if len(delay_summary_df) > 0:
        print(f"\nDelay Propagation:")
        avg_downstream = delay_summary_df['downstream_affected'].mean()
        print(f"  Average downstream events affected: {avg_downstream:.1f}")
        
        # Top delays
        top_delays = delay_summary_df.nlargest(5, 'delay_minutes')
        print(f"\n  Top 5 Delays:")
        for _, delay in top_delays.iterrows():
            print(f"    Event {delay['event_id']}: {delay['delay_minutes']:.0f} min "
                  f"({delay['asset_id']}, affected {delay['downstream_affected']} events)")
    
    return {
        'total_delay_minutes': event_queue['actual_delay'].sum(),
        'delayed_event_count': delayed_events,
        'average_delay': event_queue['actual_delay'].mean(),
        'asset_utilization': {aid: s['utilization'] for aid, s in asset_state.items()}
    }


if __name__ == '__main__':
    flights_assets_table = 'data/processed/flight_assets_table.csv'
    flights_events_table = 'data/processed/flight_events_table.csv'
    flights_dependency_table = 'data/processed/flights_dependency_table.csv'
    flights_telemetry_table = 'data/processed/telemetry_stream.csv'

    # Usage
    df_assets, df_events, df_dep = load_and_prepare_data(
        flights_assets_table,
        flights_events_table,
        flights_telemetry_table
    )


    # Step 3: Initialize
    asset_state, event_queue, dependencies_df = initialize_simulation(
        df_assets,
        df_events,
        df_dep
    )

    # Step 4: Run simulation
    updated_events, updated_assets, delay_summary = run_event_simulation(
        asset_state=asset_state,
        event_queue=event_queue,
        dependencies_df=dependencies_df,
        turnaround_time_minutes=30
    )

    # Analyze results
    results = analyze_simulation_results(updated_events, updated_assets, delay_summary)
