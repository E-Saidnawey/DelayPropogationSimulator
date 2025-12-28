# Delay Propagation & Recovery Simulator

**Operational impact modeling for tightly coupled mission systems**

## Overview

This project implements an event-driven delay propagation simulator that models how a single operational disruption cascades through a tightly coupled system and identifies the highest-leverage intervention points for operators in real time.

The system is designed to mirror mission operations environments (airline operations, hardware test campaigns, launch/test timelines), where:
- Assets are reused across sequential tasks
- Small delays propagate non-linearly
- Operators must decide where to intervene now, not analyze history later
- The simulator treats operational data as telemetry and continuously evaluates counterfactual recovery actions to support operator decision-making

### Key Capabilities
- Telemetry-driven, event-ordered simulation
- Implicit delay propagation via asset availability
- Environmental disruption injection (weather)
- Real-time detection of intervention windows
- Counterfactual rollout evaluation of operator actions
- Ranking of highest-leverage recovery options
- Operator-focused outputs (not offline analytics)

## System Architecture

```
Raw Ops Data
   ↓
Telemetry Stream
   ↓
Event-Driven Simulation Engine
   ↓
Trigger Detection
   ↓
Intervention Evaluation Engine
   ↓
Ranked Operator Actions
```

## Data Inputs

### 1. Flight / Task Events (`flight_events_table`)
- Represents scheduled and actual execution of tasks
- **Columns:** `event_id`, `asset_id` (TAIL_NUM), `scheduled_time`, `actual_time`, `event_type`, `origin`, `destination`
- Each row is converted into telemetry emissions

### 2. Asset Registry (`flight_assets_table`)
- Tracks reusable assets and baseline utilization
- **Columns:** `TAIL_NUM`, `avg_flights_per_day`, `avg_hours_per_day`, `total_flight_time`, `aircraft_model`, `aircraft_seats`
- Used to initialize asset availability and utilization tracking

### 3. Cleaned Operational Data (`flights_cleaned_us_only`)
- Source of truth for time alignment and delay attribution
- **Columns:** `scheduled_dep_utc`, `scheduled_arr_utc`, `actual_dep_utc`, `actual_arr_utc`, `dep_delay_minutes`, `arr_delay_minutes`, delay causes (weather, late aircraft, etc.)

### 4. Weather Telemetry (`flights_weather_table`)
- Environmental conditions injected as exogenous disruptions
- **Columns:** `timestamp`, `location`, `weather_severity_index`, `avg_weather_delay_min`, `max_weather_delay_min`

### 5. Dependency Table (`flights_dependency_table`)
- Explicit logical dependencies between events

## Core Design Principle

**Execution and causality are separated.**

- **Execution:** handled implicitly via asset availability and time ordering
- **Causality & leverage:** evaluated via counterfactual simulation

This allows the system to scale while still providing actionable operator insight.

## Step-by-Step Build Process

### STEP 1 — Build the Telemetry Stream

**Input:** `flight_events_table`

**Action:** For each event, generate telemetry emissions:
- Scheduled execution
- Actual execution
- Assign a timestamp to each emission

Sort globally by timestamp

**Output:**
```python
telemetry_stream = [
  {timestamp, event_id, asset_id, state, location}
]
```

This stream is the only input to the simulator.

### STEP 2 — Initialize Asset State

**Input:** `flight_assets_table`

**Action:** Create one state object per asset:

```python
asset_state = {
  asset_id: {
    "available_at": first_scheduled_time,
    "cumulative_delay": 0,
    "utilization": 0
  }
}
```

### STEP 3 — Event-Driven Simulation Engine

**Input:** `telemetry_stream`

**Logic:**
- Process telemetry in timestamp order
- For each event:
  - Check `asset.available_at`
  - If the asset is unavailable → delay event
  - Update asset availability
  - Record delay

**Key Point:** Delay propagation occurs implicitly
- No dependency graph traversal
- No global scheduling logic

### STEP 4 — Environmental Disruption Injection

**Input:** `flights_weather_table`

**Logic:** When an event is processed:
- Match on location + nearest timestamp
- If `weather_severity_index` exceeds threshold:
  - Inject stochastic delay sampled from weather stats

This models exogenous telemetry affecting system state.

### STEP 5 — Detect Intervention Triggers (Real Time)

During simulation, monitor for:
- Delay exceeding threshold
- Asset unavailability spike
- Weather severity jump

When detected, pause baseline execution and initiate evaluation.

### STEP 6 — Generate Candidate Interventions

Candidate actions are local and time-bounded, for example:
- Delay current event vs downstream event
- Swap asset (synthetic)
- Skip non-critical task
- Reorder next two tasks on the same asset

These are generated at runtime, not precomputed.

### STEP 7 — Counterfactual Rollout Evaluation

For each candidate intervention:
- Clone current simulation state
- Apply intervention
- Fast-forward simulation N hours
- Measure downstream impact

**Metrics:**
- Total future delay minutes
- Asset utilization loss
- Mission degradation score

### STEP 8 — Rank Highest-Leverage Interventions

Compute leverage as:

```
Leverage = Baseline Impact − Intervention Impact
```

Rank interventions by:
- Delay prevented
- Utilization recovered
- Mission score improvement

This produces operator-ready recommendations.

### STEP 9 — Causal Explanation (Optional, Post-Hoc)

**Input:**
- `flights_dependency_table`
- Simulation outcomes

Used to explain:
- Why an intervention worked
- Which dependency chains were broken
- Why a node is high-leverage

This supports trust and explainability, not execution.

## Outputs

- Ranked intervention list
- Timeline comparisons (baseline vs recovered)
- Asset availability plots
- Mission degradation metrics

All outputs are framed for operator decision support, not analytics.

## Why This Approach Works

- Scales to large systems (no global graph traversal)
- Mirrors real mission ops execution
- Supports real-time intervention
- Separates execution from explanation
- Aligns with Nominal-style mission operations thinking

## Example Operator Language

> "The simulator continuously replays mission telemetry and evaluates short-horizon counterfactuals to identify where operator action prevents the most downstream degradation."

## Technologies Used

- Python
- pandas / NumPy
- Event-driven simulation
- Counterfactual rollouts
- Time-aligned telemetry modeling

## Future Extensions

- Streaming telemetry input
- Operator UI
- Multi-asset branching dependencies
- Probabilistic risk envelopes

## Final Note

This project is intentionally **not** a schedule optimizer or historical analysis tool.

It is a mission operations simulator designed to answer:

**"Given what just happened, where should I intervene right now?"**
