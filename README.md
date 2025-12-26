# DelayPropogationSimulator


# Flight Delay Propagation Simulation - Database Documentation

## Project Overview

This project transforms raw flight operation data into a structured database designed to simulate how delays ripple through an airline network. The simulation tracks how a single delayed flight can cascade through the system via shared aircraft resources.

---

## Database Schema

The database consists of four primary tables that work together to model operational dependencies and delay propagation:

### 1. Events Table (`events.csv`)

**Purpose**: Records every operational event (departure and arrival) in the system.

**Key Concept**: Each flight generates TWO events:
- One DEPARTURE event at the origin airport
- One ARRIVAL event at the destination airport

#### Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `event_id` | String | Unique identifier for this event | EVT_00001234 |
| `task_id` | String | Flight number (same for dep & arr of same flight) | 1245 |
| `asset_id` | String | Aircraft tail number | N12345 |
| `scheduled_start_time` | Datetime | When event was scheduled to occur | 2024-01-15 14:30:00 |
| `scheduled_end_time` | Datetime | When event was scheduled to end (same as start for instantaneous events) | 2024-01-15 14:30:00 |
| `actual_start_time` | Datetime | When event actually occurred | 2024-01-15 14:47:00 |
| `actual_end_time` | Datetime | When event actually ended | 2024-01-15 14:47:00 |
| `event_type` | String | Type of event | DEPARTURE or ARRIVAL |
| `location` | String | Airport code where event occurs | ORD (origin for dep, dest for arr) |
| `delay_minutes` | Integer | Actual - Scheduled time in minutes | 17 |
| `flight_date` | Date | Date of the flight | 2024-01-15 |
| `carrier` | String | Operating airline code | UA, AA, DL, etc. |
| `flight_number` | String | Flight number | 1245 |
| `origin` | String | Origin airport code | ORD |
| `destination` | String | Destination airport code | LAX |

---

### 2. Assets Table (`assets.csv`)

**Purpose**: Registry of all aircraft (resources) in the system, each identified by tail number.

**Key Concept**: One row per unique aircraft. This table describes the RESOURCE, not its activities.

#### Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `asset_id` | String | Aircraft tail number (unique identifier)
| `asset_type` | String | Airline/carrier operating this aircraft
| `home_base` | String | Primary hub (most frequent origin airport) 
| `max_utilization_per_day` | Float | Average number of flights per day
| `avg_flight_hours_per_day` | Float | Average hours in air per day
| `total_flights_observed` | Integer | Total flights in observation period
| `observation_days` | Integer | Number of days observed

---

### 3. Dependencies Table (`dependencies.csv`)

**Purpose**: Defines sequencing constraints between events—specifically when events must occur in a particular order.

**Key Concept**: When the same aircraft operates consecutive flights, the arrival of Flight N **must complete** before the departure of Flight N+1 can begin. This is the PRIMARY mechanism for delay propagation.

#### Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `dependency_id` | String | Unique identifier for this dependency | DEP_00000123 |
| `upstream_event_id` | String | Event that must complete first | EVT_00000002 (arrival) |
| `downstream_event_id` | String | Event that depends on upstream | EVT_00000003 (next departure) |
| `upstream_task_id` | String | Flight number of upstream flight | 1245 |
| `downstream_task_id` | String | Flight number of downstream flight | 1832 |
| `asset_id` | String | Aircraft tail number linking the flights | N12345 |
| `min_separation_minutes` | Integer | Minimum required turnaround time | 45 |
| `scheduled_separation_minutes` | Integer | Actual scheduled buffer time | 65 |
| `upstream_location` | String | Airport where upstream event ends | LAX |
| `downstream_location` | String | Airport where downstream event begins | LAX |
| `turnaround_airport` | String | Airport where turnaround occurs | LAX |
| `dependency_type` | String | Type of dependency | AIRCRAFT_TURNAROUND |

---

### 4. Environmental Data Table (`environment.csv`)

**Purpose**: Captures external factors that can cause delays (primarily weather).

**Key Concept**: Environmental conditions affect all operations at a given location during a given time period.

#### Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `timestamp` | Date/Datetime | Date or time period | 2024-01-15 |
| `location` | String | Airport code | ORD |
| `weather_severity_index` | Float | Severity score 0-10 (0=clear, 10=severe) | 7.2 |
| `flights_affected` | Integer | Number of flights at this location/time | 245 |
| `cancellations` | Integer | Weather-related cancellations | 12 |
| `avg_delay` | Float | Average departure delay in minutes | 34.5 |
| `diversions` | Integer | Number of diverted flights | 3 |
| `data_source` | String | Source of weather data | PROXY_FROM_OPERATIONS |

#### Weather Severity Calculation (Proxy Method)

When actual weather data is unavailable, calculate severity from operational impacts:

```
severity = (cancellation_rate × 5) + (normalized_delay × 3) + (diversion_rate × 2)

Where:
- cancellation_rate = weather_cancellations / total_flights
- normalized_delay = min(avg_delay, 60) / 60
- diversion_rate = diversions / total_flights
```

---

## Database Relationships

### Entity-Relationship Diagram (Conceptual)

```
┌─────────────┐
│   ASSETS    │
│ (Aircraft)  │
└──────┬──────┘
       │ 1
       │
       │ operates
       │
       │ N
┌──────▼──────┐         ┌──────────────┐
│   EVENTS    ├────────►│ ENVIRONMENT  │
│(Dep & Arr)  │ N     1 │  (Weather)   │
└──────┬──────┘ occurs  └──────────────┘
       │        at
       │
       │ linked by
       │
┌──────▼──────┐
│DEPENDENCIES │
│(Turnarounds)│
└─────────────┘
```

### Relationship Details

1. **Assets → Events**: One-to-Many
   - One aircraft operates many flights (events)
   - `assets.asset_id` → `events.asset_id`

2. **Events → Dependencies**: Many-to-Many (via upstream/downstream)
   - An arrival event can be upstream for a departure event
   - A departure event can be downstream from an arrival event
   - `events.event_id` → `dependencies.upstream_event_id`
   - `events.event_id` → `dependencies.downstream_event_id`

3. **Events → Environment**: Many-to-One
   - Many events occur at one location on one day
   - Joined on: `events.location` = `environment.location` AND `events.flight_date` = `environment.timestamp`

---

## Data Flow for Delay Simulation

### Step-by-Step Simulation Logic

#### 1. **Initialize State**
```
FOR each event in events table:
    IF event is a DEPARTURE:
        actual_departure_time = scheduled_start_time + delay_minutes
    IF event is an ARRIVAL:
        actual_arrival_time = scheduled_start_time + delay_minutes
```

#### 2. **Inject Initial Delays**
Use actual delays from historical data, or inject hypothetical delays:
```
# Historical replay
delays = events.delay_minutes (from actual data)

# Hypothetical scenario
Set Flight X departure delay = 120 minutes
Set all other events delay = 0
```

#### 3. **Propagate Delays Through Dependencies**
```
FOR each dependency in dependencies table (ordered by time):
    upstream_event = get_event(dependency.upstream_event_id)
    downstream_event = get_event(dependency.downstream_event_id)
    
    # Calculate actual separation
    actual_separation = (
        downstream_event.scheduled_start_time - 
        upstream_event.actual_end_time
    )
    
    # Check if delay propagates
    IF actual_separation < dependency.min_separation_minutes:
        # Downstream must be delayed
        required_delay = (
            dependency.min_separation_minutes - actual_separation
        )
        downstream_event.delay_minutes += required_delay
        downstream_event.actual_start_time += required_delay
        
        # Arrival inherits departure delay (with potential recovery)
        IF downstream_event.type == DEPARTURE:
            corresponding_arrival = get_arrival_for_flight(downstream_event.task_id)
            corresponding_arrival.delay_minutes += (required_delay * recovery_factor)
```

#### 4. **Account for In-Flight Recovery**
```
recovery_factor = 0.7  # Crews can make up ~30% of delay in flight

FOR each flight:
    arrival_delay = departure_delay × recovery_factor
```

#### 5. **Measure Impact**
```
total_delay_minutes = SUM(all_events.delay_minutes)
flights_affected = COUNT(events WHERE delay_minutes > 0)
max_cascade_depth = longest dependency chain triggered
```

---

## Example Delay Cascade Scenario

### Initial State
```
Flight 100 (ORD→LAX): Scheduled Dep 08:00, Arr 10:30
Flight 200 (LAX→SFO): Scheduled Dep 11:30, Arr 12:45
Flight 300 (SFO→SEA): Scheduled Dep 13:45, Arr 15:00

All use aircraft N12345
Turnarounds: 60 min minimum
```

### Scenario: Weather Delay at ORD

**T=08:00** - Flight 100 scheduled to depart
- **Weather severity at ORD = 8.5**
- **INJECT: Flight 100 departure delayed 90 minutes**

**T=09:30** - Flight 100 actually departs
- Arrival originally scheduled: 10:30
- New estimated arrival: 12:00 (90 min delay, assume no recovery)

**Dependency Check #1**: Flight 100 → Flight 200
- Flight 200 needs to depart: 11:30
- Flight 100 now arrives: 12:00
- Separation: 12:00 - 11:30 = -30 minutes (VIOLATION!)
- **PROPAGATE: Flight 200 delayed 30 minutes** → New departure: 12:00

**T=12:00** - Flight 200 departs (30 min late)
- Scheduled arrival SFO: 12:45
- New arrival: 13:15 (assuming 30 min delay persists)

**Dependency Check #2**: Flight 200 → Flight 300
- Flight 300 needs to depart: 13:45
- Flight 200 now arrives: 13:15
- Separation: 13:45 - 13:15 = 30 minutes
- Minimum required: 60 minutes
- **PROPAGATE: Flight 300 delayed 30 minutes** → New departure: 14:15

### Impact Summary
```
Initial delay: 90 minutes (1 flight)
Cascaded delays: 60 minutes (2 flights)
Total system delay: 150 minutes
Delay ratio: 1.67x (initial delay amplified by 67%)
Flights affected: 3
```

---

## Query Examples

### Find All Flights for a Specific Aircraft
```sql
SELECT * FROM events
WHERE asset_id = 'N12345'
ORDER BY scheduled_start_time
```

### Find Tight Turnarounds (High Risk)
```sql
SELECT * FROM dependencies
WHERE scheduled_separation_minutes < min_separation_minutes + 15
ORDER BY scheduled_separation_minutes
```

### Calculate Average Delay by Airport
```sql
SELECT 
    location,
    AVG(delay_minutes) as avg_delay,
    COUNT(*) as total_events
FROM events
WHERE event_type = 'DEPARTURE'
GROUP BY location
ORDER BY avg_delay DESC
```

### Find Dependency Chains (Aircraft Routes)
```sql
SELECT 
    d.asset_id,
    d.upstream_task_id as flight_1,
    d.downstream_task_id as flight_2,
    d.scheduled_separation_minutes,
    e1.location as turnaround_airport
FROM dependencies d
JOIN events e1 ON d.upstream_event_id = e1.event_id
ORDER BY d.asset_id, e1.scheduled_start_time
```

### High-Utilization Aircraft (Most Vulnerable to Cascades)
```sql
SELECT 
    asset_id,
    max_utilization_per_day,
    total_flights_observed
FROM assets
WHERE max_utilization_per_day > 6.0
ORDER BY max_utilization_per_day DESC
```

---

## Simulation Parameters & Tuning

### Key Parameters to Adjust

1. **Minimum Turnaround Time** (`min_separation_minutes`)
   - Short-haul domestic: 30-45 min
   - Medium-haul: 45-60 min
   - Long-haul/international: 60-90 min
   - Adjust based on aircraft type or historical data

2. **Recovery Factor** (in-flight time savings)
   - Typical: 0.5 - 0.8
   - Higher = more aggressive flying to make up time
   - 0.7 = 30% delay recovery in air

3. **Weather Impact Threshold**
   - Severity > 7.0: All flights delayed 15-45 min
   - Severity > 9.0: Some cancellations
   - Customize based on airport capabilities

4. **Time Window**
   - Minimum: 1 week (captures weekly patterns)
   - Recommended: 2-4 weeks (good for testing)
   - Full analysis: 1-3 months

---

## Data Quality Checks

### Before Running Simulation

✅ **Check 1**: No missing asset IDs in events
```sql
SELECT COUNT(*) FROM events WHERE asset_id IS NULL
-- Should be 0
```

✅ **Check 2**: All dependencies reference valid events
```sql
SELECT COUNT(*) FROM dependencies d
LEFT JOIN events e1 ON d.upstream_event_id = e1.event_id
WHERE e1.event_id IS NULL
-- Should be 0
```

✅ **Check 3**: Events in chronological pairs
```sql
-- For each flight, departure should come before arrival
SELECT task_id, COUNT(*) FROM (
    SELECT task_id, 
           MAX(CASE WHEN event_type='DEPARTURE' THEN scheduled_start_time END) as dep_time,
           MAX(CASE WHEN event_type='ARRIVAL' THEN scheduled_start_time END) as arr_time
    FROM events
    GROUP BY task_id
    HAVING dep_time > arr_time
)
-- Should be 0
```

✅ **Check 4**: Turnaround locations match
```sql
SELECT * FROM dependencies d
JOIN events e1 ON d.upstream_event_id = e1.event_id
JOIN events e2 ON d.downstream_event_id = e2.event_id
WHERE e1.location != e2.location
-- Should be 0 (aircraft can't teleport)
```

---

## File Naming & Organization

### Recommended Directory Structure
```
flight_delay_simulation/
│
├── data/
│   ├── raw/
│   │   └── flight_data_raw.csv          # Original input data
│   │
│   ├── processed/
│   │   ├── events.csv                   # Generated events table
│   │   ├── assets.csv                   # Generated assets table
│   │   ├── dependencies.csv             # Generated dependencies table
│   │   └── environment.csv              # Generated environment table
│   │
│   └── validation/
│       └── validation_report.json       # Data quality report
│
├── scripts/
│   ├── 01_clean_data.py                 # Data cleaning
│   ├── 02_generate_events.py           # Events table creation
│   ├── 03_generate_assets.py           # Assets table creation
│   ├── 04_generate_dependencies.py     # Dependencies creation
│   ├── 05_generate_environment.py      # Environment data
│   └── 06_run_simulation.py            # Delay propagation sim
│
├── docs/
│   ├── README.md                        # This file
│   └── pseudocode.md                    # Implementation guide
│
└── output/
    ├── simulation_results.csv           # Delay cascade results
    └── visualization/
        ├── delay_heatmap.png
        └── network_graph.png
```

---

## Next Steps

1. **Implement the pseudocode** in your language of choice (Python, R, SQL)
2. **Run data validation** to ensure quality
3. **Start with small test** (1 week, single carrier)
4. **Build visualization** to show delay cascades
5. **Scale up** to full dataset
6. **Add complexity**: crew constraints, gate limits, passenger connections

---

## Common Issues & Troubleshooting

### Issue: Too Many Dependencies Created
**Symptom**: Dependencies table is huge, many unrealistic links  
**Solution**: Add stricter time window filter (max 24-hour gap between flights)

### Issue: Aircraft Appears in Two Places Simultaneously
**Symptom**: Arrival at JFK at 3pm, Departure from LAX at 3:05pm  
**Solution**: Data quality issue—filter out flights with impossible turnarounds in cleaning phase

### Issue: Delays Don't Propagate
**Symptom**: Initial delay doesn't cascade  
**Solution**: Check that dependency chain is correct and min_separation < scheduled_separation

### Issue: All Flights Show Zero Delay
**Symptom**: No delays in events table  
**Solution**: Check that DEP_DELAY and ARR_DELAY fields were properly mapped from source data

---

## Glossary

- **Asset**: An aircraft (tail number) that operates flights
- **Event**: A single operational occurrence (departure or arrival)
- **Task**: A flight (consists of 2 events: departure + arrival)
- **Dependency**: A constraint requiring one event to complete before another begins
- **Turnaround**: Ground time between arrival and next departure for same aircraft
- **Propagation**: When a delay causes subsequent delays through dependency chains
- **Recovery**: Making up delay time during flight operations
- **Severity Index**: 0-10 scale rating weather/environmental impact

---

## Contact & Support

For questions about this database structure or simulation approach, refer to:
- Original flight data source documentation
- Bureau of Transportation Statistics (BTS) data dictionaries
- Airline operations management resources

---

**Version**: 1.0  
**Last Updated**: December 2024  
**Author**: Flight Delay Simulation Project
