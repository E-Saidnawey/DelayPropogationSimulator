/**
 * Real-time Telemetry Streaming Backend
 * Streams flight telemetry updates via Socket.IO and enriches with ML predictions
 */

const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const axios = require('axios');
const fs = require('fs');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

// Configuration
const PORT = process.env.PORT || 3001;
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';
const REPLAY_SPEED = parseInt(process.env.REPLAY_SPEED) || 60; // Simulated minutes per real second
const TELEMETRY_FILE = process.env.TELEMETRY_FILE || './telemetry_stream.json';

// State
let telemetryData = [];
let currentIndex = 0;
let isStreaming = false;
let startTime = null;
let simulatedStartTime = null;
let activeFlights = new Map(); // flight_number -> flight data

/**
 * Load telemetry data from JSON file
 */
function loadTelemetryData() {
  try {
    const data = fs.readFileSync(TELEMETRY_FILE, 'utf8');
    telemetryData = JSON.parse(data);
    console.log(`✓ Loaded ${telemetryData.length} telemetry updates`);
    
    if (telemetryData.length > 0) {
      simulatedStartTime = new Date(telemetryData[0].update_time);
      console.log(`✓ Simulation starts at ${simulatedStartTime.toISOString()}`);
    }
    
    return true;
  } catch (error) {
    console.error(`✗ Error loading telemetry data: ${error.message}`);
    return false;
  }
}

/**
 * Get ML cascade prediction for a delayed flight
 */
async function getCascadePrediction(flightUpdate) {
  // Only get predictions for delayed flights
  if (flightUpdate.current_delay_minutes <= 0) {
    return null;
  }
  
  try {
    const response = await axios.post(`${ML_SERVICE_URL}/predict`, flightUpdate, {
      timeout: 1000 // 1 second timeout
    });
    return response.data;
  } catch (error) {
    console.error(`ML prediction error for flight ${flightUpdate.flight_number}: ${error.message}`);
    return null;
  }
}

/**
 * Process and enrich a telemetry update
 */
async function processTelemetryUpdate(update) {
  const enrichedUpdate = { ...update };
  
  // Get ML prediction if flight is delayed
  if (update.current_delay_minutes > 0) {
    const prediction = await getCascadePrediction(update);
    if (prediction) {
      enrichedUpdate.cascade_probability = prediction.cascade_probability;
      enrichedUpdate.risk_level = prediction.risk_level;
    }
  }
  
  // Update active flights map
  const key = `${update.carrier}${update.flight_number}`;
  activeFlights.set(key, enrichedUpdate);
  
  return enrichedUpdate;
}

/**
 * Calculate which updates should be sent based on elapsed real time
 */
function getUpdatesBatch() {
  if (!isStreaming || currentIndex >= telemetryData.length) {
    return [];
  }
  
  const now = Date.now();
  const elapsedRealSeconds = (now - startTime) / 1000;
  const elapsedSimulatedMinutes = elapsedRealSeconds * REPLAY_SPEED;
  const currentSimulatedTime = new Date(simulatedStartTime.getTime() + elapsedSimulatedMinutes * 60 * 1000);
  
  const batch = [];
  
  // Collect all updates up to current simulated time
  while (currentIndex < telemetryData.length) {
    const update = telemetryData[currentIndex];
    const updateTime = new Date(update.update_time);
    
    if (updateTime <= currentSimulatedTime) {
      batch.push(update);
      currentIndex++;
    } else {
      break;
    }
  }
  
  return batch;
}

/**
 * Main streaming loop
 */
async function streamingLoop() {
  if (!isStreaming) return;
  
  const batch = getUpdatesBatch();
  
  if (batch.length > 0) {
    console.log(`📡 Sending batch of ${batch.length} updates (${currentIndex}/${telemetryData.length})`);
    
    // Process and send each update
    for (const update of batch) {
      try {
        const enrichedUpdate = await processTelemetryUpdate(update);
        io.emit('telemetry_update', enrichedUpdate);
      } catch (error) {
        console.error(`Error processing update: ${error.message}`);
      }
    }
    
    // Send active flights summary
    const activeFlightsList = Array.from(activeFlights.values());
    io.emit('active_flights', {
      count: activeFlightsList.length,
      flights: activeFlightsList
    });
  }
  
  // Check if streaming is complete
  if (currentIndex >= telemetryData.length) {
    console.log('✓ Streaming complete - restarting from beginning');
    
    // Auto-restart: reset index and clear active flights
    currentIndex = 0;
    activeFlights.clear();
    startTime = Date.now(); // Reset start time for replay timing
    
    // Notify clients of restart
    io.emit('stream_restart', { 
      message: 'Stream completed, restarting from beginning',
      total_updates: telemetryData.length 
    });
    
    // Continue streaming (don't set isStreaming = false)
  }
  
  // Schedule next iteration
  setTimeout(streamingLoop, 100); // Check every 100ms
}

/**
 * Socket.IO connection handler
 */
io.on('connection', (socket) => {
  console.log(`✓ Client connected: ${socket.id}`);
  
  // Send current active flights
  socket.emit('active_flights', {
    count: activeFlights.size,
    flights: Array.from(activeFlights.values())
  });
  
  // Send streaming status
  socket.emit('stream_status', {
    is_streaming: isStreaming,
    current_index: currentIndex,
    total_updates: telemetryData.length,
    replay_speed: REPLAY_SPEED
  });
  
  socket.on('disconnect', () => {
    console.log(`✗ Client disconnected: ${socket.id}`);
  });
  
  // Control commands
  socket.on('start_stream', () => {
    if (!isStreaming && currentIndex < telemetryData.length) {
      console.log('▶ Starting stream');
      isStreaming = true;
      startTime = Date.now();
      streamingLoop();
    }
  });
  
  socket.on('pause_stream', () => {
    console.log('⏸ Pausing stream');
    isStreaming = false;
  });
  
  socket.on('reset_stream', () => {
    console.log('⏮ Resetting stream');
    isStreaming = false;
    currentIndex = 0;
    activeFlights.clear();
    socket.emit('stream_reset');
  });
});

// Express middleware
app.use(express.json());

// API endpoints
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    telemetry_loaded: telemetryData.length > 0,
    total_updates: telemetryData.length,
    current_index: currentIndex,
    is_streaming: isStreaming,
    active_flights: activeFlights.size
  });
});

app.get('/api/stats', (req, res) => {
  const carriers = new Set();
  const delayed = telemetryData.filter(u => u.current_delay_minutes > 15);
  
  telemetryData.forEach(u => carriers.add(u.carrier));
  
  res.json({
    total_updates: telemetryData.length,
    unique_carriers: carriers.size,
    delayed_flights: delayed.length,
    current_active: activeFlights.size,
    replay_speed: REPLAY_SPEED
  });
});

// Start server
async function startServer() {
  console.log('\n='.repeat(60));
  console.log('FLIGHT TELEMETRY STREAMING BACKEND');
  console.log('='.repeat(60));
  
  // Load telemetry data
  if (!loadTelemetryData()) {
    console.error('Failed to load telemetry data. Exiting.');
    process.exit(1);
  }
  
  // Check ML service
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/health`, { timeout: 2000 });
    console.log(`✓ ML Service connected (${response.data.status})`);
  } catch (error) {
    console.log(`⚠ Warning: ML Service not available at ${ML_SERVICE_URL}`);
    console.log('  Predictions will be skipped');
  }
  
  // Start server
  server.listen(PORT, () => {
    console.log(`\n✓ Server running on port ${PORT}`);
    console.log(`✓ Socket.IO endpoint: ws://localhost:${PORT}`);
    console.log(`✓ Replay speed: ${REPLAY_SPEED}x (${REPLAY_SPEED} simulated minutes per real second)`);
    console.log('\nReady to stream telemetry data!');
    console.log('Connect a client and send "start_stream" event to begin.\n');
  });
}

startServer();