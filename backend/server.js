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
const readline = require('readline');

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
const TELEMETRY_FILE = process.env.TELEMETRY_FILE || './backend/telemetry_stream.json';
const AIRPORT_MAPPING_FILE = './data/airport_mapping_complete.csv';
let airportMap = new Map(); // Store ID -> Code

// State
let telemetryData = [];
let currentIndex = 0;
let isStreaming = false;
let startTime = null;
let simulatedStartTime = null;
let activeFlights = new Map(); // flight_number -> flight data

async function loadAirportMappings() {
  try {
    if (!fs.existsSync(AIRPORT_MAPPING_FILE)) {
      console.warn('⚠️ Airport mapping file not found. Using IDs.');
      return;
    }

    const fileStream = fs.createReadStream(AIRPORT_MAPPING_FILE);
    const rl = readline.createInterface({
      input: fileStream,
      crlfDelay: Infinity
    });

    let isHeader = true;
    for await (const line of rl) {
      if (isHeader) { isHeader = false; continue; }
      
      // Parse CSV line (assuming simple CSV without quoted commas for now)
      const cols = line.split(',');
      if (cols.length >= 3) {
        const id = cols[0].trim();       // AIRPORT_ID
        const code = cols[2].trim();     // ICAO_CODE (or IATA if available in your CSV)
        if (id && code) airportMap.set(id, code);
      }
    }
    console.log(`✓ Loaded ${airportMap.size} airport codes`);
  } catch (error) {
    console.error(`Error loading airports: ${error.message}`);
  }
}

/**
 * Load telemetry data from JSON file
 * Uses streaming for large files to avoid memory issues
 */
function loadTelemetryData() {
  try {
    // Check file size first
    const stats = fs.statSync(TELEMETRY_FILE);
    const fileSizeMB = stats.size / (1024 * 1024);
    
    console.log(`📊 Telemetry file size: ${fileSizeMB.toFixed(2)} MB`);
    
    if (fileSizeMB > 400) {
      console.log('⚠️  WARNING: Large file detected. This may take a moment...');
    }
    
    // Read file in chunks to handle large files
    const data = fs.readFileSync(TELEMETRY_FILE, 'utf8');
    
    console.log(`📖 Parsing JSON...`);
    telemetryData = JSON.parse(data);
    
    console.log(`✓ Loaded ${telemetryData.length} telemetry updates`);
    console.log(`✓ Memory usage: ~${(process.memoryUsage().heapUsed / 1024 / 1024).toFixed(2)} MB`);
    
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
  
  if (airportMap.has(update.origin)) {
    enrichedUpdate.origin = airportMap.get(update.origin);
  }
  if (airportMap.has(update.destination)) {
    enrichedUpdate.destination = airportMap.get(update.destination);
  }

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
  
  // Remove flight if it has arrived (status = 'arrived')
  if (update.status === 'arrived') {
    activeFlights.delete(key);
  } else {
    // Update active flights map for in-progress flights
    activeFlights.set(key, enrichedUpdate);
  }

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
  const elapsedSimulatedMinutes = elapsedRealSeconds * REPLAY_SPEED / 60;
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
  
  // Load airport mapping
  await loadAirportMappings();

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
    console.log('\n🚀 AUTO-STARTING STREAM...\n');
    
    // Auto-start streaming
    isStreaming = true;
    startTime = Date.now();
    simulatedStartTime = new Date(telemetryData[0].update_time);
    streamingLoop();
    
    console.log('✓ Stream is now running automatically!');
    console.log('  Clients will receive updates as soon as they connect.\n');
  });
}

startServer();