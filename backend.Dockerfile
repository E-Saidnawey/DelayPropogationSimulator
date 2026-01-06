FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY backend/package*.json ./

# Copy telemetry json
COPY backend/telemetry_stream.json ./backend/telemetry_stream.json

# Copy airport mapping csv
COPY data/airport_mapping_complete.csv ./data/airport_mapping_complete.csv

# Install dependencies
RUN npm ci --only=production

# Copy server code
COPY backend/server.js ./

EXPOSE 8000

CMD ["node", "server.js"]