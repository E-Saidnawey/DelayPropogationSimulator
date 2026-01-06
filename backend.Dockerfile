FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy server code
COPY server.js ./
COPY src ./src

EXPOSE 8000

CMD ["node", "server.js"]