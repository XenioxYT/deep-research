#!/bin/bash

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Function to cleanup background processes
cleanup() {
    echo "Cleaning up..."
    # Kill any processes using our ports
    lsof -t -i:8993 | xargs -r kill
    lsof -t -i:8992 | xargs -r kill
    # Kill any remaining background processes
    kill $(jobs -p) 2>/dev/null
    exit 0
}

# Function to wait for a port to be available
wait_for_port() {
    local port=$1
    local timeout=30
    local count=0

    # First, check if port is in use and kill the process
    echo "Checking if port $port is in use..."
    lsof -t -i:$port | xargs -r kill
    sleep 2  # Give the process time to die

    echo "Waiting for port $port to be available..."
    while ! nc -z localhost $port; do
        sleep 1
        count=$((count + 1))
        if [ $count -ge $timeout ]; then
            echo "Timeout waiting for port $port"
            return 1
        fi
    done
    echo "Port $port is available"
    return 0
}

# Get the IP address
IP_ADDR=$(hostname -I | awk '{print $1}')
if [ -z "$IP_ADDR" ]; then
    IP_ADDR="localhost"
fi

# Set up cleanup on script termination
trap cleanup EXIT INT TERM

echo "Starting services..."

# Kill any existing processes on our ports
lsof -t -i:8993 | xargs -r kill
lsof -t -i:8992 | xargs -r kill
sleep 2  # Give processes time to die

# Start the backend server
cd "${SCRIPT_DIR}/backend"
echo "Installing backend dependencies..."
python3 -m pip install -r requirements.txt
playwright install

echo "Starting backend server..."
python3 run.py &
wait_for_port 8993
if [ $? -ne 0 ]; then
    echo "Failed to start backend server"
    exit 1
fi
echo "Backend server started on port 8993"

# Start the frontend server
cd "${SCRIPT_DIR}/frontend"
echo "Installing frontend dependencies..."
npm install

echo "Starting frontend server..."
npm run dev -- --host 0.0.0.0 --port 8992 &
wait_for_port 8992
if [ $? -ne 0 ]; then
    echo "Failed to start frontend server"
    exit 1
fi
echo "Frontend server started on port 8992"

echo "All services started successfully!"
echo "Open http://$IP_ADDR:8992 in your browser"

# Wait for both processes
wait 