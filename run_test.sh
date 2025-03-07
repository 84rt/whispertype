#!/bin/bash
# Run WhisperType test script with proper environment setup

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Check for required system dependencies
echo "Checking system dependencies..."

# Check if xdotool is installed
if ! command -v xdotool &> /dev/null; then
    echo "xdotool is not installed. Installing..."
    sudo apt-get install -y xdotool
fi

# Check if PortAudio is installed
if ! pkg-config --exists portaudio-2.0; then
    echo "PortAudio is not installed. Installing..."
    sudo apt-get install -y portaudio19-dev
fi

# Run the test script with sudo while preserving the environment
echo "Starting WhisperType Test..."
sudo -E "$(which python3)" "$SCRIPT_DIR/test_whispertype.py"
