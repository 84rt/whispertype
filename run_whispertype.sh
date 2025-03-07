#!/bin/bash
# Run WhisperType with proper environment setup

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

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY environment variable not set."
    echo "Transcription will not work without an API key."
    echo "You can set it with: export OPENAI_API_KEY='your-api-key'"
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

# Run the application with sudo while preserving the environment
echo "Starting WhisperType..."
sudo -E "$(which python3)" "$SCRIPT_DIR/whispertype.py"
