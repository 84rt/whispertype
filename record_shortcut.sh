#!/bin/bash
# Run the shortcut recorder with proper environment setup

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

# Make the script executable
chmod +x "$SCRIPT_DIR/record_shortcut.py"

# Run the recorder with sudo while preserving the environment
echo "Starting Shortcut Recorder..."
echo "Press the keys you want to use as your shortcut, then press Enter to save."
sudo -E "$(which python3)" "$SCRIPT_DIR/record_shortcut.py"

echo ""
echo "After recording your shortcut, run ./run_whispertype.sh to start the application"
echo "with your custom shortcut."
