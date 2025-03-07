# WhisperType

WhisperType is a voice typing application for Ubuntu Linux that enables dictation in any application using a global keyboard shortcut. It records audio from your microphone, transcribes it using OpenAI's Whisper API, and types the transcribed text at your cursor location.

## Features

- **Global Hotkey**: Use Super/Windows + Caps Lock to start and stop recording
- **Audio Recording**: Captures microphone input with minimal latency
- **Automatic Silence Detection**: Stops recording after detecting silence
- **OpenAI Whisper Integration**: High-quality speech-to-text transcription
- **Seamless Typing**: Inserts transcribed text at your cursor location
- **Audio Feedback**: Plays sounds when recording starts and stops

## Requirements

- Ubuntu Linux (or other Linux distributions)
- Python 3.8+
- xdotool (for simulating keystrokes)
- OpenAI API Key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/whispertype.git
   cd whispertype
   ```

2. Create a virtual environment and install dependencies:
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Install xdotool (if not already installed):
   ```
   sudo apt-get install xdotool
   ```

4. Set your OpenAI API key:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```
   
   For persistent usage, add this line to your `~/.bashrc` or `~/.zshrc` file.

## Usage

1. Run the application with root privileges (required for global hotkey capture):
   ```
   sudo -E python3 whispertype.py
   ```
   
   The `-E` flag preserves your environment variables, including the OpenAI API key.

2. Press **Super/Windows + Caps Lock** to start recording.
3. Speak clearly into your microphone.
4. Press **Super/Windows + Caps Lock** again to stop recording, or let the automatic silence detection stop it for you.
5. Wait for the transcription to complete, and the text will be typed at your cursor location.
6. Press **Ctrl+C** in the terminal to exit the application.

## Configuration

You can modify the following settings in the `Config` class within `whispertype.py`:

- `SAMPLE_RATE`: Audio sample rate (default: 16000 Hz)
- `HOTKEY`: Keyboard shortcut to start/stop recording (default: "meta+caps lock")
- `SILENCE_THRESHOLD`: Amplitude threshold for silence detection
- `SILENCE_DURATION`: Seconds of silence to auto-stop recording
- `MAX_RECORDING_TIME`: Maximum recording time in seconds

## Troubleshooting

- **Hotkey not working**: Make sure you're running the application with `sudo`.
- **No transcription**: Check that your OpenAI API key is set correctly.
- **Audio not recording**: Verify your microphone is working and properly configured.
- **Text not typing**: Ensure xdotool is installed and working properly.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [OpenAI Whisper](https://openai.com/research/whisper) for speech recognition
- [keyboard](https://github.com/boppreh/keyboard) for global hotkey support
- [sounddevice](https://github.com/spatialaudio/python-sounddevice) for audio recording
- [xdotool](https://github.com/jordansissel/xdotool) for simulating keystrokes