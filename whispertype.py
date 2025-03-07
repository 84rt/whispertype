#!/usr/bin/env python3
"""
WhisperType - Voice Typing Application for Ubuntu Linux

This application enables voice typing using a global keyboard shortcut (Meta + Caps Lock).
It records audio from the microphone, transcribes it using OpenAI's Whisper API,
and simulates keystrokes to insert the transcribed text at the cursor location.
"""

import os
import sys
import time
import tempfile
import subprocess
import threading
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import keyboard
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import requests
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("WhisperType")

# Configuration
class Config:
    # Audio recording settings
    SAMPLE_RATE = 16000  # Hz
    CHANNELS = 1  # Mono
    DTYPE = 'float32'
    
    # Recording state
    MAX_RECORDING_TIME = 60  # seconds
    SILENCE_THRESHOLD = 0.01  # Amplitude threshold for silence detection
    SILENCE_DURATION = 1.5  # seconds of silence to auto-stop
    
    # Hotkey configuration
    DEFAULT_HOTKEY = "windows+caps lock"  # Windows/Super key
    HOTKEY = DEFAULT_HOTKEY  # Will be overridden if config file exists
    
    # Feedback settings
    USE_BEEP = True  # Use simple beep for feedback
    BEEP_FREQUENCY_START = 440  # Hz (A4 note)
    BEEP_FREQUENCY_STOP = 880  # Hz (A5 note)
    BEEP_DURATION = 0.2  # seconds
    
    # OpenAI API settings
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    
    # Temporary file for audio
    TEMP_DIR = tempfile.gettempdir()


class WhisperType:
    def __init__(self):
        self.recording = False
        self.audio_data = []
        self.recording_thread = None
        self.silence_counter = 0
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Load configuration if it exists
        self.load_config()
        
        # Check if OpenAI API key is set
        if not Config.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY environment variable not set. Transcription will not work.")
            print("Warning: OPENAI_API_KEY environment variable not set.")
            print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        
        # Register hotkey
        keyboard.add_hotkey(Config.HOTKEY, self.toggle_recording)
        
        logger.info("WhisperType initialized. Press %s to start/stop recording.", Config.HOTKEY)
        print(f"WhisperType initialized. Press {Config.HOTKEY} to start/stop recording.")
    
    def load_config(self):
        """Load configuration from config.json if it exists."""
        config_path = Path(__file__).parent / 'config.json'
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Update hotkey if it exists in the config
                if 'hotkey' in config and config['hotkey']:
                    Config.HOTKEY = config['hotkey']
                    logger.info("Loaded custom hotkey from config: %s", Config.HOTKEY)
            except Exception as e:
                logger.error("Failed to load config: %s", e)
                print(f"Error loading configuration: {e}")
                print(f"Using default hotkey: {Config.DEFAULT_HOTKEY}")

    def play_beep(self, frequency: float, duration: float) -> None:
        """Play a beep sound for feedback using numpy and sounddevice."""
        try:
            if Config.USE_BEEP:
                # Generate a sine wave for the beep
                t = np.linspace(0, duration, int(duration * Config.SAMPLE_RATE), False)
                beep = 0.5 * np.sin(2 * np.pi * frequency * t)
                
                # Play the beep
                sd.play(beep, Config.SAMPLE_RATE)
                sd.wait()
        except Exception as e:
            logger.error("Failed to play beep: %s", e)

    def toggle_recording(self) -> None:
        """Toggle recording state when hotkey is pressed."""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self) -> None:
        """Start recording audio from microphone."""
        if self.recording:
            return
        
        self.recording = True
        self.audio_data = []
        self.silence_counter = 0
        
        # Play feedback sound
        self.play_beep(Config.BEEP_FREQUENCY_START, Config.BEEP_DURATION)
        
        logger.info("Recording started...")
        print("Recording started... (Press %s again to stop)", Config.HOTKEY)
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()

    def _record_audio(self) -> None:
        """Record audio from microphone in a separate thread."""
        try:
            with sd.InputStream(
                samplerate=Config.SAMPLE_RATE,
                channels=Config.CHANNELS,
                dtype=Config.DTYPE,
                callback=self._audio_callback
            ):
                # Keep the stream open until recording is stopped
                start_time = time.time()
                while self.recording and (time.time() - start_time) < Config.MAX_RECORDING_TIME:
                    time.sleep(0.1)
                
                # If we reached max recording time, stop recording
                if self.recording and (time.time() - start_time) >= Config.MAX_RECORDING_TIME:
                    logger.info("Maximum recording time reached.")
                    self.stop_recording()
        
        except Exception as e:
            logger.error("Error during recording: %s", e)
            self.recording = False
            print(f"Error during recording: {e}")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Dict[str, Any], status: Any) -> None:
        """Callback function for audio stream."""
        if status:
            logger.warning("Audio callback status: %s", status)
        
        # Add the recorded audio to our buffer
        self.audio_data.append(indata.copy())
        
        # Check for silence to auto-stop recording
        if np.max(np.abs(indata)) < Config.SILENCE_THRESHOLD:
            self.silence_counter += frames / Config.SAMPLE_RATE
            if self.silence_counter >= Config.SILENCE_DURATION:
                logger.info("Silence detected, stopping recording.")
                self.recording = False
        else:
            self.silence_counter = 0

    def stop_recording(self) -> None:
        """Stop recording and process the audio."""
        if not self.recording:
            return
        
        self.recording = False
        logger.info("Recording stopped.")
        print("Recording stopped. Transcribing...")
        
        # Play feedback sound
        self.play_beep(Config.BEEP_FREQUENCY_STOP, Config.BEEP_DURATION)
        
        # Wait for recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1.0)
        
        # Process the recorded audio
        if self.audio_data:
            threading.Thread(target=self._process_audio).start()
        else:
            logger.warning("No audio data recorded.")
            print("No audio data recorded.")

    def _process_audio(self) -> None:
        """Process recorded audio and get transcription."""
        try:
            # Convert audio data to a single numpy array
            audio_data = np.concatenate(self.audio_data, axis=0)
            
            # Normalize audio
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Save as WAV file
            temp_file = os.path.join(Config.TEMP_DIR, "whispertype_recording.wav")
            wav.write(temp_file, Config.SAMPLE_RATE, audio_data)
            
            logger.info("Audio saved to temporary file: %s", temp_file)
            
            # Get transcription
            transcription = self._transcribe_audio(temp_file)
            
            if transcription:
                # Type the transcribed text
                self._type_text(transcription)
                logger.info("Transcription complete: %s", transcription)
                print(f"Transcribed: {transcription}")
            else:
                logger.warning("Transcription failed or returned empty result.")
                print("Transcription failed or returned empty result.")
            
            # Clean up temporary file
            try:
                os.remove(temp_file)
            except Exception as e:
                logger.warning("Failed to remove temporary file: %s", e)
        
        except Exception as e:
            logger.error("Error processing audio: %s", e)
            print(f"Error processing audio: {e}")

    def _transcribe_audio(self, audio_file: str) -> Optional[str]:
        """Transcribe audio using OpenAI's Whisper API."""
        if not Config.OPENAI_API_KEY:
            logger.error("OpenAI API key not set. Cannot transcribe audio.")
            print("Error: OpenAI API key not set. Cannot transcribe audio.")
            return None
        
        try:
            logger.info("Sending audio to OpenAI Whisper API...")
            
            with open(audio_file, "rb") as audio:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio
                )
            
            return response.text
        
        except Exception as e:
            logger.error("Error transcribing audio: %s", e)
            print(f"Error transcribing audio: {e}")
            return None

    def _type_text(self, text: str) -> None:
        """Type the transcribed text using xdotool."""
        try:
            # Escape special characters for shell
            escaped_text = text.replace('"', '\\"')
            
            # Use xdotool to type the text
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", escaped_text],
                check=True
            )
            
            logger.info("Text typed successfully.")
        
        except Exception as e:
            logger.error("Error typing text: %s", e)
            print(f"Error typing text: {e}")

    def run(self) -> None:
        """Run the application in the background."""
        print(f"WhisperType is running. Press {Config.HOTKEY} to start/stop recording.")
        print("Press Ctrl+C to exit.")
        
        try:
            # Keep the application running
            keyboard.wait()
        except KeyboardInterrupt:
            print("\nExiting WhisperType...")
        finally:
            keyboard.unhook_all()


if __name__ == "__main__":
    print("Starting WhisperType...")
    print("This application requires root privileges to capture global keyboard events.")
    print("If you encounter permission issues, try running with sudo.")
    
    # Check if running as root (required for keyboard module on Linux)
    if os.geteuid() != 0:
        print("Warning: Not running as root. Global hotkeys may not work properly.")
        print("Consider running with: sudo python3 whispertype.py")
    
    app = WhisperType()
    app.run()
