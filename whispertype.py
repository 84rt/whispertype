#!/usr/bin/env python3
"""
WhisperType - Voice Typing Application for Ubuntu Linux

This application enables voice typing using a global keyboard shortcut (Meta + Caps Lock).
It records audio from the microphone, transcribes it using OpenAI's Whisper API,
and simulates keystrokes to insert the transcribed text at the cursor location.

Usage:
  whispertype.py         # Normal mode: stops recording after silence
  whispertype.py -f      # Flowing mode: continuously records and transcribes after silence
"""

import os
import sys
import time
import tempfile
import subprocess
import threading
import json
import logging
import argparse
from typing import Optional, Dict, Any
from pathlib import Path

import keyboard
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import requests
from openai import OpenAI
from scipy.signal import resample

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
    RECORDING_SAMPLE_RATE = 44100  # Hz (system default)
    TARGET_SAMPLE_RATE = 16000  # Hz (required by Whisper API)
    CHANNELS = 1  # Mono
    DTYPE = 'float32'
    
    # Recording state
    MAX_RECORDING_TIME = 60  # seconds
    SILENCE_THRESHOLD = 0.008  # Amplitude threshold for silence detection (adjusted for 44100 Hz)
    SILENCE_DURATION = 1.5  # seconds of silence to auto-stop
    
    # Hotkey configuration
    DEFAULT_HOTKEY = "windows+caps lock"  # Windows/Super key
    HOTKEY = DEFAULT_HOTKEY  # Will be overridden if config file exists
    
    # Feedback settings
    USE_BEEP = True  # Use sound for feedback
    BEEP_FREQUENCY_START = 523.25  # Hz (C5 - musical note)
    BEEP_FREQUENCY_STOP = 392.00  # Hz (G4 - musical note)
    BEEP_DURATION = 0.5  # seconds (longer for more elegant decay)
    
    # OpenAI API settings
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    
    # Temporary file for audio
    TEMP_DIR = tempfile.gettempdir()
    
    # Application mode
    FLOWING_MODE = False  # Continuous transcription mode


class WhisperType:
    def __init__(self):
        self.recording = False
        self.audio_data = []
        self.recording_thread = None
        self.silence_counter = 0
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.continuous_recording = False  # Flag for continuous recording mode
        
        # Load configuration if it exists
        self.load_config()
        
        # Check if OpenAI API key is set
        if not Config.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY environment variable not set. Transcription will not work.")
            print("Warning: OPENAI_API_KEY environment variable not set.")
            print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        
        # Register hotkey
        keyboard.add_hotkey(Config.HOTKEY, self.toggle_recording)
        
        mode_str = " (Flowing Mode)" if Config.FLOWING_MODE else ""
        logger.info("WhisperType initialized%s. Press %s to start/stop recording.", mode_str, Config.HOTKEY)
        print(f"WhisperType initialized{mode_str}. Press {Config.HOTKEY} to start/stop recording.")

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
        """Play piano chord sounds for recording feedback."""
        try:
            if Config.USE_BEEP:
                sample_rate = Config.RECORDING_SAMPLE_RATE
                
                # Create time array
                t = np.linspace(0, duration, int(duration * sample_rate), False)
                
                if frequency == Config.BEEP_FREQUENCY_START:
                    # Start recording sound - gentle ascending C-E-G chord
                    # Define C major chord notes
                    c_note = 523.25  # C5
                    e_note = 659.25  # E5
                    g_note = 783.99  # G5
                    
                    # Create a gentle piano sound
                    sound = np.zeros_like(t)
                    
                    # Piano envelope with natural decay
                    piano_env = np.exp(-3 * t/duration)  # Natural decay
                    piano_env = piano_env * (0.9 + 0.1 * np.exp(-5 * t/duration))  # Add slight non-linearity
                    piano_env = piano_env / np.max(piano_env) * 0.5  # Normalize
                    
                    # C note (first)
                    sound += 0.5 * np.sin(2 * np.pi * c_note * t) * piano_env
                    # Add harmonics for richness
                    sound += 0.2 * np.sin(2 * np.pi * c_note * 2 * t) * piano_env * np.exp(-5 * t/duration)
                    sound += 0.1 * np.sin(2 * np.pi * c_note * 3 * t) * piano_env * np.exp(-8 * t/duration)
                    
                    # E note (delayed slightly)
                    delay_e = int(0.05 * sample_rate)  # 50ms delay for ascending effect
                    if delay_e < len(t):
                        e_env = np.zeros_like(t)
                        e_env[delay_e:] = np.exp(-3.5 * t[:-delay_e]/duration)
                        e_env = e_env / np.max(e_env) * 0.4  # Normalize
                        sound += 0.4 * np.sin(2 * np.pi * e_note * t) * e_env
                        # Add a subtle harmonic
                        sound += 0.15 * np.sin(2 * np.pi * e_note * 2 * t) * e_env * np.exp(-6 * t/duration)
                    
                    # G note (delayed more)
                    delay_g = int(0.1 * sample_rate)  # 100ms delay for ascending effect
                    if delay_g < len(t):
                        g_env = np.zeros_like(t)
                        g_env[delay_g:] = np.exp(-4 * t[:-delay_g]/duration)
                        g_env = g_env / np.max(g_env) * 0.35  # Normalize
                        sound += 0.35 * np.sin(2 * np.pi * g_note * t) * g_env
                        # Add a subtle harmonic
                        sound += 0.1 * np.sin(2 * np.pi * g_note * 2 * t) * g_env * np.exp(-7 * t/duration)
                    
                    # Apply a gentle low-pass filter for warmth
                    window_size = 7
                    kernel = np.hamming(window_size) / np.sum(np.hamming(window_size))
                    sound = np.convolve(sound, kernel, mode='same')
                    
                    # Final volume adjustment
                    sound = np.tanh(sound * 1.1) * 0.7
                    
                else:
                    # Stop recording sound - descending G-E-C chord
                    # Define C major chord notes
                    c_note = 523.25  # C5
                    e_note = 659.25  # E5
                    g_note = 783.99  # G5
                    
                    # Create a gentle piano sound
                    sound = np.zeros_like(t)
                    
                    # Piano envelope with natural decay
                    piano_env = np.exp(-2.5 * t/duration)  # Natural decay
                    piano_env = piano_env * (0.9 + 0.1 * np.exp(-4 * t/duration))  # Add slight non-linearity
                    piano_env = piano_env / np.max(piano_env) * 0.5  # Normalize
                    
                    # G note (first)
                    sound += 0.5 * np.sin(2 * np.pi * g_note * t) * piano_env
                    # Add harmonics for richness
                    sound += 0.2 * np.sin(2 * np.pi * g_note * 2 * t) * piano_env * np.exp(-5 * t/duration)
                    sound += 0.1 * np.sin(2 * np.pi * g_note * 3 * t) * piano_env * np.exp(-8 * t/duration)
                    
                    # E note (delayed slightly)
                    delay_e = int(0.05 * sample_rate)  # 50ms delay for descending effect
                    if delay_e < len(t):
                        e_env = np.zeros_like(t)
                        e_env[delay_e:] = np.exp(-3 * t[:-delay_e]/duration)
                        e_env = e_env / np.max(e_env) * 0.4  # Normalize
                        sound += 0.4 * np.sin(2 * np.pi * e_note * t) * e_env
                        # Add a subtle harmonic
                        sound += 0.15 * np.sin(2 * np.pi * e_note * 2 * t) * e_env * np.exp(-6 * t/duration)
                    
                    # C note (delayed more for resolving effect)
                    delay_c = int(0.1 * sample_rate)  # 100ms delay for descending effect
                    if delay_c < len(t):
                        c_env = np.zeros_like(t)
                        c_env[delay_c:] = np.exp(-2 * t[:-delay_c]/duration)  # Slower decay for final resolving note
                        c_env = c_env / np.max(c_env) * 0.45  # Normalize and slightly louder for resolution
                        sound += 0.45 * np.sin(2 * np.pi * c_note * t) * c_env
                        # Add harmonics for a rich resolving sound
                        sound += 0.2 * np.sin(2 * np.pi * c_note * 2 * t) * c_env * np.exp(-4 * t/duration)
                        sound += 0.1 * np.sin(2 * np.pi * c_note * 3 * t) * c_env * np.exp(-7 * t/duration)
                    
                    # Apply a gentle low-pass filter for warmth
                    window_size = 9
                    kernel = np.hamming(window_size) / np.sum(np.hamming(window_size))
                    sound = np.convolve(sound, kernel, mode='same')
                    
                    # Final volume adjustment
                    sound = np.tanh(sound * 1.1) * 0.7
                
                # Play the sound
                sd.play(sound, sample_rate)
                sd.wait()
        except Exception as e:
            logger.error("Failed to play sound: %s", e)

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
        
        try:
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
        except Exception as e:
            logger.error("Error in start_recording: %s", str(e), exc_info=True)
            print(f"Error starting recording: {e}")
            self.recording = False

    def _record_audio(self) -> None:
        """Record audio from microphone in a separate thread."""
        try:
            logger.info("Starting audio stream with sample rate: %d, channels: %d", Config.RECORDING_SAMPLE_RATE, Config.CHANNELS)
            with sd.InputStream(
                samplerate=Config.RECORDING_SAMPLE_RATE,
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
            logger.error("Error during recording: %s", str(e), exc_info=True)
            self.recording = False
            print(f"Error during recording: {e}")
            
            # Check if this is a sounddevice-related error
            if "sounddevice" in str(e).lower():
                print("Audio device error detected. Please check your microphone settings.")
                print("Make sure your microphone is properly connected and not being used by another application.")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Dict[str, Any], status: Any) -> None:
        """Callback function for audio stream."""
        if status:
            logger.warning("Audio callback status: %s", status)
        
        # Add the recorded audio to our buffer
        self.audio_data.append(indata.copy())
        
        # Calculate RMS value for better silence detection
        rms = np.sqrt(np.mean(np.square(indata)))
        
        # Check for silence to auto-stop recording or process chunk in flowing mode
        if rms < Config.SILENCE_THRESHOLD:
            self.silence_counter += frames / Config.RECORDING_SAMPLE_RATE
            if self.silence_counter >= Config.SILENCE_DURATION:
                if Config.FLOWING_MODE:
                    logger.info("Silence detected, processing audio chunk while continuing to record.")
                    # Process the current audio chunk in a separate thread
                    threading.Thread(target=self._process_audio_chunk).start()
                    # Clear the audio data to start a new chunk
                    self.audio_data = []
                    self.silence_counter = 0
                else:
                    logger.info("Silence detected, stopping recording.")
                    self.recording = False
        else:
            # Reset silence counter if we detect sound
            self.silence_counter = 0
            # Log audio levels periodically for debugging
            if len(self.audio_data) % 10 == 0:
                logger.debug("Current audio RMS level: %f (threshold: %f)", rms, Config.SILENCE_THRESHOLD)

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
            if not self.audio_data:
                logger.warning("No audio data to process")
                print("No audio data to process. Check if your microphone is capturing audio correctly.")
                return
                
            audio_data = np.concatenate(self.audio_data, axis=0)
            
            # Check if audio data is valid
            if len(audio_data) == 0 or np.max(np.abs(audio_data)) == 0:
                logger.warning("Empty or silent audio detected, skipping processing")
                print("Empty or silent audio detected. Check if your microphone is capturing audio correctly.")
                return
            
            # Resample audio to target sample rate
            resampled_audio = resample(audio_data, int(len(audio_data) * Config.TARGET_SAMPLE_RATE / Config.RECORDING_SAMPLE_RATE))
            
            # Normalize audio
            resampled_audio = resampled_audio / np.max(np.abs(resampled_audio))
            
            # Save as WAV file
            temp_file = os.path.join(Config.TEMP_DIR, "whispertype_recording.wav")
            wav.write(temp_file, Config.TARGET_SAMPLE_RATE, resampled_audio)
            
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
        
        except ValueError as e:
            logger.error("ValueError in processing audio: %s", str(e), exc_info=True)
            print(f"Error processing audio: {e}")
            if "zero-size array" in str(e):
                print("Received empty audio data. Check if your microphone is capturing audio correctly.")
        except Exception as e:
            logger.error("Error processing audio: %s", str(e), exc_info=True)
            print(f"Error processing audio: {e}")

    def _process_audio_chunk(self) -> None:
        """Process a chunk of audio during continuous recording."""
        if not self.audio_data:
            logger.warning("No audio data to process in chunk")
            return
            
        try:
            # Make a copy of the current audio data
            audio_chunk = self.audio_data.copy()
            
            # Convert audio data to a single numpy array
            audio_data = np.concatenate(audio_chunk, axis=0)
            
            # Check if audio data is valid
            if len(audio_data) == 0 or np.max(np.abs(audio_data)) == 0:
                logger.warning("Empty or silent audio chunk detected, skipping processing")
                return
            
            # Resample audio to target sample rate
            resampled_audio = resample(audio_data, int(len(audio_data) * Config.TARGET_SAMPLE_RATE / Config.RECORDING_SAMPLE_RATE))
            
            # Normalize audio
            resampled_audio = resampled_audio / np.max(np.abs(resampled_audio))
            
            # Save as WAV file
            temp_file = os.path.join(Config.TEMP_DIR, f"whispertype_chunk_{int(time.time())}.wav")
            wav.write(temp_file, Config.TARGET_SAMPLE_RATE, resampled_audio)
            
            logger.info("Audio chunk saved to temporary file: %s", temp_file)
            
            # Get transcription
            transcription = self._transcribe_audio(temp_file)
            
            if transcription:
                # Type the transcribed text
                self._type_text(transcription)
                logger.info("Chunk transcription complete: %s", transcription)
                print(f"Transcribed chunk: {transcription}")
            else:
                logger.warning("Chunk transcription failed or returned empty result.")
                print("Chunk transcription failed or returned empty result.")
            
            # Clean up temporary file
            try:
                os.remove(temp_file)
            except Exception as e:
                logger.warning("Failed to remove temporary file: %s", e)
        
        except ValueError as e:
            logger.error("ValueError in processing audio chunk: %s", str(e), exc_info=True)
            print(f"Error processing audio chunk: {e}")
            if "zero-size array" in str(e):
                print("Received empty audio data. Check if your microphone is capturing audio correctly.")
        except Exception as e:
            logger.error("Error processing audio chunk: %s", str(e), exc_info=True)
            print(f"Error processing audio chunk: {e}")

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
        mode_str = " (Flowing Mode)" if Config.FLOWING_MODE else ""
        print(f"WhisperType is running{mode_str}. Press {Config.HOTKEY} to start/stop recording.")
        print("Press Ctrl+C to exit.")
        
        try:
            # Keep the application running
            keyboard.wait()
        except KeyboardInterrupt:
            print("\nExiting WhisperType...")
        finally:
            keyboard.unhook_all()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="WhisperType - Voice Typing Application")
    parser.add_argument("-f", "--flowing", action="store_true", 
                        help="Enable flowing mode for continuous transcription")
    args = parser.parse_args()
    
    # Set flowing mode if specified
    Config.FLOWING_MODE = args.flowing
    
    print("Starting WhisperType..." + (" (Flowing Mode)" if Config.FLOWING_MODE else ""))
    print("This application requires root privileges to capture global keyboard events.")
    print("If you encounter permission issues, try running with sudo.")
    
    # Check if running as root (required for keyboard module on Linux)
    if os.geteuid() != 0:
        print("Warning: Not running as root. Global hotkeys may not work properly.")
        print("Consider running with: sudo python3 whispertype.py" + (" -f" if Config.FLOWING_MODE else ""))
    
    app = WhisperType()
    app.run()
