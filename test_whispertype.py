#!/usr/bin/env python3
"""
Test script for WhisperType

This script tests the basic functionality of WhisperType without requiring an OpenAI API key.
It simulates the recording and typing process with a predefined text.
"""

import os
import time
import threading
import subprocess
import keyboard
import numpy as np
import sounddevice as sd

def play_beep(frequency=440, duration=0.2, sample_rate=16000):
    """Play a beep sound for feedback using numpy and sounddevice."""
    try:
        # Generate a sine wave for the beep
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        beep = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Play the beep
        sd.play(beep, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"Failed to play beep: {e}")

def simulate_typing(text):
    """Simulate typing text using xdotool."""
    try:
        # Escape special characters for shell
        escaped_text = text.replace('"', '\\"')
        
        # Use xdotool to type the text
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", escaped_text],
            check=True
        )
        print("Text typed successfully.")
    except Exception as e:
        print(f"Error typing text: {e}")

def on_hotkey():
    """Simulate the recording and transcription process."""
    # Play start sound (A4 note)
    play_beep(frequency=440, duration=0.2)
    print("Recording started...")
    
    # Simulate recording for 2 seconds
    time.sleep(2)
    
    # Play stop sound (A5 note - one octave higher)
    play_beep(frequency=880, duration=0.2)
    print("Recording stopped. Transcribing...")
    
    # Simulate transcription delay
    time.sleep(1)
    
    # Simulate typing the transcribed text
    test_text = "This is a test of the WhisperType application. It works without requiring an OpenAI API key."
    simulate_typing(test_text)
    print(f"Transcribed: {test_text}")

def main():
    # Register hotkey (Super/Windows + Caps Lock)
    hotkey = "windows+caps lock"
    keyboard.add_hotkey(hotkey, on_hotkey)
    
    print(f"WhisperType Test is running. Press {hotkey} to simulate recording and typing.")
    print("Press Ctrl+C to exit.")
    
    try:
        # Keep the application running
        keyboard.wait()
    except KeyboardInterrupt:
        print("\nExiting WhisperType Test...")
    finally:
        keyboard.unhook_all()

if __name__ == "__main__":
    print("Starting WhisperType Test...")
    print("This test requires root privileges to capture global keyboard events.")
    
    # Check if running as root (required for keyboard module on Linux)
    if os.geteuid() != 0:
        print("Warning: Not running as root. Global hotkeys may not work properly.")
        print("Consider running with: sudo python3 test_whispertype.py")
    
    main()
