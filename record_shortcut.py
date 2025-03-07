#!/usr/bin/env python3
"""
Shortcut Recorder for WhisperType

This utility helps you identify the correct key names for your keyboard
and configure a custom shortcut for WhisperType.
"""

import os
import sys
import time
import json
import keyboard

def on_key_event(event):
    """Handle key events and print them."""
    if event.event_type == 'down':
        print(f"Key pressed: {event.name} (scan code: {event.scan_code})")
        
        # Add the key to our current combination
        if event.name not in current_keys:
            current_keys.append(event.name)
            
        # Update the display of current combination
        if current_keys:
            print(f"Current combination: {'+'.join(current_keys)}")

def on_key_release(event):
    """Handle key release events."""
    if event.name in current_keys:
        current_keys.remove(event.name)

def save_shortcut(shortcut):
    """Save the shortcut to the configuration file."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    
    # Load existing config or create a new one
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError:
            config = {}
    else:
        config = {}
    
    # Update the shortcut
    config['hotkey'] = shortcut
    
    # Save the config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Shortcut '{shortcut}' saved to {config_path}")

def main():
    """Main function to record keyboard shortcuts."""
    print("=== WhisperType Shortcut Recorder ===")
    print("Press the keys you want to use as your shortcut.")
    print("When you're satisfied with your combination, press Enter to save it.")
    print("Press Esc to cancel.")
    print("\nListening for key presses...")
    
    # Hook keyboard events
    keyboard.on_press(on_key_event)
    keyboard.on_release(on_key_release)
    
    try:
        # Wait for Enter key to save or Esc to cancel
        while True:
            if keyboard.is_pressed('enter') and current_keys:
                # Remove 'enter' from the combination if it's there
                if 'enter' in current_keys:
                    current_keys.remove('enter')
                
                shortcut = '+'.join(current_keys)
                save_shortcut(shortcut)
                print(f"\nShortcut '{shortcut}' has been saved!")
                print("You can now use this shortcut with WhisperType.")
                break
            
            elif keyboard.is_pressed('esc'):
                print("\nCancelled. No shortcut was saved.")
                break
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nCancelled. No shortcut was saved.")
    
    finally:
        # Unhook keyboard events
        keyboard.unhook_all()

if __name__ == "__main__":
    # Check if running as root (required for keyboard module on Linux)
    if os.geteuid() != 0:
        print("This script requires root privileges to capture keyboard events.")
        print("Please run with: sudo python3 record_shortcut.py")
        sys.exit(1)
    
    # Initialize global variables
    current_keys = []
    
    main()
