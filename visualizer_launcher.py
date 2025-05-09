#!/usr/bin/env python3
"""
Launcher for the Psychedelic Music Visualizer.
This script launches the Tkinter interface for selecting music
and configuring the visualizer.
"""

import os
import sys
import tkinter as tk
from tkinter_interface import VisualizerApp

def check_dependencies():
    """Check if all required dependencies are installed."""
    missing_packages = []
    
    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        import pygame
    except ImportError:
        missing_packages.append("pygame")
    
    try:
        import librosa
    except ImportError:
        missing_packages.append("librosa")
    
    try:
        import scipy
    except ImportError:
        missing_packages.append("scipy")
    
    try:
        from PIL import Image, ImageTk
    except ImportError:
        missing_packages.append("pillow")
        
    try:
        import cv2
    except ImportError:
        missing_packages.append("opencv-python")
    
    return missing_packages

def install_dependencies(packages):
    """Install missing dependencies."""
    import subprocess
    
    print("Installing missing dependencies...")
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}. Please install it manually.")
            return False
    
    return True

def check_files():
    """Check if all required Python files exist."""
    required_files = [
        "tkinter_interface.py",
        "psychedelic_visualizer.py",
        "enhanced_visualizer.py",
        "main_visualizer.py",
        "visualizer_recorder.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

def main():
    """Main function to start the application."""
    # Check for required files
    missing_files = check_files()
    if missing_files:
        print("Error: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease make sure all Python files are in the same directory.")
        input("Press Enter to exit...")
        return
    
    # Check dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        print("Some required packages are missing:")
        for package in missing_packages:
            print(f"  - {package}")
        
        install = input("Do you want to install them now? (y/n): ").lower()
        if install == 'y':
            if not install_dependencies(missing_packages):
                input("Press Enter to exit...")
                return
        else:
            print("Cannot continue without required packages.")
            input("Press Enter to exit...")
            return
    
    # Start the application
    root = tk.Tk()
    app = VisualizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()