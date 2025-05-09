import os
import cv2
import numpy as np
import pygame
import time
import threading
from datetime import datetime

class VisualizerRecorder:
    """
    Class to record the visualizer output as a video file.
    This uses OpenCV to capture pygame display frames and save them as a video.
    """
    def __init__(self, fps=30, codec="mp4v"):
        self.fps = fps
        self.codec = codec
        self.video_writer = None
        self.recording = False
        self.frame_count = 0
        self.start_time = 0
        self.recording_thread = None
        self.output_file = None
        
    def start_recording(self, screen, output_path=None):
        """Start recording the visualizer"""
        if self.recording:
            return False
            
        # Create output filename with timestamp if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"visualizer_recording_{timestamp}.mp4"
        
        self.output_file = output_path
        
        # Get screen dimensions
        width, height = screen.get_size()
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.video_writer = cv2.VideoWriter(
            output_path, 
            fourcc, 
            self.fps, 
            (width, height)
        )
        
        if not self.video_writer.isOpened():
            print("Error: Could not open video writer")
            return False
        
        self.recording = True
        self.frame_count = 0
        self.start_time = time.time()
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(
            target=self._record_thread, 
            args=(screen,),
            daemon=True
        )
        self.recording_thread.start()
        
        return True
    
    def _record_thread(self, screen):
        """Background thread for recording frames"""
        while self.recording:
            self.capture_frame(screen)
            # Limit capture rate to maintain performance
            time.sleep(1/self.fps)
    
    def capture_frame(self, screen):
        """Capture a single frame from pygame display"""
        if not self.recording or self.video_writer is None:
            return
            
        # Get pygame surface as a string buffer
        buffer = pygame.image.tostring(screen, "RGB")
        width, height = screen.get_size()
        
        # Convert buffer to numpy array for OpenCV
        img = np.frombuffer(buffer, dtype=np.uint8)
        img = img.reshape((height, width, 3))
        
        # OpenCV uses BGR format, but pygame uses RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Write frame
        self.video_writer.write(img)
        self.frame_count += 1
    
    def stop_recording(self):
        """Stop recording and finalize the video file"""
        if not self.recording:
            return None
            
        self.recording = False
        
        # Wait for recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        # Release video writer
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        duration = time.time() - self.start_time
        
        print(f"Recording finished: {self.output_file}")
        print(f"Frames: {self.frame_count}, Duration: {duration:.2f} seconds")
        
        return self.output_file
    
    def get_recording_stats(self):
        """Get current recording statistics"""
        if not self.recording:
            return None
            
        duration = time.time() - self.start_time
        return {
            "frames": self.frame_count,
            "duration": duration,
            "fps": self.frame_count / duration if duration > 0 else 0
        }