import numpy as np
import librosa
import pygame
import sys
import os
import time
import random
from pygame import gfxdraw
from scipy.ndimage import gaussian_filter
from style_manager import StyleManager

class PsychedelicVisualizer:
    def __init__(self, width=1280, height=720):
        # Initialize pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Psychedelic Music Visualizer")
        self.clock = pygame.time.Clock()
        
         # Add style manager
        self.style_manager = StyleManager()
        self.current_style = "default"
        self.style_strength = 0.7  # How strongly to apply the style
        
        # Update color palettes from style manager
        self.palettes = [self.style_manager.get_color_palette(style) for style in 
                        ["default", "cyberpunk", "industrial", "nature", "abstract"]]
        
        # Get the default palettes for cycling
        self.palettes = self.style_manager.get_default_palettes()
        self.current_palette = random.choice(self.palettes)
        
        # Visual parameters
        self.particles = []
        self.num_particles = 50
        self.shapes = []
        self.rotation = 0
        self.wave_phase = 0
        self.last_beat_time = 0
        self.beat_triggered = False
        self.beat_duration = 0.1
        self.energy_smoothed = 0
        
        # Audio analysis parameters
        self.sample_rate = 0
        self.hop_length = 512
        self.onset_envelope = None
        self.tempo = 0
        self.beats = None
        self.beat_times = None
        self.spectral_centroids = None
        self.spectral_rolloff = None
        self.current_audio_time = 0
        self.audio_duration = 0
        
        # Effect parameters
        self.glow_surface = pygame.Surface((width, height))
        self.blur_amount = 15
        
        # Initialize particles
        self.init_particles()
        
    def init_particles(self):
        # Get colors from current style
        palette = self.style_manager.get_color_palette(self.current_style)
        
        self.particles = []
        for _ in range(self.num_particles):
            particle = {
                'x': random.randint(0, self.width),
                'y': random.randint(0, self.height),
                'size': random.randint(3, 15),
                'color': random.choice(palette),
                'speed': random.uniform(0.5, 2.0),
                'direction': random.uniform(0, 2 * np.pi)
            }
            self.particles.append(particle)
    
    # Add method to set style
    def set_style(self, style_name, strength=0.7):
        """Set the current visual style"""
        self.current_style = style_name
        self.style_strength = strength
        
        # Get style parameters
        # Use default if requested style doesn't exist
        if style_name in self.style_manager.effect_params:
            params = self.style_manager.get_effect_params(style_name)
        else:
            params = self.style_manager.get_effect_params("default")
            params = self.style_manager.get_effect_params(style_name)
            self.num_particles = params["num_particles"]
            # Reinitialize particles with new count
            self.init_particles()
    
    def analyze_audio(self, audio_path):
        """Analyze audio file to extract tempo, beats, and spectral features."""
        self.audio_path = audio_path
        y, sr = librosa.load(audio_path)
        self.sample_rate = sr
        self.audio_duration = librosa.get_duration(y=y, sr=sr)
        
        # Extract tempo and beat frames
        self.onset_envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        self.tempo, self.beats = librosa.beat.beat_track(onset_envelope=self.onset_envelope, sr=sr)
        self.tempo = float(self.tempo)  # Add this line to convert tempo to float
        self.beat_times = librosa.frames_to_time(self.beats, sr=sr, hop_length=self.hop_length)
        
        # Extract spectral features
        self.spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        self.spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)[0]
        
        # For energies, we'll use the harmonic component
        y_harmonic = librosa.effects.harmonic(y)
        spec_harmonic = np.abs(librosa.stft(y_harmonic, hop_length=self.hop_length))
        self.energies = np.mean(spec_harmonic, axis=0)
        
        # Tempo-based parameters
        self.beat_duration = 60.0 / self.tempo
        print(f"Tempo: {float(self.tempo):.2f} BPM")
        print(f"Audio duration: {self.audio_duration:.2f} seconds")
        print(f"Detected {len(self.beats)} beats")
        
        # Start audio playback
        pygame.mixer.init(frequency=sr)
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        
        return True
    
    def get_beat_intensity(self, current_time):
        """Calculate beat intensity based on proximity to the nearest beat."""
        if len(self.beat_times) == 0:
            return 0
        
        # Find the nearest beat
        nearest_beat_idx = np.argmin(np.abs(self.beat_times - current_time))
        nearest_beat_time = self.beat_times[nearest_beat_idx]
        
        # Calculate time difference to nearest beat
        time_diff = abs(current_time - nearest_beat_time)
        
        # Calculate beat intensity (1.0 at beat, diminishing as we move away)
        beat_intensity = max(0, 1.0 - (time_diff / (self.beat_duration * 0.5)))
        
        # Trigger beat effect if we're close to a beat
        if time_diff < 0.05 and current_time > self.last_beat_time + 0.2:
            self.last_beat_time = current_time
            self.beat_triggered = True
            
            # Change palette occasionally on beat
            if random.random() < 0.1:
                self.current_palette = random.choice(self.palettes)
            
            return 1.0
        
        return beat_intensity
    
    def get_energy_at_time(self, current_time):
        """Get the energy level at the current time point."""
        frame_idx = int(current_time * self.sample_rate / self.hop_length)
        if 0 <= frame_idx < len(self.energies):
            energy = self.energies[frame_idx]
            # Normalize and smooth energy
            energy_norm = min(1.0, energy / np.max(self.energies))
            self.energy_smoothed = 0.9 * self.energy_smoothed + 0.1 * energy_norm
            return self.energy_smoothed
        return 0.5  # Default energy if out of bounds
    
    def draw_psychedelic_background(self, beat_intensity, energy):
        # Create gradients and patterns that shift with time and music
        angle = self.rotation * 0.01
        
        # Create a base gradient
        for i in range(0, self.height, 2):
            # Oscillate between colors based on position and time
            color_idx1 = int(time.time() * 0.5 + i * 0.01) % len(self.current_palette)
            color_idx2 = (color_idx1 + 1) % len(self.current_palette)
            
            # Interpolate between colors based on beat intensity
            color1 = self.current_palette[color_idx1]
            color2 = self.current_palette[color_idx2]
            
            # Create a wave pattern
            wave_offset = int(20 * np.sin(i * 0.01 + self.wave_phase))
            line_width = int(self.width + 40 * beat_intensity)
            
            # Draw the wave line
            pygame.draw.line(self.screen, color1, 
                             (max(0, wave_offset), i), 
                             (min(self.width, wave_offset + line_width), i), 
                             2)
            
        # Increment wave phase based on energy and tempo
        self.wave_phase += 0.01 + 0.05 * energy
        
    def draw_mandala(self, center_x, center_y, beat_intensity, energy):
        # Number of circles increases with energy
        num_circles = int(5 + 15 * energy)
        
        # Base radius changes with beat intensity
        base_radius = 50 + beat_intensity * 100
        
        for i in range(num_circles):
            # Calculate radius and thickness
            radius = base_radius + i * 10
            thickness = max(1, int(3 * beat_intensity))
            
            # Calculate color based on position in the palette
            color_idx = (i + int(time.time() * 2)) % len(self.current_palette)
            color = self.current_palette[color_idx]
            
            # Add some alpha for a layered effect
            alpha = int(255 * (1.0 - i / num_circles))
            color_with_alpha = (*color, alpha)
            
            # Create a temporary surface for the circle with alpha
            circle_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(circle_surf, color_with_alpha, (radius, radius), radius, thickness)
            
            # Rotate slightly for a spinning effect
            rot_angle = (self.rotation + i * 30) % 360
            rotated_surf = pygame.transform.rotate(circle_surf, rot_angle)
            
            # Blit the rotated surface
            rect = rotated_surf.get_rect(center=(center_x, center_y))
            self.screen.blit(rotated_surf, rect)
    
    def draw_particles(self, beat_intensity, energy):
        for particle in self.particles:
            # Update particle position based on direction and speed
            # Speed increases with beat intensity
            speed = particle['speed'] * (1 + 2 * beat_intensity)
            particle['x'] += np.cos(particle['direction']) * speed
            particle['y'] += np.sin(particle['direction']) * speed
            
            # Wrap particles around screen edges
            if particle['x'] < 0:
                particle['x'] = self.width
            elif particle['x'] > self.width:
                particle['x'] = 0
            if particle['y'] < 0:
                particle['y'] = self.height
            elif particle['y'] > self.height:
                particle['y'] = 0
            
            # Particle size pulses with energy
            size = particle['size'] * (1 + energy)
            
            # Draw particle
            pygame.draw.circle(self.screen, particle['color'], 
                              (int(particle['x']), int(particle['y'])), 
                              int(size))
            
            # Add glow effect
            pygame.draw.circle(self.glow_surface, 
                              (*particle['color'], 100),  # Add alpha for glow
                              (int(particle['x']), int(particle['y'])), 
                              int(size * 2))
    
    def apply_glow_effect(self):
        # Apply gaussian blur to the glow surface
        # This is a simple approximation since pygame doesn't have built-in gaussian blur
        # We'll use a fake blur by drawing multiple semi-transparent circles
        
        # Clear the glow surface
        self.glow_surface.fill((0, 0, 0, 0))
        
        # Draw particles with glow
        for particle in self.particles:
            # Draw multiple circles with decreasing opacity
            max_radius = particle['size'] * 3
            for r in range(int(max_radius), 0, -1):
                alpha = int(100 * (1 - r / max_radius))
                pygame.draw.circle(self.glow_surface, 
                                  (*particle['color'], alpha),
                                  (int(particle['x']), int(particle['y'])), 
                                  r)
        
        # Blit the glow surface onto the main screen
        self.screen.blit(self.glow_surface, (0, 0), special_flags=pygame.BLEND_ADD)
    
    def run(self, audio_path=None):
        print(f"Starting visualizer with style: {self.current_style}, strength: {self.style_strength}")
    
        # Force style for testing
        self.current_style = "cyberpunk"
        self.style_strength = 1.0
        print(f"FORCED style to: {self.current_style}")
        if audio_path:
            if not self.analyze_audio(audio_path):
                print("Error analyzing audio file")
                return
        
        start_time = time.time()
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Pause/resume music
                        if pygame.mixer.music.get_busy():
                            pygame.mixer.music.pause()
                        else:
                            pygame.mixer.music.unpause()
            
            # Clear the screen
            self.screen.fill((0, 0, 0))
            
            # Calculate current time in the audio
            if audio_path:
                # Get current audio position from pygame mixer
                if pygame.mixer.music.get_busy():
                    self.current_audio_time = (time.time() - start_time) % self.audio_duration
                else:
                    # If music is paused, time doesn't advance
                    start_time = time.time() - self.current_audio_time
            else:
                # If no audio, just use real time
                self.current_audio_time = time.time() - start_time
            
            # Get beat intensity and energy for the current time
            beat_intensity = self.get_beat_intensity(self.current_audio_time) if audio_path else 0.5
            energy = self.get_energy_at_time(self.current_audio_time) if audio_path else 0.5
            
            # Draw psychedelic background
            self.draw_psychedelic_background(beat_intensity, energy)
            
            # Draw mandala in the center
            self.draw_mandala(self.width // 2, self.height // 2, beat_intensity, energy)
            
            # Draw and update particles
            self.draw_particles(beat_intensity, energy)
            
            # Apply glow effect
            self.apply_glow_effect()
            
            # Update rotation based on energy and tempo
            self.rotation += 0.2 + 0.8 * energy
            
            # Handle beat triggered events
            if self.beat_triggered:
                # Do something on beat
                self.beat_triggered = False
            
            # Apply style transfer if enabled
            if self.current_style != "default":
                print(f"Applying {self.current_style} style with strength {self.style_strength}")
                # Get screen content as numpy array
                screen_array = pygame.surfarray.array3d(self.screen)
                screen_array = screen_array.transpose([1, 0, 2])  # Pygame format to OpenCV format
                
                # Apply style
                styled_array = self.style_manager.apply_style(
                    screen_array, self.current_style, self.style_strength)
                
                print(f"Style transfer complete - before: {screen_array.shape}, after: {styled_array.shape}")
                
                # Update screen
                styled_surface = pygame.surfarray.make_surface(
                    styled_array.transpose([1, 0, 2]))  # OpenCV format to Pygame format
                self.screen.blit(styled_surface, (0, 0))
            
            # Update display
            print(f"Current style: {self.current_style}, Strength: {self.style_strength}")
            pygame.display.flip()
            self.clock.tick(60)  # Cap at 60 FPS
        
        pygame.quit()

if __name__ == "__main__":
    # Get audio file from command line argument or use default
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        if not os.path.exists(audio_path):
            print(f"Error: File '{audio_path}' not found.")
            sys.exit(1)
    else:
        print("Usage: python visualizer.py <audio_file>")
        sys.exit(1)
    
    # Create and run visualizer
    visualizer = PsychedelicVisualizer()
    visualizer.run(audio_path)