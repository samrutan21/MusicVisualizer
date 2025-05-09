import numpy as np
import pygame
import random
import colorsys
from pygame import gfxdraw

class AdvancedEffects:
    """
    Add-on class to enhance the PsychedelicVisualizer with more effects.
    This class can be integrated with the main visualizer.
    """
    
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.width = visualizer.width
        self.height = visualizer.height
        self.screen = visualizer.screen
        
        # Effect parameters
        self.fractals = []
        self.num_fractals = 5
        self.fract_depth = 3
        self.ribbons = []
        self.num_ribbons = 10
        self.ribbon_points = 20
        
        # Create effects
        self.init_fractals()
        self.init_ribbons()
        
        # Offscreen surfaces for effects
        self.fractal_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.ribbon_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
    def init_fractals(self):
        """Initialize fractal data structures"""
        self.fractals = []
        for _ in range(self.num_fractals):
            fractal = {
                'x': random.randint(0, self.width),
                'y': random.randint(0, self.height),
                'size': random.randint(50, 150),
                'rotation': random.uniform(0, 2 * np.pi),
                'depth': random.randint(2, self.fract_depth),
                'color': random.choice(self.visualizer.current_palette)
            }
            self.fractals.append(fractal)
    
    def init_ribbons(self):
        """Initialize ribbon data structures"""
        self.ribbons = []
        for _ in range(self.num_ribbons):
            # Create a ribbon with multiple control points
            points = []
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            
            for i in range(self.ribbon_points):
                points.append((x, y))
                # Random walk for next point
                x += random.randint(-20, 20)
                y += random.randint(-20, 20)
                
            ribbon = {
                'points': points,
                'width': random.randint(5, 15),
                'color': random.choice(self.visualizer.current_palette),
                'speed': random.uniform(0.5, 2.0)
            }
            self.ribbons.append(ribbon)
    
    def draw_fractal_tree(self, surface, x, y, length, angle, depth, color, branch_angle=np.pi/5):
        """Recursively draw a fractal tree"""
        if depth == 0:
            return
            
        # Calculate end point of branch
        end_x = x + length * np.cos(angle)
        end_y = y + length * np.sin(angle)
        
        # Draw the branch
        pygame.draw.line(surface, color, (int(x), int(y)), (int(end_x), int(end_y)), 
                         max(1, int(depth)))
        
        # Draw left branch
        self.draw_fractal_tree(surface, end_x, end_y, length * 0.7, angle - branch_angle, 
                              depth - 1, color, branch_angle)
        
        # Draw right branch
        self.draw_fractal_tree(surface, end_x, end_y, length * 0.7, angle + branch_angle, 
                              depth - 1, color, branch_angle)
    
    def draw_fractals(self, beat_intensity, energy):
        """Draw fractal patterns"""
        # Clear the fractal surface
        self.fractal_surface.fill((0, 0, 0, 0))
        
        for fractal in self.fractals:
            # Adjust fractal parameters based on music
            size = fractal['size'] * (1 + beat_intensity * 0.5)
            rotation = fractal['rotation'] + 0.01 * energy
            fractal['rotation'] = rotation
            
            # Draw the fractal tree
            self.draw_fractal_tree(
                self.fractal_surface, 
                fractal['x'], 
                fractal['y'], 
                size,
                rotation, 
                fractal['depth'], 
                fractal['color'],
                np.pi/4 * (0.5 + 0.5 * energy)  # Branch angle varies with energy
            )
        
        # Apply the fractal surface with alpha blending
        self.screen.blit(self.fractal_surface, (0, 0), special_flags=pygame.BLEND_ADD)
    
    def update_ribbons(self, beat_intensity, energy):
        """Update and draw flowing ribbons"""
        # Clear the ribbon surface
        self.ribbon_surface.fill((0, 0, 0, 0))
        
        for ribbon in self.ribbons:
            # Update ribbon points
            for i in range(len(ribbon['points']) - 1, 0, -1):
                # Points follow their predecessor with some noise
                x_prev, y_prev = ribbon['points'][i-1]
                x, y = ribbon['points'][i]
                
                # Calculate new position
                noise_x = random.uniform(-5, 5) * energy
                noise_y = random.uniform(-5, 5) * energy
                
                # Move toward previous point with noise
                move_speed = ribbon['speed'] * (1 + beat_intensity)
                dx = (x_prev - x) * 0.1 * move_speed + noise_x
                dy = (y_prev - y) * 0.1 * move_speed + noise_y
                
                # Update position
                new_x = x + dx
                new_y = y + dy
                
                # Keep within bounds
                new_x = max(0, min(self.width, new_x))
                new_y = max(0, min(self.height, new_y))
                
                ribbon['points'][i] = (new_x, new_y)
            
            # Update lead point based on music
            x, y = ribbon['points'][0]
            
            # Move in a more dramatic way on beats
            if beat_intensity > 0.8:
                dx = random.uniform(-30, 30)
                dy = random.uniform(-30, 30)
            else:
                dx = random.uniform(-10, 10)
                dy = random.uniform(-10, 10)
            
            new_x = x + dx
            new_y = y + dy
            
            # Keep within bounds
            new_x = max(0, min(self.width, new_x))
            new_y = max(0, min(self.height, new_y))
            
            ribbon['points'][0] = (new_x, new_y)
            
            # Draw the ribbon as a smooth curve
            if len(ribbon['points']) >= 3:
                # Get color with intensity based on energy
                base_color = ribbon['color']
                h, s, v = colorsys.rgb_to_hsv(base_color[0]/255, base_color[1]/255, base_color[2]/255)
                # Increase saturation and value with energy
                s = min(1.0, s + energy * 0.3)
                v = min(1.0, v + beat_intensity * 0.3)
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                color = (int(r * 255), int(g * 255), int(b * 255))
                
                # Draw as a series of connected quads with alpha gradient
                points = ribbon['points']
                width = ribbon['width'] * (1 + beat_intensity * 2)
                
                for i in range(len(points) - 1):
                    x1, y1 = points[i]
                    x2, y2 = points[i+1]
                    
                    # Calculate perpendicular direction for width
                    dx = x2 - x1
                    dy = y2 - y1
                    length = max(0.001, np.sqrt(dx*dx + dy*dy))
                    
                    # Normalize and perpendicular
                    nx = -dy / length
                    ny = dx / length
                    
                    # Calculate corner points for quad
                    half_width = width / 2
                    p1 = (x1 + nx * half_width, y1 + ny * half_width)
                    p2 = (x1 - nx * half_width, y1 - ny * half_width)
                    p3 = (x2 - nx * half_width, y2 - ny * half_width)
                    p4 = (x2 + nx * half_width, y2 + ny * half_width)
                    
                    # Create alpha based on position in ribbon
                    alpha = int(255 * (1.0 - i / len(points)))
                    
                    # Draw anti-aliased polygon with alpha
                    gfxdraw.aapolygon(self.ribbon_surface, [p1, p2, p3, p4], (*color, alpha))
                    gfxdraw.filled_polygon(self.ribbon_surface, [p1, p2, p3, p4], (*color, alpha))
        
        # Apply the ribbon surface with alpha blending
        self.screen.blit(self.ribbon_surface, (0, 0), special_flags=pygame.BLEND_ADD)
    
    def draw_spectrum_analyzer(self, beat_intensity, energy):
        """Draw a spectrum analyzer effect at the bottom of the screen"""
        # Calculate spectrum height based on energy and beat intensity
        if not hasattr(self.visualizer, 'spectral_centroids') or self.visualizer.spectral_centroids is None:
            return
            
        # Get current frame index
        frame_idx = int(self.visualizer.current_audio_time * self.visualizer.sample_rate / self.visualizer.hop_length)
        if frame_idx >= len(self.visualizer.spectral_centroids):
            return
            
        # Height of the analyzer
        analyzer_height = 100
        bar_spacing = 2
        num_bars = 64
        bar_width = (self.width // num_bars) - bar_spacing
        
        # Bottom of the screen
        y_base = self.height - 20
        
        # Create spectrum values (simulated here based on centroid)
        centroid = self.visualizer.spectral_centroids[frame_idx]
        rolloff = self.visualizer.spectral_rolloff[frame_idx]
        
        # Normalize
        centroid_norm = centroid / 8000  # Typical range for centroid
        rolloff_norm = rolloff / 12000  # Typical range for rolloff
        
        # Create a spectrum shape based on centroid and rolloff
        spectrum = []
        for i in range(num_bars):
            # Position in spectrum (0-1)
            pos = i / num_bars
            
            # Create a shape peaking at the centroid position
            value = np.exp(-10 * ((pos - centroid_norm) ** 2))
            
            # Add another peak at the rolloff position
            value += 0.7 * np.exp(-10 * ((pos - rolloff_norm) ** 2))
            
            # Add some noise and beat response
            value += 0.2 * random.random() * beat_intensity
            
            # Scale to analyzer height
            value = min(1.0, value) * analyzer_height
            spectrum.append(value)
            
        # Draw the bars
        for i, height in enumerate(spectrum):
            # Position of this bar
            x = i * (bar_width + bar_spacing)
            
            # Color changes across spectrum
            hue = i / num_bars
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            color = (int(r * 255), int(g * 255), int(b * 255))
            
            # Draw the bar
            bar_height = int(height * (1 + 0.5 * beat_intensity))
            pygame.draw.rect(self.screen, color, 
                            (x, y_base - bar_height, bar_width, bar_height))
            
            # Add glow
            glow_surf = pygame.Surface((bar_width, bar_height * 2), pygame.SRCALPHA)
            for j in range(bar_height):
                alpha = int(150 * (1 - j / bar_height))
                pygame.draw.rect(glow_surf, (*color, alpha), 
                                (0, bar_height - j, bar_width, 2))
            
            self.screen.blit(glow_surf, (x, y_base - bar_height * 2))
    
    def add_text_overlay(self):
        """Add song information or visualizer controls"""
        # Only do this if we have audio loaded
        if not hasattr(self.visualizer, 'audio_duration') or self.visualizer.audio_duration == 0:
            return
            
        # Create a small font
        font = pygame.font.SysFont(None, 24)
        
        # Show current time / total time
        current_time = self.visualizer.current_audio_time
        total_time = self.visualizer.audio_duration
        
        time_text = f"{int(current_time // 60)}:{int(current_time % 60):02d} / {int(total_time // 60)}:{int(total_time % 60):02d}"
        time_surf = font.render(time_text, True, (255, 255, 255))
        
        # Show tempo
        tempo_text = f"{float(self.visualizer.tempo):.1f} BPM"
        tempo_surf = font.render(tempo_text, True, (255, 255, 255))
        
        # Draw with semi-transparent background
        padding = 5
        bg_rect1 = pygame.Rect(10, 10, time_surf.get_width() + padding*2, time_surf.get_height() + padding*2)
        bg_rect2 = pygame.Rect(10, 40, tempo_surf.get_width() + padding*2, tempo_surf.get_height() + padding*2)
        
        # Draw semi-transparent backgrounds
        s = pygame.Surface((bg_rect1.width, bg_rect1.height))
        s.set_alpha(128)
        s.fill((0, 0, 0))
        self.screen.blit(s, (bg_rect1.x, bg_rect1.y))
        
        s = pygame.Surface((bg_rect2.width, bg_rect2.height))
        s.set_alpha(128)
        s.fill((0, 0, 0))
        self.screen.blit(s, (bg_rect2.x, bg_rect2.y))
        
        # Draw text
        self.screen.blit(time_surf, (bg_rect1.x + padding, bg_rect1.y + padding))
        self.screen.blit(tempo_surf, (bg_rect2.x + padding, bg_rect2.y + padding))
        
    def update(self, beat_intensity, energy):
        """Update and draw all advanced effects"""
        # Draw flowing ribbons
        self.update_ribbons(beat_intensity, energy)
        
        # Draw fractals
        self.draw_fractals(beat_intensity, energy)
        
        # Draw spectrum analyzer
        self.draw_spectrum_analyzer(beat_intensity, energy)
        
        # Add text overlay
        self.add_text_overlay()