import pygame
import sys
import os
import argparse
from psychedelic_visualizer import PsychedelicVisualizer
from enhanced_visualizer import AdvancedEffects
from visualizer_recorder import VisualizerRecorder
import time

class CompleteVisualizer:
    """
    Combines the base visualizer with enhanced effects.
    """
    def __init__(self, width=1280, height=720):
        # Initialize base visualizer
        self.base_visualizer = PsychedelicVisualizer(width, height)
        
        # Initialize enhanced effects
        self.advanced_effects = AdvancedEffects(self.base_visualizer)
        
        # Initialize recorder
        self.recorder = VisualizerRecorder()
        self.is_recording = False
        
        # Effect toggle flags
        self.show_particles = True
        self.show_mandala = True
        self.show_ribbons = True
        self.show_fractals = True
        self.show_spectrum = True
        
        # Style properties
        self.current_style = "default"
        self.style_strength = 0.7
    
    # Style control method
    def set_style(self, style_name, strength=0.7):
        """Set the current visual style"""
        print(f"Switching to style: {style_name} with strength {strength}")
        self.current_style = style_name
        self.style_strength = strength
        self.base_visualizer.set_style(style_name, strength)
            
    def toggle_effect(self, effect_name):
        """Toggle specific visual effects on/off"""
        if effect_name == 'particles':
            self.show_particles = not self.show_particles
        elif effect_name == 'mandala':
            self.show_mandala = not self.show_mandala
        elif effect_name == 'ribbons':
            self.show_ribbons = not self.show_ribbons
        elif effect_name == 'fractals':
            self.show_fractals = not self.show_fractals
        elif effect_name == 'spectrum':
            self.show_spectrum = not self.show_spectrum
    
    def start_recording(self, output_path=None):
        """Start recording the visualizer"""
        if not self.is_recording:
            # Create output filename based on audio file if not provided
            if output_path is None:
                import os
                import time
                
                # Get base audio filename
                audio_filename = os.path.basename(self.base_visualizer.audio_path)
                audio_name, _ = os.path.splitext(audio_filename)
                
                # Create output directory if it doesn't exist
                output_dir = "recordings"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Create timestamped filename
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(output_dir, f"{audio_name}_{timestamp}.mp4")
            
            # Start recording
            result = self.recorder.start_recording(self.base_visualizer.screen, output_path)
            if result:
                self.is_recording = True
                print(f"Started recording to: {output_path}")
                return True
            else:
                print("Failed to start recording")
                return False
        return False
        
    def stop_recording(self):
        """Stop recording the visualizer"""
        if self.is_recording:
            output_file = self.recorder.stop_recording()
            self.is_recording = False
            print(f"Recording saved to: {output_file}")
            return output_file
        return None
    
    def run(self, audio_path, auto_record=False, record_output=None):
        """Run the complete visualizer with all effects"""
        # Initialize audio analysis
        if not self.base_visualizer.analyze_audio(audio_path):
            print("Error analyzing audio file")
            return
        
        # Main loop variables
        running = True
        start_time = pygame.time.get_ticks() / 1000.0
        clock = pygame.time.Clock()
        
        if auto_record:
            self.start_recording(record_output)
        
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
                    # Toggle effects with number keys
                    elif event.key == pygame.K_1:
                        self.toggle_effect('particles')
                    elif event.key == pygame.K_2:
                        self.toggle_effect('mandala')
                    elif event.key == pygame.K_3:
                        self.toggle_effect('ribbons')
                    elif event.key == pygame.K_4:
                        self.toggle_effect('fractals')
                    elif event.key == pygame.K_5:
                        self.toggle_effect('spectrum')
                    # Change color palette with 'c' key
                    elif event.key == pygame.K_c:
                        self.base_visualizer.current_palette = next(
                            (p for i, p in enumerate(self.base_visualizer.palettes) 
                             if i > self.base_visualizer.palettes.index(self.base_visualizer.current_palette)),
                            self.base_visualizer.palettes[0]
                        )
                    # Toggle styles with keys
                    elif event.key == pygame.K_F1:
                        self.set_style("default", 1.0)
                    elif event.key == pygame.K_F2:
                        self.set_style("cyberpunk", 1.0)
                    elif event.key == pygame.K_F3:
                        self.set_style("industrial", 1.0)
                    elif event.key == pygame.K_F4:
                        self.set_style("nature", 1.0)
                    elif event.key == pygame.K_F5:
                        self.set_style("abstract", 1.0)
                    # Adjust style strength
                    elif event.key == pygame.K_MINUS:
                        self.style_strength = max(0.1, self.style_strength - 0.1)
                        self.set_style(self.current_style, self.style_strength)
                    elif event.key == pygame.K_EQUALS:  # Plus key
                        self.style_strength = min(1.0, self.style_strength + 0.1)
                        self.set_style(self.current_style, self.style_strength)
                    # Toggle recording with 'r' key
                    elif event.key == pygame.K_r:
                        if not self.is_recording:
                            self.start_recording()
                        else:
                            self.stop_recording()
            
            # Update current audio time
            if pygame.mixer.music.get_busy():
                self.base_visualizer.current_audio_time = (pygame.time.get_ticks() / 1000.0 - start_time) % self.base_visualizer.audio_duration
            else:
                # If music is paused, time doesn't advance
                start_time = pygame.time.get_ticks() / 1000.0 - self.base_visualizer.current_audio_time
            
            # Get music analysis data
            beat_intensity = self.base_visualizer.get_beat_intensity(self.base_visualizer.current_audio_time)
            energy = self.base_visualizer.get_energy_at_time(self.base_visualizer.current_audio_time)
            
            # Clear screen
            self.base_visualizer.screen.fill((0, 0, 0))
            
            # Draw base visualizer elements
            self.base_visualizer.draw_psychedelic_background(beat_intensity, energy)
            
            if self.show_mandala:
                self.base_visualizer.draw_mandala(
                    self.base_visualizer.width // 2, 
                    self.base_visualizer.height // 2, 
                    beat_intensity, 
                    energy
                )
            
            if self.show_particles:
                self.base_visualizer.draw_particles(beat_intensity, energy)
                self.base_visualizer.apply_glow_effect()
            
            # Draw enhanced effects
            if self.show_ribbons or self.show_fractals or self.show_spectrum:
                # Update rotation for consistent motion
                self.base_visualizer.rotation += 0.2 + 0.8 * energy
                
                # Draw ribbons if enabled
                if self.show_ribbons:
                    self.advanced_effects.update_ribbons(beat_intensity, energy)
                
                # Draw fractals if enabled
                if self.show_fractals:
                    self.advanced_effects.draw_fractals(beat_intensity, energy)
                
                # Draw spectrum analyzer if enabled
                if self.show_spectrum:
                    self.advanced_effects.draw_spectrum_analyzer(beat_intensity, energy)
            
            # Always show text overlay
            self.advanced_effects.add_text_overlay()
            
            # Display help text
            font = pygame.font.SysFont(None, 24)
            help_text = "Keys: 1-5 toggle effects, F1-F5 change styles, +/- adjust style strength, C changes colors, R toggles recording, Space pause/play, ESC quit"
            help_surf = font.render(help_text, True, (255, 255, 255))
            
            # Draw with semi-transparent background
            s = pygame.Surface((help_surf.get_width() + 10, help_surf.get_height() + 10))
            s.set_alpha(128)
            s.fill((0, 0, 0))
            self.base_visualizer.screen.blit(s, (10, self.base_visualizer.height - 40))
            self.base_visualizer.screen.blit(help_surf, (15, self.base_visualizer.height - 35))
            
            # Display recording status if recording
            if self.is_recording:
                # Get recording stats
                stats = self.recorder.get_recording_stats()
                if stats:
                    rec_text = f"RECORDING • {int(stats['duration'])}s • {int(stats['fps'])} FPS"
                    rec_surf = font.render(rec_text, True, (255, 50, 50))
                    
                    # Draw with flashing background based on time
                    flash = int(time.time() * 2) % 2 == 0
                    rec_bg = pygame.Surface((rec_surf.get_width() + 20, rec_surf.get_height() + 10))
                    rec_bg.set_alpha(200 if flash else 150)
                    rec_bg.fill((50, 0, 0) if flash else (30, 0, 0))
                    
                    self.base_visualizer.screen.blit(rec_bg, (self.base_visualizer.width - rec_bg.get_width() - 10, 10))
                    self.base_visualizer.screen.blit(rec_surf, (self.base_visualizer.width - rec_surf.get_width() - 20, 15))
            
            # Update display
            pygame.display.flip()
            
            if self.is_recording:
                self.recorder.capture_frame(self.base_visualizer.screen)
            
            clock.tick(60)  # Cap at 60 FPS
            
        # Stop recording if still active
        if self.is_recording:
            self.stop_recording()
        
        pygame.quit()

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Psychedelic Music Visualizer")
    
    # Required arguments
    parser.add_argument("audio_file", help="Path to the audio file to visualize")
    
    # Optional arguments
    parser.add_argument("--width", type=int, default=1280, help="Window width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Window height (default: 720)")
    parser.add_argument("--fullscreen", action="store_true", help="Run in fullscreen mode")

    parser.add_argument("--style", default="default", help="Visual style (default: default)")
    parser.add_argument("--style-strength", type=float, default=0.7, help="Style strength (0.1-1.0)")
    
    # Effect toggles
    parser.add_argument("--no-particles", action="store_true", help="Disable particle effects")
    parser.add_argument("--no-mandala", action="store_true", help="Disable mandala effects")
    parser.add_argument("--no-ribbons", action="store_true", help="Disable ribbon effects")
    parser.add_argument("--no-fractals", action="store_true", help="Disable fractal effects")
    parser.add_argument("--no-spectrum", action="store_true", help="Disable spectrum analyzer")
    
    # Recording options
    parser.add_argument("--record", action="store_true", help="Automatically start recording")
    parser.add_argument("--output", help="Output video file path (default: auto-generated filename)")
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: File '{args.audio_file}' not found.")
        sys.exit(1)
    
    # Create visualizer with specified dimensions
    visualizer = CompleteVisualizer(args.width, args.height)

    # After creating the visualizer
    visualizer.set_style(args.style, args.style_strength)
    
    # Set initial effect states based on command line args
    if args.no_particles:
        visualizer.show_particles = False
    if args.no_mandala:
        visualizer.show_mandala = False
    if args.no_ribbons:
        visualizer.show_ribbons = False
    if args.no_fractals:
        visualizer.show_fractals = False
    if args.no_spectrum:
        visualizer.show_spectrum = False
    
    # Launch fullscreen if requested
    if args.fullscreen:
        pygame.display.set_mode((args.width, args.height), pygame.FULLSCREEN)
    
    # Run the visualizer
    visualizer.run(args.audio_file, args.record, args.output)