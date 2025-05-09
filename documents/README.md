Psychedelic Music Visualizer
An advanced music visualization tool that creates dynamic, synchronized visual experiences from audio files. This application analyzes music in real-time, generating psychedelic visuals that react to beats, tempo, energy, and spectral characteristics of the audio.
Show Image
Features

Real-time Audio Analysis: Analyzes music to detect beats, tempo, and spectral features
Multiple Visual Effects:

Mandala patterns
Particle systems
Fractal trees
Flowing ribbons
Spectrum analyzer


Visual Styles:

5 built-in styles: Default, Cyberpunk, Industrial, Nature, and Abstract
Custom style training using your own image collections


Recording Capability: Save visualizations as video files
User-friendly Interface: Easy-to-use GUI for selecting music and configuring visualizations
Keyboard Controls: Interactive control during visualization

Requirements

Python 3.7+
Dependencies:

numpy
pygame
librosa
scipy
cv2 (OpenCV)
torch (for style training)
pillow (for GUI)



Installation

Clone this repository:
Copygit clone https://github.com/yourusername/psychedelic-music-visualizer.git
cd psychedelic-music-visualizer

Install dependencies:
Copypip install -r requirements.txt

Launch the application:
Copypython visualizer_launcher.py


Usage
Basic Usage

Launch the application using visualizer_launcher.py
Select a music file (MP3, WAV, or OGG format)
Preview the audio if desired
Configure visualization settings:

Resolution
Visual effects
Style selection


Click "Start Visualizer" to begin

During Visualization

ESC: Exit the visualizer
Space: Pause/play music
Keys 1-5: Toggle individual effects (particles, mandala, ribbons, fractals, spectrum)
F1-F5: Switch between visual styles
+/-: Adjust style strength
C: Change color palette
R: Toggle recording

Creating Custom Styles

Click "Train Custom Style" in the settings panel
Enter a name for your new style
Add 20-100 images that represent your desired style
Set the number of training epochs (more epochs = better results but longer training)
Click "Start Training" and wait for completion
Your new style will appear in the style dropdown menu

Project Structure

visualizer_launcher.py: Main entry point
tkinter_interface.py: GUI implementation
psychedelic_visualizer.py: Core visualization engine
enhanced_visualizer.py: Advanced visual effects
main_visualizer.py: Combined visualizer with all features
style_manager.py: Style management and color palettes
visualizer_recorder.py: Video recording functionality
style_model.py: Neural style transfer models
train_style_models.py: Custom style training

Style Guide
See styles_guide.md for details on visual styles and customization.
Troubleshooting

Audio file not playing: Ensure your audio format is supported by Pygame (MP3, WAV, OGG)
Visualization lag: Try a lower resolution or disable some effects
Style training errors: Make sure you have PyTorch installed and enough disk space
Recording issues: Check that you have write permissions in the output directory

Future Improvements

VR mode support
Real-time microphone input
More visual effects
Multi-monitor support
Performance optimizations

License
MIT License
Acknowledgments

Librosa team for audio analysis capabilities
PyGame community for visualization framework
All open-source contributors