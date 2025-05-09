import os
import torch
import torch.nn as nn
import cv2
import numpy as np

class StyleManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        
        # Define all color palettes including the default ones
        self.color_palettes = {
            # Default palette collection
            "default_vibrant": [(255, 0, 150), (0, 255, 200), (255, 200, 0), (120, 0, 255)],
            "default_pastel": [(255, 105, 180), (147, 112, 219), (64, 224, 208), (255, 215, 0)],
            "default_rainbow": [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 128, 0), (0, 0, 255), (75, 0, 130), (238, 130, 238)],
            "default_purple": [(50, 0, 80), (100, 0, 160), (200, 0, 200), (255, 50, 200)],
            "default_blue": [(25, 25, 112), (0, 0, 128), (0, 0, 205), (65, 105, 225), (100, 149, 237)],
            
            # Style-specific palettes
            "default": [(255, 0, 150), (0, 255, 200), (255, 200, 0), (120, 0, 255)],  # Same as default_vibrant
            "cyberpunk": [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 255)],
            "industrial": [(100, 100, 100), (50, 50, 50), (200, 150, 100), (30, 40, 50)],
            "nature": [(34, 139, 34), (0, 128, 0), (154, 205, 50), (85, 107, 47)],
            "abstract": [(255, 105, 180), (100, 149, 237), (255, 215, 0), (138, 43, 226)]
        }
        
        # Group the default palettes for easy access
        self.default_palettes = [
            self.color_palettes["default_vibrant"],
            self.color_palettes["default_pastel"],
            self.color_palettes["default_rainbow"],
            self.color_palettes["default_purple"],
            self.color_palettes["default_blue"]
        ]
 
        # Effect parameters per style
        self.effect_params = {
            "default": {
                "num_particles": 50,
                "num_fractals": 5,
                "num_ribbons": 10,
                "mandala_complexity": 1.0,
                "wave_complexity": 1.0
            },
            "cyberpunk": {
                "num_particles": 100,
                "num_fractals": 3,
                "num_ribbons": 15,
                "mandala_complexity": 0.8,
                "wave_complexity": 1.5
            },
            "industrial": {
                "num_particles": 30,
                "num_fractals": 8,
                "num_ribbons": 5,
                "mandala_complexity": 1.2,
                "wave_complexity": 0.7
            },
            "nature": {
                "num_particles": 70,
                "num_fractals": 10,
                "num_ribbons": 12,
                "mandala_complexity": 1.5,
                "wave_complexity": 0.9
            },
            "abstract": {
                "num_particles": 60,
                "num_fractals": 6,
                "num_ribbons": 8,
                "mandala_complexity": 1.3,
                "wave_complexity": 1.2
            }
        }
        
        self.load_models()
        
    def load_models(self):
        """Load all available style models"""
        styles_dir = "models/styles"
        print(f"Looking for style models in {os.path.abspath(styles_dir)}")
        if not os.path.exists(styles_dir):
            print("Style models directory not found. Using default styles only.")
            return
            
        found_models = 0
        for file in os.listdir(styles_dir):
            print(f"Found file: {file}")
            if file.endswith("_inference.pt"):
                style_name = file.split("_")[0]
                try:
                    # Create a simple sequential model
                    model = self.create_inference_model()
                    model.load_state_dict(torch.load(
                        os.path.join(styles_dir, file),
                        map_location=self.device
                    ))
                    model.eval()
                    self.models[style_name] = model
                    print(f"Successfully loaded model for {style_name} style")
                    found_models += 1
                except Exception as e:
                    print(f"Error loading model for {style_name}: {e}")
    
        print(f"Total models loaded: {found_models}")
        print(f"Available styles: {list(self.models.keys())}")
    
    def create_inference_model(self):
        """Create a lightweight model for inference that matches the trained architecture"""
        return nn.Sequential(
            # Initial convolution layers
            nn.Conv2d(3, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Residual blocks
            self._create_residual_block(128),
            self._create_residual_block(128),
            self._create_residual_block(128),
            self._create_residual_block(128),
            self._create_residual_block(128),
            
            # Upsampling
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=9, padding=4),
            nn.Tanh()
        )

    def _create_residual_block(self, channels):
        """Create a residual block for the model"""
        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super(ResidualBlock, self).__init__()
                self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                
            def forward(self, x):
                residual = x
                out = nn.functional.relu(self.conv1(x))
                out = self.conv2(out)
                out = out + residual
                return out
                
        return ResidualBlock(channels)
        
    def apply_style(self, frame, style_name, strength=0.7):
        """Apply a style to a frame"""
        print(f"Style transfer request for: {style_name}")
        
        # If no style or default or model not available, return original
        if style_name == "default":
            print("Default style requested, returning original frame")
            return frame
            
        if style_name not in self.models:
            print(f"Style {style_name} not found in available models: {list(self.models.keys())}")
            return frame
        
        try:
            print(f"Processing frame with {style_name} style model")
            
            # Preprocess frame
            height, width = frame.shape[:2]
            small_frame = cv2.resize(frame, (256, 256))
            small_frame = cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR)
            small_frame = small_frame / 255.0
            
            # Convert to torch tensor
            input_tensor = torch.from_numpy(small_frame).float()
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            # Apply style transfer
            with torch.no_grad():
                output_tensor = self.models[style_name](input_tensor)
                
            # Convert back to numpy
            output_tensor = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_tensor = (output_tensor + 1) * 127.5  # Denormalize
            output_tensor = np.clip(output_tensor, 0, 255).astype(np.uint8)
            output_frame = cv2.cvtColor(output_tensor, cv2.COLOR_BGR2RGB)
            
            # Resize back to original size
            output_frame = cv2.resize(output_frame, (width, height))
            
            # Blend with original based on strength
            blended_frame = cv2.addWeighted(frame, 1 - strength, output_frame, strength, 0)
            
            print(f"Successfully applied {style_name} style")
            return blended_frame
        except Exception as e:
            print(f"Error applying style {style_name}: {e}")
            # Return original frame on error
            return frame
       
    def get_default_palettes(self):
        """Get the list of default palettes for cycling through in default mode"""
        return self.default_palettes
        
    def get_color_palette(self, style_name):
        """Get the color palette for a style"""
        if style_name in self.color_palettes:
            return self.color_palettes[style_name]
        return self.color_palettes["default"]
    
    def get_effect_params(self, style_name):
        """Get the effect parameters for a style"""
        if style_name in self.effect_params:
            return self.effect_params[style_name]
        return self.effect_params["default"]