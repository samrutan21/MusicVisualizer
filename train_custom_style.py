#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Train a custom visualization style")
    parser.add_argument("--name", required=True, help="Style name")
    parser.add_argument("--images", required=True, help="Directory containing training images")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--output", default="models/styles", help="Output directory for models")
    args = parser.parse_args()
    
    # Check if style name is valid
    if not args.name.isalnum():
        print("Error: Style name must contain only alphanumeric characters")
        return 1
    
    # Check if image directory exists
    if not os.path.isdir(args.images):
        print(f"Error: Image directory '{args.images}' not found")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process training images
    print(f"Processing training images from {args.images}...")
    images = []
    for filename in os.listdir(args.images):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(args.images, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (256, 256))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
    
    if len(images) == 0:
        print("Error: No valid images found in the directory")
        return 1
        
    print(f"Found {len(images)} training images")
    
    # Convert to torch tensors
    images = np.array(images) / 255.0  # Normalize to [0,1]
    image_tensors = torch.from_numpy(images).float().permute(0, 3, 1, 2)  # NHWC -> NCHW
    
    # Create dataset and dataloader
    dataset = TensorDataset(image_tensors)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    model = StyleModel().to(device)
    
    # Train model
    train_model(model, dataloader, args.epochs, device)
    
    # Save model
    output_path = os.path.join(args.output, f"{args.name}_model.pt")
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")
    
    # Save color palette from training images
    colors = extract_color_palette(images)
    save_color_palette(args.name, colors, args.output)
    
    return 0

class StyleModel(nn.Module):
    def __init__(self):
        super(StyleModel, self).__init__()
        # Simple encoder-decoder architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.transformer = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=9, padding=4),
            nn.Tanh(),
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer(x)
        x = self.decoder(x)
        return (x + 1) * 127.5  # Map from [-1,1] to [0,255]

def train_model(model, dataloader, epochs, device):
    """Train the style model"""
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader)
        for batch in progress_bar:
            images = batch[0].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Calculate loss
            # Scale target images to match model output
            targets = images * 255.0
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

def extract_color_palette(images, num_colors=5):
    """Extract a color palette from the training images"""
    # Flatten all images into a list of pixels
    pixels = np.vstack([img.reshape(-1, 3) for img in images])
    
    # Use K-means clustering to find dominant colors
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get the RGB values of the cluster centers
    colors = kmeans.cluster_centers_.astype(int)
    
    return colors

def save_color_palette(style_name, colors, output_dir):
    """Save the extracted color palette"""
    colors_list = colors.tolist()
    
    # Save as Python list
    palette_file = os.path.join(output_dir, "color_palettes.py")
    
    # Check if file exists and read existing content
    if os.path.exists(palette_file):
        with open(palette_file, 'r') as f:
            content = f.read()
            
        # Check if this style already exists
        if f'"{style_name}": [' in content:
            # Update existing entry
            import re
            pattern = f'"{style_name}":\\s*\\[[^\\]]*\\]'
            replacement = f'"{style_name}": {colors_list}'
            content = re.sub(pattern, replacement, content)
        else:
            # Add new entry
            if "PALETTES = {" in content:
                content = content.replace("PALETTES = {", f'PALETTES = {{\n    "{style_name}": {colors_list},')
            else:
                content = f'PALETTES = {{\n    "{style_name}": {colors_list}\n}}\n'
    else:
        # Create new file
        content = f'PALETTES = {{\n    "{style_name}": {colors_list}\n}}\n'
        
    # Write updated content
    with open(palette_file, 'w') as f:
        f.write(content)
        
    print(f"Color palette saved to {palette_file}")

if __name__ == "__main__":
    sys.exit(main())