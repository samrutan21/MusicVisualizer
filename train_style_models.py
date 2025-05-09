import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from style_model import StyleTransferModel, ContentLoss, StyleLoss

def train_style_model(style_name, epochs=50, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {style_name} model on {device}")
    
    # Load processed data
    style_path = f"training/styles/{style_name}/processed_data.npy"
    if not os.path.exists(style_path):
        print(f"Error: Processed data not found for {style_name}")
        return
    
    style_data = np.load(style_path)
    style_data = style_data / 255.0  # Normalize to [0,1]
    style_tensor = torch.from_numpy(style_data).float().permute(0, 3, 1, 2)  # NHWC -> NCHW
    
    # Create dataset and dataloader
    dataset = TensorDataset(style_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = StyleTransferModel().to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.transform.parameters(), lr=1e-3)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader)
        for batch in progress_bar:
            images = batch[0].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(images)
            
            # Extract features
            orig_features = model.extract_features(images)
            output_features = model.extract_features(output / 255.0)
            
            # Calculate style loss
            style_loss = 0
            for f_output, f_target in zip(output_features, orig_features):
                style_loss += F.mse_loss(f_output, f_target)
            
            # Calculate content loss (we want to preserve content structure)
            content_loss = F.mse_loss(output, images * 255.0)
            
            # Total loss
            loss = style_loss * 10 + content_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    # Save model
    os.makedirs("models/styles", exist_ok=True)
    torch.save(model.state_dict(), f"models/styles/{style_name}_model.pt")
    print(f"Model saved to models/styles/{style_name}_model.pt")
    
    # Create and save a smaller inference model
    inference_model = nn.Sequential(*list(model.transform.children()))
    torch.save(inference_model.state_dict(), f"models/styles/{style_name}_inference.pt")
    print(f"Inference model saved to models/styles/{style_name}_inference.pt")
    
    # Export to ONNX for faster inference
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    torch.onnx.export(inference_model, dummy_input, 
                     f"models/styles/{style_name}_model.onnx", 
                     opset_version=11)
    print(f"ONNX model saved to models/styles/{style_name}_model.onnx")

if __name__ == "__main__":
    # Train models for each style
    for style in ['cyberpunk', 'industrial', 'nature', 'abstract']:
        model_path = f"models/styles/{style}_inference.pt"
        if os.path.exists(model_path):
            print(f"Model for {style} already exists, skipping...")
        else:
            train_style_model(style)