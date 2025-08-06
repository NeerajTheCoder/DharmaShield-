#!/usr/bin/env python3
"""Convert PyTorch models to Google AI Edge format"""

import ai_edge_torch
import torch

def convert_gemma_to_edge():
    """Convert Gemma PyTorch model to Google AI Edge TFLite"""
    
    # Load PyTorch model
    model = torch.load("models/gemma3n_e2b.pth")
    
    # Convert to Google AI Edge format
    edge_model = ai_edge_torch.convert(model, sample_inputs)
    edge_model.export("models/gemma3n_e2b.tflite")
    
    print("✅ Converted to Google AI Edge format: gemma3n_e2b.tflite")

if __name__ == "__main__":
    convert_gemma_to_edge()

