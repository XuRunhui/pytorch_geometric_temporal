#!/usr/bin/env python3
"""
Example usage script for Stock GNN prediction

This demonstrates how to use the prediction script to load a checkpoint
and generate predictions with metadata tracking.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_prediction_example():
    """Example of how to run the prediction script"""
    
    # Example paths - adjust these to your actual paths
    checkpoint_path = "logs/stock_gnn/version_0/checkpoints/epoch=49-step=1000.ckpt"
    config_path = "config/config.yaml"  # Optional
    output_path = "predictions_with_metadata.csv"
    
    print("🔮 Stock GNN Prediction Example")
    print("=" * 50)
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("\n📋 To create a checkpoint, first train a model:")
        print("   python train_stock_gnn_hydra.py")
        print("\n   Then find the checkpoint in logs/stock_gnn/version_X/checkpoints/")
        return
    
    # Build command
    cmd = [
        "python", "predict.py",
        "--checkpoint", checkpoint_path,
        "--output", output_path
    ]
    
    # Add config if exists
    if config_path and os.path.exists(config_path):
        cmd.extend(["--config", config_path])
    
    print(f"🚀 Running prediction command:")
    print(f"   {' '.join(cmd)}")
    print()
    
    # Run the prediction
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print("📤 Output:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️  Warnings/Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"\n✅ Prediction completed successfully!")
            print(f"📄 Results saved to: {output_path}")
            
            # Show some info about the results
            if os.path.exists(output_path):
                try:
                    import pandas as pd
                    df = pd.read_csv(output_path)
                    print(f"\n📊 Results summary:")
                    print(f"   - Total predictions: {len(df)}")
                    print(f"   - Unique dates: {df['date'].nunique() if 'date' in df.columns else 'N/A'}")
                    print(f"   - Unique stocks: {df['stock_code'].nunique() if 'stock_code' in df.columns else 'N/A'}")
                    print(f"   - Factor columns: {[col for col in df.columns if col.startswith('factor_')]}")
                    print(f"\n📋 First few rows:")
                    print(df.head())
                except ImportError:
                    print("   (Install pandas to see detailed results summary)")
                except Exception as e:
                    print(f"   Failed to analyze results: {e}")
        else:
            print(f"\n❌ Prediction failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"❌ Failed to run prediction: {e}")

def find_latest_checkpoint():
    """Find the latest checkpoint file"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return None
    
    # Look for checkpoints in all version directories
    checkpoint_paths = []
    for version_dir in logs_dir.glob("*/version_*"):
        checkpoints_dir = version_dir / "checkpoints"
        if checkpoints_dir.exists():
            for ckpt_file in checkpoints_dir.glob("*.ckpt"):
                checkpoint_paths.append(ckpt_file)
    
    if not checkpoint_paths:
        return None
    
    # Return the most recent checkpoint
    return max(checkpoint_paths, key=os.path.getmtime)

def main():
    print("🔮 Stock GNN Prediction Example")
    print("=" * 50)
    
    # Try to find a checkpoint automatically
    latest_checkpoint = find_latest_checkpoint()
    
    if latest_checkpoint:
        print(f"📁 Found checkpoint: {latest_checkpoint}")
        
        # Run prediction with the found checkpoint
        cmd = [
            "python", "predict.py",
            "--checkpoint", str(latest_checkpoint),
            "--output", "auto_predictions.csv"
        ]
        
        print(f"🚀 Running: {' '.join(cmd)}")
        print()
        
        try:
            subprocess.run(cmd, check=True)
            print("✅ Prediction completed!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Prediction failed: {e}")
    else:
        print("❌ No checkpoint found.")
        print("\n📋 To create a checkpoint:")
        print("   1. Train a model: python train_stock_gnn_hydra.py")
        print("   2. Find checkpoint in: logs/stock_gnn/version_X/checkpoints/")
        print("   3. Run prediction: python predict.py --checkpoint <path> --output results.csv")

if __name__ == "__main__":
    main()
