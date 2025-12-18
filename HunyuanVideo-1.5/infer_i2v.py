#!/usr/bin/env python
"""
HunyuanVideo-1.5 Inference Script
Simple wrapper to generate videos from images
"""

import os
import sys
import torch
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="HunyuanVideo-1.5 Video Generation")
    parser.add_argument("--image_path", type=str, default="mint.jpg", help="Input image path")
    parser.add_argument("--prompt", type=str, default="a 3d model of mint, 360 degree turntable rotation", help="Text prompt")
    parser.add_argument("--output_path", type=str, default=None, help="Output video path")
    parser.add_argument("--model_path", type=str, default="./ckpts", help="Model checkpoint path")
    parser.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"], help="Video resolution")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--enable_step_distill", action="store_true", help="Enable step distillation for faster inference")
    parser.add_argument("--cfg_distilled", action="store_true", help="Use CFG distilled model")
    parser.add_argument("--sr", action="store_true", default=False, help="Enable super resolution")
    parser.add_argument("--offloading", action="store_true", default=True, help="Enable model offloading")
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output_path is None:
        output_dir = Path("./outputs")
        output_dir.mkdir(exist_ok=True)
        args.output_path = str(output_dir / "output_i2v.mp4")
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image not found at {args.image_path}")
        print(f"   Expected: {os.path.abspath(args.image_path)}")
        sys.exit(1)
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model path not found at {args.model_path}")
        print(f"   Please download models first: bash download_720p_i2v.sh")
        sys.exit(1)
    
    print("=" * 60)
    print("HunyuanVideo-1.5 Inference")
    print("=" * 60)
    print(f"Image:        {args.image_path}")
    print(f"Prompt:       {args.prompt}")
    print(f"Resolution:   {args.resolution}")
    print(f"Steps:        {args.num_steps}")
    print(f"Seed:         {args.seed}")
    print(f"Output:       {args.output_path}")
    print("=" * 60)
    print()
    
    try:
        # Import HunyuanVideo modules
        print("[1/4] Loading model...")
        from hyvideo.utils.file_utils import load_file_to_bytes
        from hyvideo.models import HYVideo
        from hyvideo.vae import VAE
        from hyvideo.text_encoder import TextEncoder
        from hyvideo.diffusion.pipeline import HYVideoPipeline
        
        print("✓ Modules imported successfully")
        print()
        
        # Initialize pipeline
        print("[2/4] Initializing pipeline...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✓ Using device: {device}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print()
        
        # Build command
        print("[3/4] Preparing generation command...")
        cmd = [
            "python", "generate.py",
            "--image_path", args.image_path,
            "--prompt", args.prompt,
            "--model_path", args.model_path,
            "--resolution", args.resolution,
            "--num_inference_steps", str(args.num_steps),
            "--seed", str(args.seed),
            "--output_path", args.output_path,
        ]
        
        # Add optional flags
        if args.enable_step_distill:
            cmd.append("--enable_step_distill")
        if args.cfg_distilled:
            cmd.append("--cfg_distilled")
        if args.sr:
            cmd.append("--sr")
        if not args.offloading:
            cmd.append("--offloading")
            cmd.append("false")
        
        print("✓ Command prepared")
        print()
        
        # Execute generation
        print("[4/4] Generating video...")
        print("-" * 60)
        
        import subprocess
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        print("-" * 60)
        print()
        
        if result.returncode == 0:
            print("=" * 60)
            print("✓ Video generation completed successfully!")
            print(f"✓ Output saved to: {args.output_path}")
            print("=" * 60)
        else:
            print("❌ Video generation failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠ Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
