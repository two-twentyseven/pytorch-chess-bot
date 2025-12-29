#!/usr/bin/env python3
"""
Generate a default model with randomly initialized weights for testing.
This creates an untrained baseline model that can be loaded in user_play.py.

Usage: python unit_test.py [output_path]
"""

import sys
import torch
from chess_policy_network import ChessPolicyNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_default_model(output_path: str = "policy_net_default.pt", use_conv: bool = True):
    """
    Create and save a model with random (default) weights.
    
    Args:
        output_path: Path where the model will be saved
        use_conv: Whether to use convolutional layers (default: True to match training_test.py)
    """
    print(f"Creating default model with random weights...")
    print(f"  Architecture: use_conv={use_conv}")
    
    # Initialize model with default (random) weights
    model = ChessPolicyNetwork(use_conv=use_conv).to(device)
    model.eval()
    
    # Save the model state dict
    torch.save(model.state_dict(), output_path)
    print(f"✓ Default model saved successfully to: {output_path}")
    print(f"  Model can be loaded with:")
    print(f"    model = ChessPolicyNetwork(use_conv={use_conv})")
    print(f"    model.load_state_dict(torch.load('{output_path}'))")
    print(f"\n  Test it with: python user_play.py {output_path}")


def main():
    """Main entry point."""
    output_path = "policy_net_default.pt"
    
    # Allow optional output path argument
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    
    # Parse use_conv argument (default to True)
    use_conv = True
    if len(sys.argv) > 2:
        if '--use-conv' in sys.argv:
            idx = sys.argv.index('--use-conv')
            if idx + 1 < len(sys.argv):
                use_conv = sys.argv[idx + 1].lower() in ['true', '1', 'yes']
    
    try:
        create_default_model(output_path, use_conv=use_conv)
    except Exception as e:
        print(f"❌ Error creating default model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()