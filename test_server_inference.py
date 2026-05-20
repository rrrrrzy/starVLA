"""Minimal test: load model on a given GPU, run one inference, check if it crashes."""
import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"[GPU {args.gpu}] CUDA_VISIBLE_DEVICES={args.gpu}", flush=True)
    print(f"[GPU {args.gpu}] torch.cuda.is_available()={torch.cuda.is_available()}", flush=True)

    # Import after setting CUDA_VISIBLE_DEVICES
    from starVLA.model.framework.base_framework import baseframework
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    print(f"[GPU {args.gpu}] Loading framework...", flush=True)
    wrapper = PolicyServerWrapper(ckpt_path=args.ckpt, device="cuda", use_bf16=True)
    print(f"[GPU {args.gpu}] Framework loaded.", flush=True)

    # Build a dummy input (2 images + language)
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    examples = [{
        "image": [img, img],
        "lang": "take the blue block and rotate it to the right",
    }]

    print(f"[GPU {args.gpu}] Running inference...", flush=True)
    try:
        result = wrapper.predict_action(examples=examples, unnorm_key="franka")
        print(f"[GPU {args.gpu}] SUCCESS! actions shape={result['actions'].shape}", flush=True)
    except Exception as e:
        print(f"[GPU {args.gpu}] FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
