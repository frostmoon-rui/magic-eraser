"""Inpaint a local image with the published Big LaMa TorchScript model."""

import argparse
import hashlib
import io
from pathlib import Path


MODEL_SHA256 = "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9"


def erase(image_path, mask_path, model_path, output):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    if output.suffix.lower() != ".png" or output.exists():
        raise ValueError("Choose a new PNG output file.")
    with model_path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != MODEL_SHA256:
            raise ValueError("The checkpoint does not match the published Big LaMa file.")
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        if image.width * image.height > 4_000_000:
            raise ValueError("Resize the input and mask to at most four megapixels.")
        original = np.asarray(image.convert("RGB"))
    with Image.open(mask_path) as mask:
        selection = np.asarray(ImageOps.exif_transpose(mask).convert("L")) > 127
    height, width = original.shape[:2]
    if selection.shape != (height, width) or not selection.any():
        raise ValueError("The mask must match the image and contain at least one white pixel.")
    pad = ((0, -height % 8), (0, -width % 8))
    pixels = np.pad(original, (*pad, (0, 0)), mode="edge")
    regions = np.pad(selection, pad, mode="edge")
    with torch.inference_mode():
        image_tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float().div(255)
        mask_tensor = torch.from_numpy(regions).unsqueeze(0).unsqueeze(0).float()
        model = torch.jit.load(str(model_path), map_location="cpu").eval()
        prediction = model(image_tensor, mask_tensor)[0, :, :height, :width]
        filled = prediction.permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().numpy()
    result = np.where(selection[..., None], filled, original)
    with io.BytesIO() as encoded:
        Image.fromarray(result).save(encoded, format="PNG")
        with output.open("xb") as destination:
            destination.write(encoded.getbuffer())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        erase(args.image, args.mask, args.model, args.output)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Inpainting failed: {error}\n")


if __name__ == "__main__":
    main()
