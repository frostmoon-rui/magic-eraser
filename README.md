# Magic Eraser

A local CPU command-line image repair tool maintained by CacheTide. Paint a mask over an unwanted region and let the Big LaMa model fill that area using its surroundings. The script keeps every unmasked RGB pixel and writes a new PNG.

## Setup

Use Python 3.11 or 3.12 and install the dependencies in a virtual environment.

~~~powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
~~~

Download [big-lama.pt from the upstream model release](https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt). SHA-256 is 344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9. The script checks this hash before loading the model. PyTorch publishes [CPU-specific installation instructions](https://pytorch.org/get-started/locally/) if you prefer a CPU-only wheel.

## Usage

~~~powershell
.venv/Scripts/python erase.py input.png mask.png --model big-lama.pt --output repaired.png
~~~

Use PNG or JPEG for the input and a matching grayscale PNG for the mask. White requests replacement, and black keeps the source pixels. Mask values over 127 count as white. The image and mask must match after EXIF orientation. The tool writes RGB PNG, with a four-megapixel input limit. Input images and masks stay on the machine. Fine details and large occlusions can produce artifacts.

## Lineage and distribution

The accompanying desktop folder contains a Windows installer, an offline executable, example images and a Big LaMa checkpoint. Its checkpoint matches the fingerprint recorded by IOPaint. This repository implements an independent command-line interface around that published checkpoint. The original installer, executable and local examples stay in the desktop folder. Obtain the published checkpoint from its upstream release.

- [LaMa research and original code](https://github.com/advimman/lama).
- [IOPaint model package and implementation](https://github.com/Sanster/IOPaint/blob/main/iopaint/model/lama.py).
