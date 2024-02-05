# This script runs inference on many fragments and composes those outputs into a larger image.
from PIL import Image
from infer import run_on_fragment
import numpy as np
import os
import subprocess
import cv2

Image.MAX_IMAGE_PIXELS = None

# xy values must be > 0
data = {
    # "20231022170901": {
    #     "x": 0,
    #     "y": 0,
    #     "rotate": -95,
    # },
    # "20231106155351": {
    #     "x": 16000,
    #     "y": 6300,
    # },
    "20230929220926": {
        "x": 0,
        "y": 600
    },
    "20231005123336": {
        "x": 23000,
        "y": 0
    },
    "20231007101619": {
        "x": 52000,
        "y": 0
    },
    "20231210121321": {
        "x": 86000,
        "y": 0
    },
    "20231012184423": {
        "x": 99000,
        "y":0
    },
    "20231221180251": {
        "x": 122000,
        "y": 7000
    },
    "20231022170901": {
        "x": 124000,
        "y": 0
    },
    "20231106155351": {
        "x": 135000,
        "y": 7000
    },
    "20231031143852": {
        "x": 150000,
        "y": 7000
    },
    "20230702185753": {
        "x": 160000,
        "y": 0,
        "rotate": 260,
    },
    "20231228000653" : {
        "x": 24000,
        "y": 11000,
    },
    "20231222233538" : {
        "x": 38500,
        "y": 10000,
        "rotate": 170,
        "flip": True,
    },
    "20231224042141" : {
        "x": 46000,
        "y": 11000,
        "rotate": 175,
    }
}

# 141 -> 538 -> 653 right to left

load_from_disk = True
download_fragments = True

fragments = {}

image_path = lambda id: f"Vesuvius-GrandPrize/outputs/vesuvius/pretraining_all/figures/{id}.png"
mask_path = lambda id: f"Vesuvius-GrandPrize/outputs/vesuvius/pretraining_all/figures/{id}_mask.png"

# Download a set of files from the ash2txt server
def download_files(user, password, base_url, custom_id, save_dir, postfix, post_download_rotation=None):
    # Incorporate the custom ID into the URL
    url = base_url + str(custom_id) + (postfix if postfix is not None else "")
    auth_url = f"http://{user}:{password}@{url[7:]}"  # Insert username and password
    
    # Create the rclone command
    cmd = [
        "rclone",
        "copy",
        f"--http-url={auth_url}",
        ":http:",
        save_dir,
        "--max-depth", "1",
        "--no-traverse",
        "--exclude", "*.ppm",
        "--no-update-modtime",
        "--multi-thread-streams", "8",
        "--transfers", "8",
        "-P",
        "--ignore-existing"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully cloned from {url}")
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
        if e.output is not None:
            error_output = e.output.decode('utf-8')  # Decode the error output as UTF-8
            print(f"Command failed with exit code {exit_code}.\nError output:\n{error_output}")
        else:
            print("Command failed, likely couldn't be found")

# Infer each image
for fragment_id, metadata in data.items():
    if download_fragments:
        print("Attempting download of", fragment_id)
        download_files(
            user = "registeredusers",
            password = "only",
            base_url = "http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/",
            custom_id = fragment_id,
            save_dir = f"/home/ubuntu/scroll_data/scroll_inkdetection/dataset_flat/raw_fragments/{fragment_id}",
            postfix = None,
        )

        download_files(
            user = "registeredusers",
            password = "only",
            base_url = f"http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/",
            custom_id = fragment_id,
            save_dir = f"/home/ubuntu/scroll_data/scroll_inkdetection/dataset_flat/raw_fragments/{fragment_id}/layers",
            postfix = "/layers",
        )

    if load_from_disk and os.path.exists(image_path(fragment_id)):
        print("Loading fragment", fragment_id)
        fragment_image = Image.open(image_path(fragment_id))
        fragment_mask = Image.open(mask_path(fragment_id))
    else:
        print("Running fragment", fragment_id)
        fragment_image, fragment_mask = run_on_fragment(fragment_id, use_wandb=False)
        fragment_image = Image.fromarray((fragment_image * 255).astype(np.uint8))
        fragment_mask = Image.fromarray(fragment_mask.astype(np.uint8) * 255)

        # Save
        fragment_image.save(image_path(fragment_id))
        fragment_mask.save(mask_path(fragment_id))

    # Flip the image if needed.
    if metadata.get("flip", False):
        fragment_image = fragment_image.transpose(Image.FLIP_LEFT_RIGHT)
        fragment_mask = fragment_mask.transpose(Image.FLIP_LEFT_RIGHT)

    # Rotate the images if needed. Check if rotation metadata is present, else default to 0
    rotation_angle = metadata.get("rotate", 0)
    if rotation_angle != 0:
        fragment_image = fragment_image.rotate(rotation_angle, expand=True)
        fragment_mask = fragment_mask.rotate(rotation_angle, expand=True)

    fragments[fragment_id] = {"image": fragment_image, "mask": fragment_mask, "metadata": metadata}

# Calculate final image size
max_width = max_height = 0
for fragment in fragments.values():
    img = fragment["image"]
    x, y = fragment["metadata"]["x"], fragment["metadata"]["y"]
    max_width = max(max_width, x + img.width)
    max_height = max(max_height, y + img.height)

# Create final image
final_image = Image.new('RGB', (max_width, max_height), color='black')

# Paste each fragment
for fragment_id, fragment in fragments.items():
    print("pasting", fragment_id)
    img = fragment["image"]
    mask = fragment["mask"]
    x, y = fragment["metadata"]["x"], fragment["metadata"]["y"]
    final_image.paste(img, (x, y), mask=mask)

# Also save a lower res version of the image for easy viewing
scaled_width = final_image.width // 20
scaled_height = final_image.height // 20
scaled_image = final_image.resize((scaled_width, scaled_height), Image.BILINEAR)

scaled_image.save("composition_smaller.png")

print("Saved smaller")

final_image.save("composition.png")

print("Saved larger")