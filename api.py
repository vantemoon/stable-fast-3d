import os
import io
import re
import time
import base64
import tempfile
import zipfile
from contextlib import nullcontext
from functools import lru_cache

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch
import rembg

import sf3d.utils as sf3d_utils
from sf3d.system import SF3D

app = Flask(__name__)
CORS(app)

COND_WIDTH = 512
COND_HEIGHT = 512
COND_DISTANCE = 1.6
COND_FOVY_DEG = 40
BACKGROUND_COLOR = [0.5, 0.5, 0.5]

# Cached values for conditioning
c2w_cond = sf3d_utils.default_cond_c2w(COND_DISTANCE)
intrinsic, intrinsic_normed_cond = sf3d_utils.create_intrinsic_from_fov_deg(
    COND_FOVY_DEG, COND_HEIGHT, COND_WIDTH
)

# Initialize rembg session and SF3D model
rembg_session = rembg.new_session()
device = sf3d_utils.get_device()
model = SF3D.from_pretrained(
    "stabilityai/stable-fast-3d",
    config_name="config.yaml",
    weight_name="model.safetensors",
)
model.eval()
model = model.to(device)

# Temporary storage for generated files
generated_files = []

# ------------------
# Helper Functions
# ------------------

def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=fmt)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def load_image_from_request(field: str = "image") -> Image.Image:
    if field not in request.files:
        raise ValueError("No image file provided")
    file = request.files[field]
    image = Image.open(file.stream).convert("RGBA")
    return image

def create_batch(input_image: Image.Image) -> dict:
    img_resized = input_image.resize((COND_WIDTH, COND_HEIGHT))
    img_array = np.asarray(img_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).float().clip(0, 1)
    mask_cond = img_tensor[:, :, -1:]
    rgb_cond = torch.lerp(torch.tensor(BACKGROUND_COLOR)[None, None, :],
                          img_tensor[:, :, :3],
                          mask_cond)
    batch_elem = {
        "rgb_cond": rgb_cond,
        "mask_cond": mask_cond,
        "c2w_cond": c2w_cond.unsqueeze(0),
        "intrinsic_cond": intrinsic.unsqueeze(0),
        "intrinsic_normed_cond": intrinsic_normed_cond.unsqueeze(0),
    }
    return {k: v.unsqueeze(0) for k, v in batch_elem.items()}

@lru_cache(maxsize=32)
def checkerboard(squares: int, size: int, min_value: float = 0.5):
    base = np.zeros((squares, squares)) + min_value
    base[1::2, ::2] = 1
    base[::2, 1::2] = 1
    repeat_mult = size // squares
    return (
        base.repeat(repeat_mult, axis=0)
        .repeat(repeat_mult, axis=1)[:, :, None]
        .repeat(3, axis=-1)
    )

# ------------------
# Workflow Functions
# ------------------

def remove_background_func(input_image: Image.Image) -> Image.Image:
    return rembg.remove(input_image, session=rembg_session)


def square_crop(input_image: Image.Image) -> Image.Image:
    min_size = min(input_image.size)
    left = (input_image.size[0] - min_size) // 2
    top = (input_image.size[1] - min_size) // 2
    right = (input_image.size[0] + min_size) // 2
    bottom = (input_image.size[1] + min_size) // 2
    return input_image.crop((left, top, right, bottom)).resize((COND_WIDTH, COND_HEIGHT))


def resize_foreground(image: Image.Image, ratio: float) -> Image.Image:
    image_np = np.array(image)
    if image_np.shape[-1] != 4:
        raise ValueError("Expected image with alpha channel (RGBA)")
    alpha = np.where(image_np[..., 3] > 0)
    y1, y2, x1, x2 = alpha[0].min(), alpha[0].max(), alpha[1].min(), alpha[1].max()
    # crop the foreground
    fg = image_np[y1:y2, x1:x2]
    # pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    # compute new size based on ratio
    new_size = int(new_image.shape[0] / ratio)
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    new_image = Image.fromarray(new_image, mode="RGBA").resize((COND_WIDTH, COND_HEIGHT))
    return new_image


def show_mask_img(input_image: Image.Image) -> Image.Image:
    img_np = np.array(input_image)
    alpha = img_np[:, :, 3] / 255.0
    chkb = checkerboard(32, 512) * 255
    new_img = img_np[..., :3] * alpha[:, :, None] + chkb * (1 - alpha[:, :, None])
    return Image.fromarray(new_img.astype(np.uint8), mode="RGB")


def create_batch(input_image: Image.Image) -> dict:
    # Resize and normalize input image
    img_resized = input_image.resize((COND_WIDTH, COND_HEIGHT))
    img_array = np.asarray(img_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).float().clip(0, 1)
    mask_cond = img_tensor[:, :, -1:]
    rgb_cond = torch.lerp(torch.tensor(BACKGROUND_COLOR)[None, None, :], img_tensor[:, :, :3], mask_cond)
    batch_elem = {
        "rgb_cond": rgb_cond,
        "mask_cond": mask_cond,
        "c2w_cond": c2w_cond.unsqueeze(0),
        "intrinsic_cond": intrinsic.unsqueeze(0),
        "intrinsic_normed_cond": intrinsic_normed_cond.unsqueeze(0),
    }
    batched = {k: v.unsqueeze(0) for k, v in batch_elem.items()}
    return batched


def run_model_func(input_image: Image.Image, remesh_option: str,
                   vertex_count: int, texture_size: int) -> dict:
    start = time.time()
    with torch.no_grad():
        context = torch.autocast(device_type=device, dtype=torch.float16) if "cuda" in device else nullcontext()
        with context:
            model_batch = create_batch(input_image)
            model_batch = {k: v.to(device) for k, v in model_batch.items()}
            trimesh_mesh, _ = model.generate_mesh(model_batch, texture_size, remesh_option, vertex_count)
            mesh = trimesh_mesh[0]

    # Create a temporary output directory
    output_dir = tempfile.mkdtemp()
    i = 0  # single output case
    output_subdir = os.path.join(output_dir, str(i))
    os.makedirs(output_subdir, exist_ok=True)

    # Define file paths for OBJ, MTL, and texture files
    out_mesh_path = os.path.join(output_subdir, "mesh.obj")
    out_mtl_path = os.path.join(output_subdir, "material.mtl")
    jpeg_texture_path = os.path.join(output_subdir, "material_0.jpeg")
    png_texture_path = os.path.join(output_subdir, "texture.png")

    # Export the mesh as OBJ (include normals)
    mesh.export(out_mesh_path, include_normals=True)

    # If a JPEG texture was created, convert it to PNG
    if os.path.exists(jpeg_texture_path):
        texture_img = Image.open(jpeg_texture_path)
        texture_img = texture_img.convert("RGBA")
        texture_img.save(png_texture_path, format="PNG")
        os.remove(jpeg_texture_path)

    # If a MTL file exists, update its contents to refer to the PNG texture
    if os.path.exists(out_mtl_path):
        with open(out_mtl_path, "r") as f:
            mtl_content = f.read()
        mtl_content = re.sub(r"map_Kd\s+material_0\.jpeg", "map_Kd texture.png", mtl_content)
        with open(out_mtl_path, "w") as f:
            f.write(mtl_content)

    print(f"Saved OBJ: {out_mesh_path}")
    print(f"Converted Texture: {png_texture_path}")
    print(f"Updated MTL: {out_mtl_path}")
    print("Generation took:", time.time() - start, "s")

    # Return the paths of the output files
    return {
        "obj": out_mesh_path,
        "mtl": out_mtl_path,
        "png": png_texture_path,
        "output_subdir": output_subdir
    }

# ------------------
# API Endpoints
# ------------------

@app.route("/remove_background", methods=["POST"])
def remove_background_endpoint():
    """
    Endpoint to remove the background from the input image.
    This endpoint:
     1. Removes the background using rembg.
     2. Performs a center square crop.
     3. Resizes the foreground using the provided foreground_ratio.
     4. Returns the square-cropped image, resized foreground, and a mask preview.
    """
    try:
        foreground_ratio = float(request.form.get("foreground_ratio", 0.85))
        input_image = load_image_from_request()
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Step 1: Remove background
    no_bg_img = remove_background_func(input_image)
    # Step 2: Square crop the image
    cropped_img = square_crop(no_bg_img)
    # Step 3: Resize foreground
    foreground_img = resize_foreground(cropped_img, foreground_ratio)
    # Step 4: Create mask preview image
    mask_preview = show_mask_img(foreground_img)

    return jsonify({
        "cropped_image": pil_to_base64(cropped_img),
        "foreground_image": pil_to_base64(foreground_img),
        "mask_preview": pil_to_base64(mask_preview)
    })


@app.route("/run_model", methods=["POST"])
def run_model_endpoint():
    try:
        remesh_option = request.form.get("remesh_option", "none").lower()
        vertex_count = int(request.form.get("vertex_count", -1))
        texture_size = int(request.form.get("texture_size", 1024))
        input_image = load_image_from_request()
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    try:
        output_files = run_model_func(input_image, remesh_option, vertex_count, texture_size)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Zip the output files (OBJ, MTL, PNG) for download
    zip_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    with zipfile.ZipFile(zip_temp, 'w') as zipf:
        for file_path in [output_files.get("obj"), output_files.get("mtl"), output_files.get("png")]:
            if file_path and os.path.exists(file_path):
                zipf.write(file_path, arcname=os.path.basename(file_path))
    zip_temp.close()
    resp = send_file(zip_temp.name, as_attachment=True, download_name="output.zip", mimetype="application/zip")
    resp.headers["Output-Folder"] = output_files.get("output_subdir", "")
    resp.headers["Access-Control-Expose-Headers"] = "Output-Folder"

    torch.cuda.empty_cache()

    return resp

# ------------------
# Main
# ------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
