import sys
 
# setting path
sys.path.append('../MissingDataChallenge_team13')
from skimage.util import img_as_int
import numpy as np
import matplotlib.pyplot as plt
from inpaint_tools import read_file_list
from inpaint_config import InPaintConfig
import argparse
import pathlib
import os
from skimage import io
import cv2 as cv

from PIL import Image
from skimage.morphology import disk, binary_dilation
from skimage.restoration import inpaint
from tqdm import tqdm

args = argparse.ArgumentParser(description='InpaintImages')
config = InPaintConfig(args)
settings = config.settings
input_data_dir = settings["dirs"]["input_data_dir"]
output_data_dir = settings["dirs"]["output_data_dir"]
data_set = settings["data_set"]
model_dir = os.path.join(output_data_dir, "trained_model")

inpainted_result_dir = os.path.join(output_data_dir, f"inpainted_{data_set}")
pathlib.Path(inpainted_result_dir).mkdir(parents=True, exist_ok=True)

file_list = os.path.join(input_data_dir, "data_splits", data_set + ".txt")
file_ids = read_file_list(file_list)

print(f"Inpainting {len(file_ids)} images")

for idx in tqdm(file_ids):
    image_defect = io.imread(os.path.join(input_data_dir, "masked", f"{idx}_stroke_masked.png"))
    mask = io.imread(os.path.join(input_data_dir, "masks", f"{idx}_stroke_mask.png"))
    image_orig = io.imread(os.path.join(input_data_dir, "originals", f"{idx}.jpg"))
    out_image_name = os.path.join(inpainted_result_dir, f"{idx}.png")

    mask0 = Image.fromarray(mask)
    mask0 = np.asarray(mask0.resize((64, 64), Image.BILINEAR))
    image_defect = Image.fromarray(image_defect)
    image_defect = np.asarray(image_defect.resize((64, 64), Image.BILINEAR))
    image_result = inpaint.inpaint_biharmonic(image_defect, mask0, channel_axis=-1)

    image_result = (image_result * 255).astype(np.uint8)
    image_defect = Image.fromarray(image_result)
    image_defect = np.asarray(image_defect.resize((360, 360), Image.BILINEAR))
    io.imsave(out_image_name, image_defect)