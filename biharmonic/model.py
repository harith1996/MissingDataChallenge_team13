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

    # add randomly positioned small point-like defects
    rstate = np.random.default_rng(0)
    for radius in [0, 2, 4]:
        # larger defects are less common
        thresh = 3 + 0.25 * radius  # make larger defects less common
        tmp_mask = rstate.standard_normal(image_orig.shape[:-1]) > thresh
        if radius > 0:
            tmp_mask = binary_dilation(tmp_mask, disk(radius, dtype=bool))
        mask[tmp_mask] = 1

    image_result = inpaint.inpaint_biharmonic(image_defect, mask, channel_axis=-1)
    io.imshow(image_result)
    io.show()
    
    # image_result = (image_result).astype(np.float32)
    # print(image_result)
    io.imsave(out_image_name, image_result)
    # cv.imwrite(out_image_name, img_as_int(image_result))