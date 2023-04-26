import glob
import math
import multiprocessing
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import PIL
import skimage.io as sk
import pyvips
from PIL import Image

from . import util
from .util import Time

BASE_DIR = "./workspace/"
SRC_TRAIN_DIR = "./data/"
SLIDE_EXT = "tif"
DEST_TRAIN_EXT = "png"
BASE_PAGE = 5
SCALE_FACTOR = 1

FILTER_RESULT_TEXT = "filtered"
FILTER_DIR = os.path.join(BASE_DIR, f"filter_{DEST_TRAIN_EXT}")

TILE_SUMMARY_DIR = os.path.join(BASE_DIR, f"tile_summary_{DEST_TRAIN_EXT}")
TILE_SUMMARY_ON_ORIGINAL_DIR = os.path.join(
    BASE_DIR, f"tile_summary_on_original_{DEST_TRAIN_EXT}"
)
TILE_SUMMARY_SUFFIX = "tile_summary"

TILE_DATA_DIR = os.path.join(BASE_DIR, "tile_data")
TILE_DATA_SUFFIX = "tile_data"

TOP_TILES_SUFFIX = "top_tile_summary"
TOP_TILES_DIR = os.path.join(BASE_DIR, f"{TOP_TILES_SUFFIX}_{DEST_TRAIN_EXT}")
TOP_TILES_ON_ORIGINAL_DIR = os.path.join(
    BASE_DIR, f"{TOP_TILES_SUFFIX}_on_original_{DEST_TRAIN_EXT}"
)

TILE_DIR = os.path.join(BASE_DIR, f"tiles_{DEST_TRAIN_EXT}")
TILE_SUFFIX = "tile"

STATS_DIR = os.path.join(BASE_DIR, "stats")


def get_slide(slide_name, res):
    """
    Open a whole-slide image (*.tif).
    Args:
      slide_name: Name of the slide.
    Returns:
      An skimage object representing a whole-slide image.
    """
    if res is not None:
        resizing_factor = res / 0.25
        image = pyvips.Image.new_from_file(f'{SRC_TRAIN_DIR}/{slide_name}.{SLIDE_EXT}', page=BASE_PAGE)
        image = image.affine((resizing_factor, 0, 0, resizing_factor))
    else:
        image = pyvips.Image.new_from_file(f'{SRC_TRAIN_DIR}/{slide_name}.{SLIDE_EXT}', page=BASE_PAGE)
    return np.asarray(image)


def open_image(filename):
    """
    Open an image (*.jpg, *.png, etc).
    Args:
      filename: Name of the image file.
    returns:
      A PIL.Image.Image object representing an image.
    """
    return Image.open(filename)


def open_image_np(filename):
    """
    Open an image (*.jpg, *.png, etc) as an RGB NumPy array.
    Args:
      filename: Name of the image file.
    returns:
      A NumPy representing an RGB image.
    """
    pil_img = open_image(filename)
    return util.pil_to_np_rgb(pil_img)


def get_tile_image_path(tile):
    """
    Obtain tile image path based on tile information such as row, column, row pixel position, column pixel position,
    pixel width, and pixel height.
    Args:
      tile: Tile object.
    Returns:
      Path to image tile.
    """
    t = tile
    return os.path.join(
        TILE_DIR,
        t.slide_name,
        f"-{TILE_SUFFIX}"
        + "-r%d-c%d-x%d-y%d-w%d-h%d"
        % (t.r, t.c, t.o_c_s, t.o_r_s, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s)
        + "."
        + DEST_TRAIN_EXT,
    )


def get_tile_summary_image_path(slide_name):
    """
    Convert slide name to a path to a tile summary image file.
    Example:
      5 -> ../data/tile_summary_png/TUPAC-TR-005-tile_summary.png
    Args:
      slide_name: The slide name.
    Returns:
      Path to the tile summary image file.
    """
    if not os.path.exists(TILE_SUMMARY_DIR):
        os.makedirs(TILE_SUMMARY_DIR)
    return os.path.join(
        TILE_SUMMARY_DIR, get_tile_summary_image_filename(slide_name)
    )


def get_tile_summary_image_filename(slide_name, thumbnail=False):
    """
    Convert slide name to a tile summary image file name.
    Returns:
      The tile summary image file name.
    """
    if thumbnail:
        ext = THUMBNAIL_EXT
    else:
        ext = DEST_TRAIN_EXT

    return f"{slide_name}-{TILE_SUMMARY_SUFFIX}.{ext}"


def get_top_tiles_image_filename(slide_name, thumbnail=False):
    """
    Convert slide name to a top tiles image file name.
    Example:
      5, False -> TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.png
      5, True -> TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.jpg
    Args:
      slide_name: The slide name.
      thumbnail: If True, produce thumbnail filename.
    Returns:
      The top tiles image file name.
    """
    if thumbnail:
        ext = THUMBNAIL_EXT
    else:
        ext = DEST_TRAIN_EXT

    return f"{slide_name}-{TOP_TILES_SUFFIX}.{ext}"


def get_top_tiles_image_path(slide_name):
    """
    Convert slide name to a path to a top tiles image file.
    Example:
      5 -> ../data/top_tiles_png/TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.png
    Args:
      slide_name: The slide name.
    Returns:
      Path to the top tiles image file.
    """
    if not os.path.exists(TOP_TILES_DIR):
        os.makedirs(TOP_TILES_DIR)
    return os.path.join(
        TOP_TILES_DIR, get_top_tiles_image_filename(slide_name)
    )


def get_tile_data_filename(slide_name):
    """
    Convert slide name to a tile data file name.
    Example:
      5 -> TUPAC-TR-005-32x-49920x108288-1560x3384-tile_data.csv
    Args:
      slide_name: The slide name.
    Returns:
      The tile data file name.
    """
    return f"{slide_name}-{TILE_DATA_SUFFIX}.csv"


def get_tile_data_path(slide_name):
    """
    Convert slide name to a path to a tile data file.
    Example:
      5 -> ../data/tile_data/TUPAC-TR-005-32x-49920x108288-1560x3384-tile_data.csv
    Args:
      slide_name: The slide name.
    Returns:
      Path to the tile data file.
    """
    if not os.path.exists(TILE_DATA_DIR):
        os.makedirs(TILE_DATA_DIR)
    return os.path.join(TILE_DATA_DIR, get_tile_data_filename(slide_name))


def get_filter_image_result(slide_name):
    """
    Convert slide name to the path to the file that is the final result of filtering.
    Example:
      SLIDE -> filter_png/SLIDE.png
    Args:
      slide_name: The slide name.
    Returns:
      Path to the filter image file.
    """
    return f"{FILTER_DIR}/{slide_name}.{DEST_TRAIN_EXT}"


def small_to_large_mapping(small_pixel, TARGET_PAGE=0):
    """
    Map a scaled-down pixel width and height to the corresponding pixel of the original whole-slide image.
    Args:
      small_pixel: The scaled-down width and height.
      large_dimensions: The width and height of the original whole-slide image.
    Returns:
      Tuple consisting of the scaled-up width and height.
    """
    # **2*BASE_PAGE-TARGET_PAGE
    small_x, small_y = small_pixel
    # large_w, large_h = large_dimensions
    large_x = small_x*SCALE_FACTOR
    large_y = small_y*SCALE_FACTOR
    # large_x = round((large_w / SCALE_FACTOR) / math.floor(large_w / SCALE_FACTOR) * (SCALE_FACTOR * small_x))
    # large_y = round((large_h / SCALE_FACTOR) / math.floor(large_h / SCALE_FACTOR) * (SCALE_FACTOR * small_y))
    return large_x, large_y
