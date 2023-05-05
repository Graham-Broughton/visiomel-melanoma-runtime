import glob
import math
import multiprocessing
import os
import re
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np
import PIL
import skimage.io as sk
import pyvips
from PIL import Image

from . import util
from .util import Time

THUMBNAIL_SIZE = 300
THUMBNAIL_EXT = "jpg"

BASE_DIR = "./workspace/"
SRC_TRAIN_DIR = "./data/"
SLIDE_EXT = "tif"
DEST_TRAIN_EXT = "png"
DEST_TRAIN_DIR = os.path.join(BASE_DIR, "training_" + DEST_TRAIN_EXT)
DEST_TRAIN_THUMBNAIL_DIR = os.path.join(BASE_DIR, "training_thumbnail_" + THUMBNAIL_EXT)
BASE_PAGE = 5
SCALE_FACTOR = 1

FILTER_RESULT_TEXT = "filtered"
FILTER_DIR = os.path.join(BASE_DIR, f"filter_{DEST_TRAIN_EXT}")
FILTER_SUFFIX = "filter"  # Example: "filter-"
FILTER_THUMBNAIL_DIR = os.path.join(BASE_DIR, "filter_thumbnail_" + THUMBNAIL_EXT)
FILTER_PAGINATION_SIZE = 50
FILTER_PAGINATE = True
FILTER_HTML_DIR = BASE_DIR

TILE_SUMMARY_DIR = os.path.join(BASE_DIR, f"tile_summary_{DEST_TRAIN_EXT}")
TILE_SUMMARY_ON_ORIGINAL_DIR = os.path.join(BASE_DIR, f"tile_summary_on_original_{DEST_TRAIN_EXT}")
TILE_SUMMARY_SUFFIX = "tile_summary"
TILE_SUMMARY_THUMBNAIL_DIR = os.path.join(BASE_DIR, "tile_summary_thumbnail_" + THUMBNAIL_EXT)
TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(BASE_DIR, "tile_summary_on_original_thumbnail_" + THUMBNAIL_EXT)
TILE_SUMMARY_PAGINATION_SIZE = 50
TILE_SUMMARY_PAGINATE = True
TILE_SUMMARY_HTML_DIR = BASE_DIR

TILE_DATA_DIR = os.path.join(BASE_DIR, "tile_data")
TILE_DATA_SUFFIX = "tile_data"
TILE_DIR = os.path.join(BASE_DIR, f"tiles_{DEST_TRAIN_EXT}")
TILE_SUFFIX = "tile"

TOP_TILES_SUFFIX = "top_tile_summary"
TOP_TILES_DIR = os.path.join(BASE_DIR, f"{TOP_TILES_SUFFIX}_{DEST_TRAIN_EXT}")
TOP_TILES_ON_ORIGINAL_DIR = os.path.join(BASE_DIR, f"{TOP_TILES_SUFFIX}_on_original_{DEST_TRAIN_EXT}")
TOP_TILES_THUMBNAIL_DIR = os.path.join(BASE_DIR, TOP_TILES_SUFFIX + "_thumbnail_" + THUMBNAIL_EXT)
TOP_TILES_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(BASE_DIR, TOP_TILES_SUFFIX + "_on_original_thumbnail_" + THUMBNAIL_EXT)

STATS_DIR = os.path.join(BASE_DIR, "svs_stats")


def get_slide(slide_name=None, slide_path=None, res=0.25, page=BASE_PAGE):
    """
    Open a whole-slide image (*.tif).
    Args:
      slide_name: Name of the slide.
    Returns:
      An skimage object representing a whole-slide image.
    """
    resizing_factor = res / 0.25

    if slide_path is None:
        if abs(1 - resizing_factor) > 0.05:
            image = pyvips.Image.new_from_file(f'{SRC_TRAIN_DIR}/{slide_name}.{SLIDE_EXT}', page=page)
            image = image.resize(resizing_factor)
        else:
            image = pyvips.Image.new_from_file(f'{SRC_TRAIN_DIR}/{slide_name}.{SLIDE_EXT}', page=page)
    elif abs(1 - resizing_factor) > 0.05:
        image = pyvips.Image.new_from_file(slide_path, page=page)
        image = image.resize(resizing_factor)
        return image
    else:
        return pyvips.Image.new_from_file(slide_path, page=page)
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


def get_training_slide_path(slide_number):
    """
    Convert slide number to a path to the corresponding WSI training slide file.
    Example:
    5 -> ../data/training_slides/TUPAC-TR-005.svs
    Args:
    slide_number: The slide number.
    Returns:
    Path to the WSI training slide file.
    """
    dic = pickle.load(open('../../slide_num_dict.pkl', 'rb'))
    padded_sl_num = dic[slide_number]
    return os.path.join(SRC_TRAIN_DIR, f"{padded_sl_num}.{SLIDE_EXT}")


def get_training_image_path(slide_number, large_w=None, large_h=None, small_w=None, small_h=None):
    """
    Convert slide number and optional dimensions to a training image path. If no dimensions are supplied,
    the corresponding file based on the slide number will be looked up in the file system using a wildcard.

    Example:
        5 -> ../data/training_png/TUPAC-TR-005-32x-49920x108288-1560x3384.png

    Args:
        slide_number: The slide number.
        large_w: Large image width.
        large_h: Large image height.
        small_w: Small image width.
        small_h: Small image height.

    Returns:
        Path to the image file.
    """
    dic = pickle.load(open('../../slide_num_dict.pkl', 'rb'))
    padded_sl_num = dic[slide_number]
    if large_w is None and large_h is None and small_w is None and small_h is None:
        wildcard_path = os.path.join(DEST_TRAIN_DIR, padded_sl_num + "*." + DEST_TRAIN_EXT)
        img_path = glob.glob(wildcard_path)[0]
    else:
        img_path = os.path.join(DEST_TRAIN_DIR, padded_sl_num + "-" + str(
            SCALE_FACTOR) + "x-" + str(
            large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(small_h) + "." + DEST_TRAIN_EXT)
    return img_path


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
        + "-r%d-c%d-x%d-y%d-w%d-h%d" % (t.r, t.c, t.o_c_s, t.o_r_s, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s)
        + "."
        + DEST_TRAIN_EXT,
    )


def get_training_thumbnail_path(slide_number, large_w=None, large_h=None, small_w=None, small_h=None):
    """
    Convert slide number and optional dimensions to a training thumbnail path. If no dimensions are
    supplied, the corresponding file based on the slide number will be looked up in the file system using a wildcard.

    Example:
    5 -> ../data/training_thumbnail_jpg/TUPAC-TR-005-32x-49920x108288-1560x3384.jpg

    Args:
    slide_number: The slide number.
    large_w: Large image width.
    large_h: Large image height.
    small_w: Small image width.
    small_h: Small image height.

    Returns:
        Path to the thumbnail file.
    """
    dic = pickle.load(open('../../slide_num_dict.pkl', 'rb'))
    padded_sl_num = dic[slide_number]
    if (
        large_w is not None
        or large_h is not None
        or small_w is not None
        or small_h is not None
    ):
        return os.path.join(
            DEST_TRAIN_THUMBNAIL_DIR,
            f"{padded_sl_num}-{str(SCALE_FACTOR)}x-{str(large_w)}x{str(large_h)}-{str(small_w)}x{str(small_h)}.{THUMBNAIL_EXT}",
        )
    wilcard_path = os.path.join(
        DEST_TRAIN_THUMBNAIL_DIR, f"{padded_sl_num}*.{THUMBNAIL_EXT}"
    )
    return glob.glob(wilcard_path)[0]


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
    return os.path.join(TILE_SUMMARY_DIR, get_tile_summary_image_filename(slide_name))


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
    return os.path.join(TOP_TILES_DIR, get_top_tiles_image_filename(slide_name))


def get_top_tiles_on_original_thumbnail_path(slide_number):
    """
    Convert slide number to a path to a top tiles on original thumbnail file.

    Example:
    5 -> ../data/top_tiles_on_original_thumbnail_jpg/TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.jpg

    Args:
    slide_number: The slide number.

    Returns:
    Path to the top tiles on original thumbnail file.
    """
    if not os.path.exists(TOP_TILES_ON_ORIGINAL_THUMBNAIL_DIR):
        os.makedirs(TOP_TILES_ON_ORIGINAL_THUMBNAIL_DIR)
    return os.path.join(
        TOP_TILES_ON_ORIGINAL_THUMBNAIL_DIR,
        get_top_tiles_image_filename(slide_number, thumbnail=True),
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
    large_x = small_x * SCALE_FACTOR
    large_y = small_y * SCALE_FACTOR
    # large_x = round((large_w / SCALE_FACTOR) / math.floor(large_w / SCALE_FACTOR) * (SCALE_FACTOR * small_x))
    # large_y = round((large_h / SCALE_FACTOR) / math.floor(large_h / SCALE_FACTOR) * (SCALE_FACTOR * small_y))
    return large_x, large_y


def training_slide_to_image(slide_number):
    """
    Convert a WSI training slide to a saved scaled-down image in a format such as jpg or png.

    Args:
        slide_number: The slide number.
    """

    img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(slide_number)

    img_path = get_training_image_path(slide_number, large_w, large_h, new_w, new_h)
    print("Saving image to: " + img_path)
    if not os.path.exists(DEST_TRAIN_DIR):
        os.makedirs(DEST_TRAIN_DIR)
    img.save(img_path)

    thumbnail_path = get_training_thumbnail_path(slide_number, large_w, large_h, new_w, new_h)
    save_thumbnail(img, THUMBNAIL_SIZE, thumbnail_path)


def slide_to_scaled_pil_image(slide_number):
    """
    Convert a WSI training slide to a scaled-down PIL image.

    Args:
    slide_number: The slide number.

    Returns:
    Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
    """
    slide_filepath = get_training_slide_path(slide_number)
    print("Opening Slide #%d: %s" % (slide_number, slide_filepath))
    slide = get_slide(slide_path=slide_filepath, page=0)

    large_w, large_h = slide.width, slide.height
    new_w = math.floor(large_w / SCALE_FACTOR)
    new_h = math.floor(large_h / SCALE_FACTOR)
    level = slide.get_best_level_for_downsample(SCALE_FACTOR)
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
    return img, large_w, large_h, new_w, new_h


def save_thumbnail(pil_img, size, path, display_path=False):
    """
    Save a thumbnail of a PIL image, specifying the maximum width or height of the thumbnail.

    Args:
        pil_img: The PIL image to save as a thumbnail.
        size:  The maximum width or height of the thumbnail.
        path: The path to the thumbnail.
        display_path: If True, display thumbnail path in console.
    """
    max_size = tuple(round(size * d / max(pil_img.size)) for d in pil_img.size)
    img = pil_img.resize(max_size, PIL.Image.BILINEAR)
    if display_path:
        print(f"Saving thumbnail to: {path}")
    dir = os.path.dirname(path)
    if dir != '' and not os.path.exists(dir):
        os.makedirs(dir)
    img.save(path)
