import numpy as np
import pandas as pd
import PIL
import pyvips
import os
import time
import random
import gc
import warnings
import logging
import multiprocessing
import argparse
from datetime import timedelta
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, disk

from wsi import filters, tiles, slides

START_TIME = time.time()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--dir_input_tif", type=str, default='data/originals')
parser.add_argument("--file_meta")
parser.add_argument("--dir_output", type=str, default='data/processed')
args = parser.parse_args()

print(os.getcwd())
DIR_INPUT_TIF = args.dir_input_tif
FILE_INPUT_CSV = args.file_meta  # 'data/train_metadata_eRORy1H.csv'
DIR_OUTPUT_TILES = f'./workspace/tiles/{args.dir_output}/'

PAGE_IX_MULS = {0: 32, 1: 16, 2: 8, 3: 4, 4: 2}
DIR_OUTPUT = {
    32: f'{DIR_OUTPUT_TILES}/32/',
    48: f'{DIR_OUTPUT_TILES}/48/',
    64: f'{DIR_OUTPUT_TILES}/64/',
}
PAGES_TO_EXTRACT = {}
PAGES_TO_EXTRACT[32] = [0, 3]
PAGES_TO_EXTRACT[48] = [0, 3]
PAGES_TO_EXTRACT[64] = [0, 3]

for page in PAGES_TO_EXTRACT[32]:
    os.makedirs(f'{DIR_OUTPUT[32]}/{page}', exist_ok=True)

for page in PAGES_TO_EXTRACT[48]:
    os.makedirs(f'{DIR_OUTPUT[48]}/{page}', exist_ok=True)

for page in PAGES_TO_EXTRACT[64]:
    os.makedirs(f'{DIR_OUTPUT[64]}/{page}', exist_ok=True)

slides.SRC_TRAIN_DIR = args.dir_input_tif
RANDOM_STATE = 41


def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


fix_seed(RANDOM_STATE)

logger.info('done initial setup')


def optical_density(tile):
    """
    Convert a tile to optical density values.
    Args:
        tile: A 3D NumPy array of shape (tile_size, tile_size, channels).
    Returns:
        A 3D NumPy array of shape (tile_size, tile_size, channels)
        representing optical density values.
    """
    tile = tile.astype(np.float64)
    return -np.log((tile+1)/240)


def keep_tile(tile, tissue_threshold):
    """
    Determine if a tile should be kept.

    This filters out tiles based on size and a tissue percentage
    threshold, using a custom algorithm. If a tile has height &
    width equal to (tile_size, tile_size), and contains greater
    than or equal to the given percentage, then it will be kept;
    otherwise it will be filtered out.

    Args:
        tile: A 3D NumPy array of shape(tile_size, tile_size, channels).
        tissue_threshold: Tissue percentage threshold.

    Returns:
        A Boolean indicating whether or not a tile should be kept for
        future usage.
    """
    tile_orig = tile

    # Check 1
    # Convert 3D RGB image to 2D grayscale image, from
    # 0 (dense tissue) to 1 (plain background).
    tile = rgb2gray(tile)
    # 8-bit depth complement, from 1 (dense tissue)
    # to 0 (plain background).
    tile = 1 - tile
    # Canny edge detection with hysteresis thresholding.
    # This returns a binary map of edges, with 1 equal to
    # an edge. The idea is that tissue would be full of
    # edges, while background would not.
    tile = canny(tile)
    # Binary closing, which is a dilation followed by
    # an erosion. This removes small dark spots, which
    # helps remove noise in the background.
    tile = binary_closing(tile, disk(10))
    # Binary dilation, which enlarges bright areas,
    # and shrinks dark areas. This helps fill in holes
    # within regions of tissue.
    tile = binary_dilation(tile, disk(10))
    # Fill remaining holes within regions of tissue.
    tile = binary_fill_holes(tile)
    # Calculate percentage of tissue coverage.
    percentage = tile.mean()
    check1 = percentage >= tissue_threshold

    # Check 2
    # Convert to optical density values
    tile = optical_density(tile_orig)
    # Threshold at beta
    beta = 0.15
    tile = np.min(tile, axis=2) >= beta
    # Apply morphology for same reasons as above.
    tile = binary_closing(tile, disk(2))
    tile = binary_dilation(tile, disk(2))
    tile = binary_fill_holes(tile)
    percentage = tile.mean()
    check2 = percentage >= tissue_threshold

    return check1 and check2


def normalize_staining(x, beta=0.15, alpha=1, light_intensity=255):
    """
    Normalize the staining of H&E histology slides.
    This function normalizes the staining of H&E histology slides.
    Args:
        sample_tuple: A (slide_num, sample) tuple, where slide_num is an
            integer, and sample is a 3D NumPy array of shape (H,W,C).
    Returns:
        A (slide_num, sample) tuple, where the sample is a 3D NumPy array
        of shape (H,W,C) that has been stain normalized.
    """
    # Setup.
    h, w, c = x.shape
    x = x.reshape(-1, c).astype(np.float64)  # shape (H*W, C)

    # Reference stain vectors and stain saturations.  We will normalize all slides
    # to these references.  To create these, grab the stain vectors and stain
    # saturations from a desirable slide.

    # Values in reference implementation for use with eigendecomposition approach, natural log,
    # and `light_intensity=240`.
    #stain_ref = np.array([0.5626, 0.2159, 0.7201, 0.8012, 0.4062, 0.5581]).reshape(3,2)
    #max_sat_ref = np.array([1.9705, 1.0308]).reshape(2,1)

    # SVD w/ log10, and `light_intensity=255`.
    stain_ref = (np.array([0.54598845, 0.322116, 0.72385198, 0.76419107, 0.42182333, 0.55879629])
                 .reshape(3,2))
    max_sat_ref = np.array([0.82791151, 0.61137274]).reshape(2,1)

    # Convert RGB to OD.
    # Note: The original paper used log10, and the reference implementation used the natural log.
    # OD = -np.log((x+1)/light_intensity)  # shape (H*W, C)
    OD = -np.log10(x/light_intensity + 1e-8)

    # Remove data with OD intensity less than beta.
    # I.e. remove transparent pixels.
    # Note: This needs to be checked per channel, rather than
    # taking an average over all channels for a given pixel.
    OD_thresh = OD[np.all(OD >= beta, 1), :]  # shape (K, C)

    # Calculate eigenvectors.
    # Note: We can either use eigenvector decomposition, or SVD.
    # eigvals, eigvecs = np.linalg.eig(np.cov(OD_thresh.T))  # np.cov results in inf/nans
    U, s, V = np.linalg.svd(OD_thresh, full_matrices=False)

    # Extract two largest eigenvectors.
    # Note: We swap the sign of the eigvecs here to be consistent
    # with other implementations.  Both +/- eigvecs are valid, with
    # the same eigenvalue, so this is okay.
    # top_eigvecs = eigvecs[:, np.argsort(eigvals)[-2:]] * -1
    top_eigvecs = V[0:2, :].T * -1  # shape (C, 2)

    # Project thresholded optical density values onto plane spanned by
    # 2 largest eigenvectors.
    proj = np.dot(OD_thresh, top_eigvecs)  # shape (K, 2)

    # Calculate angle of each point wrt the first plane direction.
    # Note: the parameters are `np.arctan2(y, x)`
    angles = np.arctan2(proj[:, 1], proj[:, 0])  # shape (K,)

    # Find robust extremes (a and 100-a percentiles) of the angle.
    min_angle = np.percentile(angles, alpha)
    max_angle = np.percentile(angles, 100 - alpha)

    # Convert min/max vectors (extremes) back to optimal stains in OD space.
    # This computes a set of axes for each angle onto which we can project
    # the top eigenvectors.  This assumes that the projected values have
    # been normalized to unit length.
    extreme_angles = np.array(
        [[np.cos(min_angle), np.cos(max_angle)],
            [np.sin(min_angle), np.sin(max_angle)]]
    )  # shape (2,2)
    stains = np.dot(top_eigvecs, extreme_angles)  # shape (C, 2)

    # Merge vectors with hematoxylin first, and eosin second, as a heuristic.
    if stains[0, 0] < stains[0, 1]:
        stains[:, [0, 1]] = stains[:, [1, 0]]  # swap columns

    # Calculate saturations of each stain.
    # Note: Here, we solve
    #    OD = VS
    #     S = V^{-1}OD
    # where `OD` is the matrix of optical density values of our image,
    # `V` is the matrix of stain vectors, and `S` is the matrix of stain
    # saturations.  Since this is an overdetermined system, we use the
    # least squares solver, rather than a direct solve.
    sats, _, _, _ = np.linalg.lstsq(stains, OD.T)

    # Normalize stain saturations to have same pseudo-maximum based on
    # a reference max saturation.
    max_sat = np.percentile(sats, 99, axis=1, keepdims=True)
    sats = sats / max_sat * max_sat_ref

    # Compute optimal OD values.
    OD_norm = np.dot(stain_ref, sats)

    # Recreate image.
    # Note: If the image is immediately converted to uint8 with `.astype(np.uint8)`, it will
    # not return the correct values due to the initital values being outside of [0,255].
    # To fix this, we round to the nearest integer, and then clip to [0,255], which is the
    # same behavior as Matlab.
    # x_norm = np.exp(OD_norm) * light_intensity  # natural log approach
    x_norm = 10**(-OD_norm) * light_intensity - 1e-8  # log10 approach
    x_norm = np.clip(np.round(x_norm), 0, 255).astype(np.uint8)
    x_norm = x_norm.astype(np.uint8)
    x_norm = x_norm.T.reshape(h,w,c)
    return x_norm


def save_tiles_for_page(cur_page, name, image_path, df_tissue_tiles, dir_output, logger, res):
    patch_size = PATCH_SIZES_ACT[cur_page]
    slide = slides.get_slide(slide_path=image_path, page=cur_page, res=res)
    RES_MUL = PAGE_IX_MULS[cur_page]  # 2**(base_page-cur_page)
    for idx, row in df_tissue_tiles.iterrows():
        # generated maximum tiles for page, exit
        if row.tile_id == MAX_TILES_PER_PAGE[cur_page]:
            break
        y = row['Row Start']
        x = row['Col Start']

        if (y < 0 or x < 0):
            warnings.warn(f"bad coords for {name} x:{x} y:{y}", RuntimeWarning)

        x1 = max(0, x) * RES_MUL
        y1 = max(0, y) * RES_MUL

        region_width = region_height = patch_size  # PATCH_SIZES_ACT[cur_page]
        if x1 + region_width > slide.width:
            logger.info(
                f'reducing {name} since {x1} + {region_width} >{slide.width}')
            region_width = slide.width - x1
        if y1 + region_height > slide.height:
            logger.info(
                f'reducing {name} since {y1} + {region_height} >{slide.height}')
            region_height = slide.height - y1
        try:
            # method 2
            region = pyvips.Region.new(slide).fetch(
                x1, y1, region_width, region_height)
            bands = 3
            img = np.ndarray(
                buffer=region,
                dtype=np.uint8,
                shape=(region_height, region_width, bands))

            if keep_tile(img, 0.1):
                img = normalize_staining(img)
                img = PIL.Image.fromarray(img)
                img.save(f'{dir_output}/{cur_page}/{name}_{idx}.jpeg', quality=90)
            print("tile rejected via keep tile fnc")

        except Exception as ex:
            logger.info(
                f'Failed for {name}. x: {x}, y: {y} x1: {x1}, y1: {y1} reg_w: {region_width}, reg_h: {region_height} ')
            logger.info(
                f'slide width: {slide.width} height: {slide.height}  cur_page: {cur_page}')
            logger.info(f'exc: {ex}')
            logger.info(f"{os.popen('df -h').read()}")


def gen_tiles(DIR_INPUT_TIF, dir_output, df_tile_data, pages_to_extract, res):
    ix = -1
    for name, df in list(df_tile_data.groupby('tissue_id')):
        ix += 1
        logger.info(f'processing {ix}')  #: {name}')
        image_path = f'{DIR_INPUT_TIF}/{name}.tif'
        df = df.sort_values(by='tile_id').reset_index(drop=True)
        for page in pages_to_extract:
            save_tiles_for_page(page, name, image_path, df, dir_output, logger, res)


def generate_tiles_for_slide_list(slide_names, dir_output, pages_to_extract, res_list):
    for slide_name, res in zip(slide_names, res_list):
        ##generate tiles
        df = pd.read_csv(f'{slides.TILE_DATA_DIR}/{slide_name}-tile_data.csv',
                         skiprows=14).sort_values(by='Score', ascending=False).reset_index(drop=True)
        # filter scores
        df1 = df[df.Score > 0.1]
        if len(df1) >= 1:
            df = df1
        else:
            logger.info(f'Ignoring Score: {slide_name}')

        df['tile_id'] = df.index
        df['tissue_id'] = slide_name
        df['filename'] = df['tissue_id'] + '.tif'
        gen_tiles(DIR_INPUT_TIF, dir_output, df, pages_to_extract, res)


def multiprocess_generate_tiles(dir_output, pages_to_extract, slides, resols):
    slides_list = slides
    res_list = resols
    num_slides = len(slides_list)

    num_processes = max(multiprocessing.cpu_count(), 5)
    pool = multiprocessing.Pool(num_processes)

    if num_processes > num_slides:
        num_processes = num_slides
    slides_per_process = num_slides / num_processes

    tasks = []
    for num_process in range(1, num_processes + 1):
        start_index = (num_process - 1) * slides_per_process + 1
        end_index = num_process * slides_per_process
        start_index = int(start_index)
        end_index = int(end_index)
        sublist = slides_list[start_index - 1:end_index]
        reslist = res_list[start_index - 1:end_index]
        tasks.append((sublist, dir_output, pages_to_extract, reslist))
        logger.info(f"Task # {num_process} Process slides {list(zip(sublist, reslist))}")

    # start tasks
    results = []
    for t in tasks:
        results.append(pool.apply_async(generate_tiles_for_slide_list, t))

    for result in results:
        _ = result.get()


df_input = pd.read_csv(FILE_INPUT_CSV)
df_input = df_input[df_input.filename.isin([f for f in os.listdir(
    DIR_INPUT_TIF) if f.split('.')[-1] == 'tif'])].reset_index(drop=True)
df_input = df_input[['filename', 'resolution']]
df_input['tissue_id'] = df_input.filename.str.split('.').str[0].values
logger.info('loaded training file')


# Generate tiles
logger.info('************** GENERATING MASKS *********************')

NAMES = [n.split('.')[0] for n in df_input.filename.values]
RESOLUTION = df_input['resolution'].values.tolist()
#IMAGE_DICT = {'names': NAMES, 'resolution': RESOLUTION}
df_submission = pd.DataFrame()

n_files = len(NAMES)
filters.multiprocess_apply_filters_to_images(names=NAMES, ress=RESOLUTION)
elapsed = time.time() - START_TIME
logger.info(
    f'######### DONE GENERATING MASKS ######## TOTAL TIME: {timedelta(seconds=elapsed)}')
gc.collect()

# 48
BASE_SZ = 48
tiles.TILE_SIZE_BASE = BASE_SZ
slides.TOP_TILES_DIR = os.path.join(slides.BASE_DIR, f"top_tiles/{BASE_SZ}")

# maximum number of tiles to extract per page
MAX_TILES_PER_PAGE = {0: 128, 1: 64, 2: 80, 3: 128, 4: 128}
# patch size to extract for each page
PATCH_SIZES_ACT = {0: 1536, 1: 512, 2: 512, 3: 192, 4: 512}

logger.info(f'********* GENERATING TILE META {BASE_SZ} **********')
tiles.multiprocess_filtered_images_to_tiles(
    image_list=NAMES, display=False, save_summary=False, save_data=True, save_top_tiles=False)
elapsed = time.time() - START_TIME
logger.info(
    f'######### DONE GENERATING TILE META {BASE_SZ} ######## TOTAL TIME: {timedelta(seconds=elapsed)}')
gc.collect()

logger.info(f'********* GENERATING TILES {BASE_SZ} **********')

multiprocess_generate_tiles(
    DIR_OUTPUT[BASE_SZ], PAGES_TO_EXTRACT[BASE_SZ], NAMES, RESOLUTION)

elapsed = time.time() - START_TIME
logger.info(
    f'######### DONE GENERATING TILES {BASE_SZ} ######## TOTAL TIME: {timedelta(seconds=elapsed)}')
gc.collect()

# 64
BASE_SZ = 64
tiles.TILE_SIZE_BASE = BASE_SZ
slides.TILE_DATA_DIR = os.path.join(slides.BASE_DIR, f"tile_data/{BASE_SZ}")
slides.TOP_TILES_DIR = os.path.join(slides.BASE_DIR, f"top_tiles/{BASE_SZ}")

# maximum number of tiles to extract per page
MAX_TILES_PER_PAGE = {0: 128, 1: 64, 2: 80, 3: 128, 4: 128}
# patch size to extract for each page
PATCH_SIZES_ACT = {0:2048, 1:512, 2: 512, 3: 256, 4: 512}

logger.info(f'********* GENERATING TILE META {BASE_SZ} **********')
tiles.multiprocess_filtered_images_to_tiles(
    image_list=NAMES, display=False, save_summary=False, save_data=True, save_top_tiles=False)
elapsed = time.time() - START_TIME
logger.info(
    f'######### DONE GENERATING TILE META {BASE_SZ} ######## TOTAL TIME: {timedelta(seconds=elapsed)}')
gc.collect()

logger.info(f'********* GENERATING TILES {BASE_SZ} **********')

multiprocess_generate_tiles(
    DIR_OUTPUT[BASE_SZ], PAGES_TO_EXTRACT[BASE_SZ], NAMES, RESOLUTION)

elapsed = time.time() - START_TIME
logger.info(
    f'######### DONE GENERATING TILES {BASE_SZ} ######## TOTAL TIME: {timedelta(seconds=elapsed)}')
gc.collect()
