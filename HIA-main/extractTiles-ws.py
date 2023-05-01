# Development of script by Narmin Ghaffari, Jakob Kather and James Dolezal

# Narmin Ghaffari <narminghaffari23@gmail.com>, Nov 2020
# Jakob Nikolas Kather <jkather@ukaachen.de>, Nov 2020
# James Dolezal <jamesmdolezal@gmail.com>, March 2019

###############################################################################

# Requires: Openslide (https://openslide.org/download/)

import argparse
import json
import os
import pickle
from glob import glob
import tqdm.contrib.concurrent as conc
from os.path import exists, isfile, join

import cv2
import imageio
import numpy as np
import pyvips
import math
import pandas as pd
import shapely.geometry as sg
from PIL import Image

# conda config --add channels conda-forge
# conda install shapely
# It is explained in https://conda-forge.org/


#  HOW TO RUN IT :
# run ----> Configuration per file  -------> command line option ------->
# --px 512 --um 256 --export -s inputPath  --num_threads 8 -o outputPath

###############################################################################

Image.MAX_IMAGE_PIXELS = 100000000000
NUM_THREADS = 8
DEFAULT_JPG_MPP = 0.2494
JSON_ANNOTATION_SCALE = 10

###############################################################################


class AnnotationObject:
    def __init__(self, name):
        self.name = name
        self.coordinates = []

    def add_coord(self, coord):
        self.coordinates.append(coord)

    def scaled_area(self, scale):
        return np.multiply(self.coordinates, 1 / scale)

    def print_coord(self):
        for c in self.coordinates:
            print(c)

    def add_shape(self, shape):
        for point in shape:
            self.add_coord(point)


###############################################################################


class JPGSlide:
    def __init__(self, path, mpp):
        self.loaded_image = imageio.imread(path)
        self.dimensions = (self.loaded_image.shape[1], self.loaded_image.shape[0])
        self.properties = {ops.PROPERTY_NAME_MPP_X: mpp}
        self.level_dimensions = [self.dimensions]
        self.level_count = 1

    def get_thumbnail(self, dimensions):
        return cv2.resize(self.loaded_image, dsize=dimensions, interpolation=cv2.INTER_CUBIC)

    def read_region(self, topleft, level, window):
        return self.loaded_image[
            topleft[1] : topleft[1] + window[1],
            topleft[0] : topleft[0] + window[0],
        ]


###############################################################################


class wsi:
    def __init__(self, path):
        self.path = path
        self.slide = self.open()
        self.width = self.slide.width
        self.height = self.slide.height
        self.dimensions = (self.width, self.height)
        self.level_count = self.slide.get_n_pages()
        self.level_dimensions = [(self.width // (2**i), self.height // (2**i)) for i in range(self.level_count)]
        self.level_downsamples = [2**i for i in range(self.level_count)]

    def open(self, page=0):
        return pyvips.Image.new_from_file(self.path, page=page, access='sequential')

    def get_best_level_for_downsample(self, downsample: int):
        return round(math.log2(downsample))

    def read_region(self, coords, level, shape):
        region = pyvips.Region.new(self.open(page=level)).fetch(coords[0], coords[1], *shape)
        return np.ndarray(buffer=region, dtype=np.uint8, shape=(shape[0], shape[1], 3))


###############################################################################


class SlideReader:
    def __init__(self, path, filetype, export_folder=None, pb=None):
        self.coord = []
        self.annotations = []
        self.export_folder = export_folder
        self.pb = pb
        self.p_id = None
        self.extract_px = None
        self.shape = None
        self.basename = path.replace('.' + path.split('.')[-1], '')
        self.name = self.basename.split('/')[-1]
        self.has_anno = True
        self.annPolys = []
        self.ignoredFiles = []
        self.noMPPFlag = 0
        self.NotAbleToLoad = False

        if filetype in ["svs", "mrxs", 'ndpi', 'scn', 'tif']:
            try:
                self.slide = wsi(path)
            except:
                outputFile.write('Unable to read ' + filetype + ',' + path + '\n')
                self.NotAbleToLoad = True
                return None
        elif filetype == "jpg":
            self.slide = JPGSlide(path, mpp=DEFAULT_JPG_MPP)
        else:
            outputFile.write('Unsupported file type ' + filetype + ',' + path + '\n')
            return None

        thumbs_path = join(export_folder, "thumbs")
        if not os.path.exists(thumbs_path):
            os.makedirs(thumbs_path)

        # Load ROIs if available
        roi_path_csv = self.basename + ".csv"
        roi_path_json = self.basename + ".json"

        if exists(roi_path_csv) and os.path.getsize(roi_path_csv) != 0:
            self.load_csv_roi(roi_path_csv)
        elif exists(roi_path_json) and os.path.getsize(roi_path_json) != 0:
            self.load_json_roi(roi_path_json)
        else:
            self.has_anno = False

        if not self.NotAbleToLoad:
            try:
                self.shape = self.slide.dimensions
                self.filter_dimensions = self.slide.level_dimensions[5]
                self.filter_magnification = self.filter_dimensions[0] / self.shape[0]
                goal_thumb_area = 4096 * 4096
                y_x_ratio = self.shape[1] / self.shape[0]
                thumb_x = math.sqrt(goal_thumb_area / y_x_ratio)
                thumb_y = thumb_x * y_x_ratio
                self.thumb = pyvips.Image.thumbnail(path, thumb_x, height=thumb_y)
                self.thumb_file = thumbs_path + '/' + self.name + '_thumb.jpg'
                self.thumb.write_to_file(self.thumb_file)
            except:
                outputFile.write('Can not Load thumb File' + ',' + path + '\n')
            # try:
            #     if ops.PROPERTY_NAME_MPP_X in self.slide.properties:
            #         self.MPP = float(self.slide.properties[ops.PROPERTY_NAME_MPP_X])
            #     elif 'tiff.XResolution' in self.slide.properties:
            #         self.MPP = 1 / float(self.slide.properties['tiff.XResolution']) * 10000
            #     else:
            #         self.noMPPFlag = 1
            #         outputFile.write('No PROPERTY_NAME_MPP_X' + ',' + path + '\n')
            #         return None
            # except:
            #     self.noMPPFlag = 1
            #     outputFile.write('No PROPERTY_NAME_MPP_X' + ',' + path + '\n')
            #     return None

    def loaded_correctly(self):
        return bool(self.shape)

    def build_generator(
        self, size_px, size_um, stride_div, case_name, tiles_path, category, fileSize, mpp, export=False, augment=False
    ):
        self.extract_px = int(size_um / mpp)
        # stride = int(self.extract_px * stride_div)

        slide_x_size = self.shape[0] - self.extract_px
        slide_y_size = self.shape[1] - self.extract_px

        for y in range(0, (self.shape[1] + 1) - self.extract_px, self.extract_px):
            for x in range(0, (self.shape[0] + 1) - self.extract_px, self.extract_px):
                is_unique = (y % self.extract_px == 0) and (x % self.extract_px == 0)
                self.coord.append([x, y, is_unique])

        self.annPolys = [sg.Polygon(annotation.coordinates) for annotation in self.annotations]

        tile_mask = np.asarray([0 for i in range(len(self.coord))])
        self.tile_mask = None

        def generator():
            for ci in range(len(self.coord)):
                c = self.coord[ci]
                filter_px = int(self.extract_px * self.filter_magnification)
                c_filt = [int(c[0] * self.filter_magnification), int(c[1] * self.filter_magnification), c[2]]
                if filter_px == 0:
                    filter_px = 1

                # Check if the center of the current window lies within any annotation; if not, skip
                if bool(self.annPolys) and not any(
                    annPoly.contains(
                        sg.Point(
                            int(c[0] + self.extract_px / 2),
                            int(c[1] + self.extract_px / 2),
                        )
                    )
                    for annPoly in self.annPolys
                ):
                    continue

                # Read the low-mag level for filter checking
                filter_region = self.slide.read_region(c_filt, 5, [filter_px, filter_px])[
                    :, :, :-1
                ]
                median_brightness = int(sum(np.median(filter_region, axis=(0, 1))))
                if median_brightness > 500:
                    continue

                # Read the region and discard the alpha pixels
                try:
                    region = self.slide.read_region(c, 0, [self.extract_px, self.extract_px])[:, :, 0:3]
                    region = cv2.resize(region, dsize=(size_px, size_px), interpolation=cv2.INTER_CUBIC)
                except Exception as e:
                    print(f'Reading region encoutered exception: {e}')
                    continue

                edge = cv2.Canny(region, 40, 100)
                edge = edge / np.max(edge)
                edge = (np.sum(np.sum(edge)) / (size_px * size_px)) * 100

                if (edge < 4) or np.isnan(edge):
                    continue

                tile_mask[ci] = 1
                coord_label = ci
                unique_tile = c[2]

                if stride_div == 1:
                    exportFlag = export and unique_tile
                else:
                    exportFlag = export

                if exportFlag:
                    imageio.imwrite(join(tiles_path, case_name + '_(' + str(c[0]) + ',' + str(c[1]) + ').jpg'), region)
                    if augment:
                        imageio.imwrite(
                            join(tiles_path, case_name + '_(' + str(c[0]) + ',' + str(c[1]) + ')._aug1.jpg'), np.rot90(region)
                        )
                        imageio.imwrite(
                            join(tiles_path, case_name + '_(' + str(c[0]) + ',' + str(c[1]) + ')._aug2.jpg'), np.flipud(region)
                        )
                        imageio.imwrite(
                            join(tiles_path, case_name + '_(' + str(c[0]) + ',' + str(c[1]) + ')._aug3.jpg'),
                            np.flipud(np.rot90(region)),
                        )
                        imageio.imwrite(
                            join(tiles_path, case_name + '_(' + str(c[0]) + ',' + str(c[1]) + ')._aug4.jpg'), np.fliplr(region)
                        )
                        imageio.imwrite(
                            join(tiles_path, case_name + '_(' + str(c[0]) + ',' + str(c[1]) + ')._aug5.jpg'),
                            np.fliplr(np.rot90(region)),
                        )
                        imageio.imwrite(
                            join(tiles_path, case_name + '_(' + str(c[0]) + ',' + str(c[1]) + ')._aug6.jpg'),
                            np.flipud(np.fliplr(region)),
                        )
                        imageio.imwrite(
                            join(tiles_path, case_name + '_(' + str(c[0]) + ',' + str(c[1]) + ')._aug7.jpg'),
                            np.flipud(np.fliplr(np.rot90(region))),
                        )
                yield region, coord_label, unique_tile

            if self.pb:
                if sum(tile_mask) < 4:
                    outputFile.write('Number of Extracted Tiles < 4 ' + ',' + join(tiles_path, case_name) + '\n')

                print(f'Remained Slides: {str(fileSize)}')
                print('***************************************************************************')

            self.tile_mask = tile_mask

        return generator, slide_x_size, slide_y_size, #stride

    def load_csv_roi(self, path):
        reader = pd.read_csv(path)
        headers = [col.strip() for col in reader.columns]
        if 'X_base' not in headers or 'Y_base' not in headers:
            raise IndexError('Unable to find "X_base" and "Y_base" columns in CSV file.')
        index_x = headers.index('X_base')
        index_y = headers.index('Y_base')
        self.annotations.append(AnnotationObject(f"Object{len(self.annotations)}"))

        for index, row in reader.iterrows():
            if str(row[index_x]).strip() == 'X_base' or str(row[index_y]).strip() == 'Y_base':
                self.annotations.append(AnnotationObject(f"Object{len(self.annotations)}"))
                continue

            x_coord = int(float(row[index_x]))
            y_coord = int(float(row[index_y]))
            self.annotations[-1].add_coord((x_coord, y_coord))

    def load_json_roi(self, path):
        with open(path, "r") as json_file:
            json_data = json.load(json_file)['shapes']
        for shape in json_data:
            area_reduced = np.multiply(shape['points'], JSON_ANNOTATION_SCALE)
            self.annotations.append(AnnotationObject("Object" + len(self.annotations)))
            self.annotations[-1].add_shape(area_reduced)


###############################################################################


class Convoluter:
    def __init__(self, size_px, size_um, stride_div, save_folder='', skipws=False, augment=False):
        self.SLIDES = {}
        self.SIZE_PX = size_px
        self.SIZE_UM = size_um
        self.SAVE_FOLDER = save_folder
        self.STRIDE_DIV = stride_div
        self.AUGMENT = augment
        self.skipws = skipws

    def load_slides(self, slides_array, res_list, directory="None", category="None"):
        self.fileSize = len(slides_array)
        self.iterator = 0
        print(f'TOTAL NUMBER OF SLIDES IN THIS FOLDER : {self.fileSize}')

        for slide, res in zip(slides_array, res_list):
            name = slide.split('.')[:-1]
            name = '.'.join(name)
            name = name.split('/')[-1]
            filetype = slide.split('.')[-1]
            path = slide

            self.SLIDES.update({name: {
                "name": name, "path": path, "type": filetype, "category": category, "resolution": float(res)
            }})

        return self.SLIDES

    def convolute_slides(self):
        '''Parent function to guide convolution across a whole-slide image and execute desired functions.'''
        if not os.path.exists(join(self.SAVE_FOLDER, "BLOCKS")):
            os.makedirs(join(self.SAVE_FOLDER, "BLOCKS"))

        # pool = ThreadPool(NUM_THREADS)
        conc.thread_map(lambda slide: self.export_tiles(self.SLIDES[slide]), self.SLIDES, max_workers=NUM_THREADS)

    def export_tiles(self, slide):
        case_name = slide['name']
        category = slide['category']
        path = slide['path']
        filetype = slide['type']
        res = slide['resolution']
        self.iterator = self.iterator + 1
        whole_slide = SlideReader(path, filetype, self.SAVE_FOLDER)

        if not whole_slide.has_anno and self.skipws:
            return

        if whole_slide.NotAbleToLoad:
            return

        if whole_slide.noMPPFlag:
            return

        tiles_path = whole_slide.export_folder + '/' + "BLOCKS"
        if not os.path.exists(tiles_path):
            os.makedirs(tiles_path)

        tiles_path = f'{tiles_path}/{case_name}-{res}'

        if not os.path.exists(tiles_path):
            os.makedirs(tiles_path)

        counter = len(os.listdir(tiles_path))
        if counter > 6:
            print("Folder already filled")
            print('***************************************************************************')
            return

        gen_slice, _, _ = whole_slide.build_generator(
            self.SIZE_PX,
            self.SIZE_UM,
            self.STRIDE_DIV,
            case_name,
            tiles_path,
            category,
            mpp=res,
            fileSize=self.fileSize - self.iterator,
            export=True,
            augment=self.AUGMENT,
        )
        for tile, coord, unique in gen_slice():
            pass


###############################################################################


def get_args():
    parser = argparse.ArgumentParser(description='The script to generate the tiles for Whole Slide Image (WSI).')
    parser.add_argument(
        '-s', '--slide', help='Path to whole-slide image (SVS or JPG format) or folder of images (SVS or JPG) to analyze.'
    )
    parser.add_argument('-o', '--out', help='Path to directory in which exported images and data will be saved.')
    parser.add_argument('--skipws', type=bool, default=False, help='Shall we use whole slide images?')
    parser.add_argument('--px', type=int, default=512, help='Size of image patches to analyze, in pixels.')
    parser.add_argument('--ov', type=float, default=0.0, help='The Size of overlappig. It can be values between 0 and 1.')
    parser.add_argument('--um', type=float, default=255.3856, help='Size of image patches to analyze, in microns.')
    parser.add_argument('--augment', action="store_true", help='Augment extracted tiles with flipping/rotating.')
    parser.add_argument('--num_threads', type=int, help='Number of threads to use when tessellating.')
    parser.add_argument('--res-dict', type=str, help='Path to a pickled dictionary mapping of each image to its resolutions')

    return parser.parse_args()


###############################################################################

if __name__ == ('__main__'):
    args = get_args()
    if not args.out:
        args.out = args.slide
    if not exists(str(args.out)):
        os.mkdir(str(args.out))
    if args.num_threads:
        NUM_THREADS = args.num_threads
    res_dict = pickle.load(open(str(args.res_dict), 'rb'))

    c = Convoluter(args.px, args.um, args.ov, args.out, augment=args.augment, skipws=args.skipws)

    global outputFile
    with open(os.path.join(args.out, 'report.txt'), 'a', encoding="utf-8") as outputFile:
        outputFile.write('The Features Selected For this Experiment: ' + '\n')
        outputFile.write('InputPath: ' + args.slide + '\n')
        outputFile.write('OutPutPath: ' + args.out + '\n')
        outputFile.write(
            f'Size of image patches to analyze, in pixels: {str(args.px)}'
            + '\n'
        )
        outputFile.write(
            f'Size of image patches to analyze, in microns: {str(args.um)}'
            + '\n'
        )
        outputFile.write(f'Size of overlapping: {str(args.ov)}' + '\n')
        outputFile.write(f'Did we skip WSI: {str(args.skipws)}' + '\n')
        outputFile.write('#########################################################################' + '\n')

        if isfile(args.slide):
            path_sep = os.path.sep
            slide_list = [args.slide.split('/')[-1]]
            slide_dir = '/'.join(args.slide.split('/')[:-1])
            c.load_slides(slide_list, slide_dir)
        else:
            slide_list = []
            res_list = []
            for root, dirs, files in os.walk(args.slide):
                for file in files:
                    if ('.ndpi' in file or '.scn' in file or 'svs' in file or 'tif' in file) and 'csv' not in file:
                        res = res_dict[file]
                        fileType = file.split('.')[-1]
                        if os.path.exists(join(args.out, "BLOCKS")):
                            flist = glob(f'{join(args.out, "BLOCKS")}/{file}*.{fileType}')
                            if flist:
                                break
                        slide_list.append(os.path.join(root, file))
                        res_list.append(res)

            # if os.path.exists(join(args.out, "BLOCKS")):
            #     temp = os.listdir(os.path.join(args.out, 'BLOCKS'))
            #     for item in temp:
            #         for s in slide_list:
            #             if item + '.' + fileType in s:
            #                 slide_list.remove(s)
            c.load_slides(slide_list, res_list)

        pb = c.convolute_slides()
