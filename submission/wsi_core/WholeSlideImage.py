import math
import os
from xml.dom import minidom
import multiprocessing as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyvips
from PIL import Image
from wsi_core.wsi_utils import (
    savePatchIter_bag_hdf5, initialize_hdf5_bag, save_hdf5, screen_coords, isBlackPatch, isWhitePatch, to_percentiles
)
from wsi_core.util_classes import isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard, Contour_Checking_fn
from utils.file_utils import load_pkl, save_pkl
import utils.utils as utils
from utils.utils import iprint
Image.MAX_IMAGE_PIXELS = 933120000


class WholeSlideImage(object):
    def __init__(self, path):

        """
        Args:
            path (str): fullpath to WSI file
        """

        self.name = ".".join(path.split("/")[-1].split('.')[:-1])
        self.wsi = pyvips.Image.new_from_file(path)
        self.level_downsamples = self._assertLevelDownsamples()
        self.level_dim = self.wsi.level_dimensions

        self.contours_tissue = None
        self.contours_tumor = None
        self.hdf5_file = None

    def getOpenSlide(self):
        return self.wsi

    def initXML(self, xml_path):
        def _createContour(coord_list):
            return np.array([[[int(float(coord.attributes['X'].value)),
                               int(float(coord.attributes['Y'].value))]] for coord in coord_list], dtype='int32')

        xmldoc = minidom.parse(xml_path)
        annotations = [anno.getElementsByTagName('Coordinate') for anno in xmldoc.getElementsByTagName('Annotation')]
        self.contours_tumor = [_createContour(coord_list) for coord_list in annotations]
        self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)

    def initTxt(self,annot_path):
        def _create_contours_from_dict(annot):
            all_cnts = []
            for idx, annot_group in enumerate(annot):
                contour_group = annot_group['coordinates']
                if annot_group['type'] == 'Polygon':
                    for idx, contour in enumerate(contour_group):
                        contour = np.array(contour).astype(np.int32).reshape(-1,1,2)
                        all_cnts.append(contour)

                else:
                    for idx, sgmt_group in enumerate(contour_group):
                        contour = []
                        for sgmt in sgmt_group:
                            contour.extend(sgmt)
                        contour = np.array(contour).astype(np.int32).reshape(-1,1,2)
                        all_cnts.append(contour)

            return all_cnts

        with open(annot_path, "r") as f:
            annot = f.read()
            annot = eval(annot)
        self.contours_tumor = _create_contours_from_dict(annot)
        self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)

    def initSegmentation(self, mask_file):
        # load segmentation results from pickle file
        asset_dict = load_pkl(mask_file)
        self.holes_tissue = asset_dict['holes']
        self.contours_tissue = asset_dict['tissue']

    def saveSegmentation(self, mask_file):
        # save segmentation results using pickle
        asset_dict = {'holes': self.holes_tissue, 'tissue': self.contours_tissue}
        save_pkl(mask_file, asset_dict)

    def segmentTissue(
        self, seg_level=0, sthresh=20, sthresh_up=255, mthresh=7, close=0, use_otsu=False,
        filter_params={'a_t':100}, ref_patch_size=512, exclude_ids=[], keep_ids=[]
    ):
        """
            Segment the tissue via HSV -> Median thresholding -> Binary threshold
        """

        def _filter_contours(contours, hierarchy, filter_params):
            """
                Filter contours by: area.
            """
            filtered = []

            # find indices of foreground contours (parent == -1)
            hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)  # lizx: hierarchy_1 contains contours without parents
            all_holes = []

            # loop through foreground contour indices
            for cont_idx in hierarchy_1:
                # actual contour
                cont = contours[cont_idx]
                # indices of holes contained in this contour (children of parent contour)
                holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
                # take contour area (includes holes)
                a = cv2.contourArea(cont)
                # calculate the contour area of each hole
                hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
                # actual area of foreground contour region
                a = a - np.array(hole_areas).sum()
                if a == 0:
                    continue
                if (filter_params['a_t'],) < (a,):
                    filtered.append(cont_idx)
                    all_holes.append(holes)

            foreground_contours = [contours[cont_idx] for cont_idx in filtered]

            hole_contours = []

            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids]
                unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                # take max_n_holes largest holes by area
                unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
                filtered_holes = []

                # filter these holes
                for hole in unfilered_holes:
                    if cv2.contourArea(hole) > filter_params['a_h']:
                        filtered_holes.append(hole)

                hole_contours.append(filtered_holes)

            return foreground_contours, hole_contours

        img = np.array(self.wsi.read_region((0,0), seg_level, self.level_dim[seg_level]))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
        img_med = cv2.medianBlur(img_hsv[:,:,1], mthresh)  # Apply median blurring

        # Thresholding
        if use_otsu:
            _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        else:
            _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

        # Morphological closing
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            # kernel = np.zeros((close, close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

        scale = self.level_downsamples[seg_level]
        scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))
        filter_params = filter_params.copy()
        filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
        filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area

        # Find and filter contours
        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # Find contours
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        if filter_params:
            foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)
            # Necessary for filtering out artifacts
        self.contours_tissue = self.scaleContourDim(foreground_contours, scale)  # lizx: now in level 0 coordinates
        self.holes_tissue = self.scaleHolesDim(hole_contours, scale)  # lizx: now in level 1 coordinates

        # exclude_ids = [0,7,9]
        if len(keep_ids) > 0:
            contour_ids = set(keep_ids) - set(exclude_ids)
        else:
            contour_ids = set(np.arange(len(self.contours_tissue))) - set(exclude_ids)

        self.contours_tissue = [self.contours_tissue[i] for i in contour_ids]
        self.holes_tissue = [self.holes_tissue[i] for i in contour_ids]

    def visWSI(
        self, vis_level=0, color=(0,255,0), hole_color=(0,0,255), annot_color=(255,0,0),
        line_thickness=250, max_size=None, top_left=None, bot_right=None, custom_downsample=1, view_slide_only=False,
        number_contours=False, seg_display=True, annot_display=True
    ):

        downsample = self.level_downsamples[vis_level]
        scale = [1 / downsample[0], 1 / downsample[1]]

        if top_left is not None and bot_right is not None:
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)
        else:
            top_left = (0,0)
            region_size = self.level_dim[vis_level]

        img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))

        if not view_slide_only:
            offset = tuple(-(np.array(top_left) * scale).astype(int))
            line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            if self.contours_tissue is not None and seg_display:
                if not number_contours:
                    cv2.drawContours(img, self.scaleContourDim(self.contours_tissue, scale),
                                     -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)

                else:  # add numbering to each contour
                    for idx, cont in enumerate(self.contours_tissue):
                        contour = np.array(self.scaleContourDim(cont, scale))
                        M = cv2.moments(contour)
                        cX = int(M["m10"] / (M["m00"] + 1e-9))
                        cY = int(M["m01"] / (M["m00"] + 1e-9))
                        # draw the contour and put text next to center
                        cv2.drawContours(img, [contour], -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
                        cv2.putText(
                            img,
                            f"{idx}",
                            (cX, cY),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (255, 0, 0),
                            10,
                        )

                for holes in self.holes_tissue:
                    cv2.drawContours(img, self.scaleContourDim(holes, scale),
                                     -1, hole_color, line_thickness, lineType=cv2.LINE_8)

            if self.contours_tumor is not None and annot_display:
                cv2.drawContours(img, self.scaleContourDim(self.contours_tumor, scale),
                                 -1, annot_color, line_thickness, lineType=cv2.LINE_8, offset=offset)

        img = Image.fromarray(img)

        w, h = img.size
        if custom_downsample > 1:
            img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size / w if w > h else max_size / h
            img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

        return img

    def createPatches_bag_hdf5(
        self, save_path, patch_level=0, patch_size=256, step_size=256, save_coord=True, **kwargs
    ):
        contours = self.contours_tissue

        print("Creating patches for: ", self.name, "...",)
        for idx, cont in enumerate(contours):
            patch_gen = self._getPatchGenerator(cont, idx, patch_level, save_path, patch_size, step_size, **kwargs)

            if self.hdf5_file is None:
                try:
                    first_patch = next(patch_gen)

                # empty contour, continue
                except StopIteration:
                    continue

                file_path = initialize_hdf5_bag(first_patch, save_coord=save_coord)
                self.hdf5_file = file_path

            for patch in patch_gen:
                savePatchIter_bag_hdf5(patch)

        return self.hdf5_file

    def _getPatchGenerator(
        self, cont, cont_idx, patch_level, save_path, patch_size=256, step_size=256, custom_downsample=1,
        white_black=True, white_thresh=15, black_thresh=50, contour_fn='four_pt', use_padding=True
    ):
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else \
            (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])
        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))

        if custom_downsample > 1:
            assert custom_downsample == 2
            target_patch_size = patch_size
            patch_size = target_patch_size * 2
            step_size = step_size * 2
            print(
                f"Custom Downsample: {custom_downsample}, Patching at {patch_size} x {patch_size},"
                f"But Final Patch Size is {target_patch_size} x {target_patch_size}"
            )

        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size * patch_downsample[0], patch_size * patch_downsample[1])

        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]

        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'four_pt_hard':
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'center':
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            elif contour_fn == 'basic':
                cont_check_fn = isInContourV1(contour=cont)
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y + h
            stop_x = start_x + w
        else:
            stop_y = min(start_y + h, img_h - ref_patch_size[1])
            stop_x = min(start_x + w, img_w - ref_patch_size[0])

        count = 0
        for y in range(start_y, stop_y, step_size_y):
            for x in range(start_x, stop_x, step_size_x):

                # point not inside contour and its associated holes
                if not self.isInContours(cont_check_fn, (x,y), self.holes_tissue[cont_idx], ref_patch_size[0]):
                    continue

                count += 1
                patch_PIL = self.wsi.read_region((x,y), patch_level, (patch_size, patch_size)).convert('RGB')
                if custom_downsample > 1:
                    patch_PIL = patch_PIL.resize((target_patch_size, target_patch_size))

                if white_black and (
                    isBlackPatch(np.array(patch_PIL), rgbThresh=black_thresh)
                    or isWhitePatch(np.array(patch_PIL), satThresh=white_thresh)
                ):
                    continue

                yield {
                    'x': x // (patch_downsample[0] * custom_downsample),
                    'y': y // (patch_downsample[1] * custom_downsample),
                    'cont_idx': cont_idx,
                    'patch_level': patch_level,
                    'downsample': self.level_downsamples[patch_level],
                    'downsampled_level_dim': tuple(
                        np.array(self.level_dim[patch_level]) // custom_downsample
                    ),
                    'level_dim': self.level_dim[patch_level],
                    'patch_PIL': patch_PIL,
                    'name': self.name,
                    'save_path': save_path,
                }
        print(f"patches extracted: {count}")

    @staticmethod
    def isInHoles(holes, pt, patch_size):
        return next(
            (
                1
                for hole in holes
                if cv2.pointPolygonTest(
                    hole, (pt[0] + patch_size / 2, pt[1] + patch_size / 2), False
                )
                > 0
            ),
            0,
        )

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
        if cont_check_fn(pt):
            if holes is not None:
                return not WholeSlideImage.isInHoles(holes, pt, patch_size)
            else:
                return 1
        return 0

    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype='int32') for hole in holes] for holes in contours]

    def _assertLevelDownsamples(self):
        level_downsamples = []
        dim_0 = self.wsi.level_dimensions[0]

        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) \
                else level_downsamples.append((downsample, downsample))

        return level_downsamples

    def process_contours(self, save_path, patch_level=0, patch_size=256, step_size=256, **kwargs):
        save_path_hdf5 = os.path.join(save_path, f'{str(self.name)}.h5')
        print("Creating patches for: ", self.name, "...",)
        n_contours = len(self.contours_tissue)
        print("Total number of contours to process: ", n_contours)
        fp_chunk_size = math.ceil(n_contours * 0.05)
        init = True

        for idx, cont in enumerate(self.contours_tissue):
            if (idx + 1) % fp_chunk_size == fp_chunk_size:
                print(f'Processing contour {idx}/{n_contours}')

            asset_dict, attr_dict = self.process_contour(
                cont, self.holes_tissue[idx], patch_level, save_path, patch_size, step_size, **kwargs
            )
            print("********attr_dict: *********")
            print(attr_dict)
            if len(asset_dict) >= 0:
                if init:
                    save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')
                    init = False
                else:
                    save_hdf5(save_path_hdf5, asset_dict, mode='a')

        return self.hdf5_file

    def process_contour(
        self, cont, contour_holes, patch_level, save_path, patch_size=256, step_size=256,
        contour_fn='four_pt', use_padding=True, top_left=None, bot_right=None
    ):
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (
            0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1]
        )

        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size * patch_downsample[0], patch_size * patch_downsample[1])

        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y + h
            stop_x = start_x + w
        else:
            stop_y = min(start_y + h, img_h - ref_patch_size[1] + 1)
            stop_x = min(start_x + w, img_w - ref_patch_size[0] + 1)

        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))

        if bot_right is not None:
            stop_y = min(bot_right[1], stop_y)
            stop_x = min(bot_right[0], stop_x)
        if top_left is not None:
            start_y = max(top_left[1], start_y)
            start_x = max(top_left[0], start_x)

        if bot_right is not None or top_left is not None:
            w, h = stop_x - start_x, stop_y - start_y
            if w <= 0 or h <= 0:
                print("Contour is not in specified ROI, skip")
                return {}, {}
            else:
                print("Adjusted Bounding Box:", start_x, start_y, w, h)

        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'four_pt_hard':
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'center':
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            elif contour_fn == 'basic':
                cont_check_fn = isInContourV1(contour=cont)
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]

        x_range = np.arange(start_x, stop_x, step=step_size_x)
        y_range = np.arange(start_y, stop_y, step=step_size_y)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

        num_workers = mp.cpu_count()
        num_workers = min(num_workers, 4)
        pool = mp.Pool(num_workers)

        iterable = [(coord, contour_holes, ref_patch_size[0], cont_check_fn) for coord in coord_candidates]
        results = pool.starmap(WholeSlideImage.process_coord_candidate, iterable)
        pool.close()
        results = np.array([result for result in results if result is not None])

        print(f'Extracted {len(results)} coordinates')

        if len(results) > 1:
            asset_dict = {'coords': results}

        else:
            asset_dict = {'coords': np.zeros((0,2),dtype=np.int64)}

        attr = {'patch_size':            patch_size,  # To be considered...
                'patch_level':           patch_level,
                'downsample':            self.level_downsamples[patch_level],
                'downsampled_level_dim': tuple(np.array(self.level_dim[patch_level])),
                'level_dim':             self.level_dim[patch_level],
                'name':                  self.name,
                'save_path':             save_path}

        attr_dict = {'coords': attr}
        return asset_dict, attr_dict

    @staticmethod
    def process_coord_candidate(coord, contour_holes, ref_patch_size, cont_check_fn):
        if WholeSlideImage.isInContours(cont_check_fn, coord, contour_holes, ref_patch_size):
            return coord
        else:
            return None

    @staticmethod
    def two_parts_linear_normalization(scores,normalization_method,ident_level=0):
        assert normalization_method in ("min_max","quantile")
        scores_pos_idx = (scores > 0).nonzero()[0]
        scores_neg_idx = (scores < 0).nonzero()[0]
        heatmap_vis_data = {}

        if len(scores_pos_idx) > 0:
            if normalization_method == "min_max":
                scores_pos = scores[scores_pos_idx]
                pos_mn,pos_mx = scores_pos.min(),scores_pos.max()

            elif normalization_method == "quantile":
                scores_pos = scores[scores_pos_idx]
                pos_mn,pos_mx = np.quantile(scores_pos,[0.05,0.95])
                scores_pos = np.clip(scores_pos,pos_mn,pos_mx)

            if pos_mx > pos_mn:
                scores[scores_pos_idx] = (scores_pos - pos_mn) / (pos_mx - pos_mn)
            else:
                scores[scores_pos_idx] = 1
                iprint(f"All positive values are the same ({pos_mn}), setting to 1",ident=ident_level)
            heatmap_vis_data["normalization_max_pos"],heatmap_vis_data["normalization_min_pos"] = pos_mx,pos_mn
        else:
            iprint("No positive values, skipped",ident=ident_level)
            heatmap_vis_data["normalization_max_pos"],heatmap_vis_data["normalization_min_pos"] = None,None

        if len(scores_neg_idx) > 0:
            if normalization_method == "min_max":
                scores_neg = scores[scores_neg_idx]
                neg_mn,neg_mx = scores_neg.min(),scores_neg.max()
            elif normalization_method == "quantile":
                scores_neg = scores[scores_neg_idx]
                neg_mn,neg_mx = np.quantile(scores_neg,[0.05,0.95])
                scores_neg = np.clip(scores[scores_neg_idx],neg_mn,neg_mx)

            if neg_mx > neg_mn:
                scores[scores_neg_idx] = (scores_neg - neg_mx) / (neg_mx - neg_mn)
            else:
                scores[scores_neg_idx] = -1
                iprint(f"All negative values are the same ({neg_mn}), setting to -1",ident=ident_level)

            heatmap_vis_data["normalization_max_neg"],heatmap_vis_data["normalization_min_neg"] = neg_mx,neg_mn
        else:
            iprint("No negative values, skipped",ident=ident_level)

            heatmap_vis_data["normalization_max_neg"],heatmap_vis_data["normalization_min_neg"] = None,None

        return scores,heatmap_vis_data

    def visHeatmap(self,
                   scores, coords, vis_level=-1,
                   value_type="one_part",
                   top_left=None, bot_right=None,
                   patch_size=(256, 256),
                   blank_canvas=False, canvas_color=(220, 20, 50), alpha=0.4,
                   blur=False, overlap=0.0,
                   segment=True, use_holes=True,
                   normalization_method=None,
                   binarize=False, thresh=0.5,
                   max_size=None,
                   custom_downsample=1,
                   return_overlay=False,
                   cmap_1='coolwarm',
                   cmap_2=None,
                   return_PIL=True,
                   return_heatmap_vis_data=False,
                   ident_level=0):

        """
        Args:
            scores (numpy array of float): Attention scores
            coords (numpy array of int, n_patches x 2): Corresponding coordinates (relative to lvl 0)
            vis_level (int): WSI pyramid level to visualize
            patch_size (tuple of int): Patch dimensions (relative to lvl 0)
            blank_canvas (bool): Whether to use a blank canvas to draw the heatmap (vs. using the original slide)
            canvas_color (tuple of uint8): Canvas color
            alpha (float [0, 1]): blending coefficient for overlaying heatmap onto original slide
            blur (bool): apply gaussian blurring
            overlap (float [0 1]): percentage of overlap between neighboring patches (only affect radius of blurring)
            segment (bool): whether to use tissue segmentation contour (must have already called self.segmentTissue such that
                            self.contours_tissue and self.holes_tissue are not None
            use_holes (bool): whether to also clip out detected tissue cavities (only in effect when segment == True)
            convert_to_percentiles (bool): whether to convert attention scores to percentiles
            binarize (bool): only display patches > threshold
            threshold (float): binarization threshold
            max_size (int): Maximum canvas size (clip if goes over)
            custom_downsample (int): additionally downscale the heatmap by specified factor
            cmap (str): name of matplotlib colormap to use
        """
        iprint("=========== visHeatmap ===========", ident=ident_level)
        heatmap_vis_data = dict()
        assert np.issubdtype(scores.dtype, np.floating)
        if vis_level < 0:
            vis_level = self.wsi.get_best_level_for_downsample(32)
            iprint(f"best_level_for_downsample (32): {vis_level}", ident=ident_level + 1)

        downsample = self.level_downsamples[vis_level]
        iprint(f"level_downsamples[{vis_level}]: {downsample}", ident=ident_level + 1)

        scale = [1 / downsample[0], 1 / downsample[1]]  # Scaling from 0 to desired level
        heatmap_vis_data["scale"] = scale

        scores = scores.copy()
        if len(scores.shape) == 2:
            scores = scores.flatten()

        if binarize:
            if thresh < 0:
                threshold = 1.0 / len(scores)

            else:
                threshold = thresh
            iprint(f"binarization is on, threshold = {thresh}",ident=ident_level + 1)

        else:
            threshold = thresh
            iprint(f"binarization is off, threshold = {thresh} (only scores greater than this will be drawn)",
                   ident=ident_level + 1)

        # calculate size of heatmap and filter coordinates/scores outside specified bbox region #####
        if top_left is not None and bot_right is not None:
            scores, coords = screen_coords(scores, coords, top_left, bot_right)
            coords = coords - top_left
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)

        else:
            region_size = self.level_dim[vis_level]
            top_left = (0,0)
            bot_right = self.level_dim[0]
            w, h = region_size

        patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
        coords = np.ceil(coords * np.array(scale)).astype(int)
        iprint('creating heatmap for: ',ident=ident_level + 1)
        iprint('top_left: ', top_left, 'bot_right: ', bot_right,ident=ident_level + 2)
        iprint(f'w: {w}, h: {h}',ident=ident_level + 2)
        iprint('scaled patch size: ', patch_size,ident=ident_level + 2)

        # normalize filtered scores ######
        assert normalization_method in ("convert_to_percentiles","min_max","quantile",None)
        assert value_type in ("one_part","pos_neg_two_parts")
        if value_type == "one_part":
            if normalization_method == "convert_to_percentiles":
                scores = to_percentiles(scores)
                scores /= 100
                heatmap_vis_data["normalization_max"] = 1.0
                heatmap_vis_data["normalization_min"] = 0.0
                iprint("using percentile normalization, the percentiles are normalized to [0,1]", ident=ident_level + 2)
            elif normalization_method == "min_max":
                mn = scores.min()
                mx = scores.max()
                if mx > mn:
                    scores = (scores - mn) / (mx - mn)
                heatmap_vis_data["normalization_max"],heatmap_vis_data["normalization_min"] = mx,mn

                iprint("using min max normalization, the percentiles are normalized to [0,1]", ident=ident_level + 2)
            elif normalization_method == "quantile":
                qs = np.quantile(scores,[0.05,0.95])
                scores = np.clip(scores,qs[0],qs[1])
                mn = scores.min()
                mx = scores.max()
                if mx > mn:
                    scores = (scores - mn) / (mx - mn)
                else:
                    scores[:] = 0
                    iprint(f"All values are the same ({mn}), setting to 0",ident=ident_level + 2)
                heatmap_vis_data["normalization_max"],heatmap_vis_data["normalization_min"] = mx,mn
                iprint("using quantile normalization, the percentiles are normalized to [0,1]", ident=ident_level + 2)
            elif normalization_method is None:
                mn = scores.min()
                mx = scores.max()
                if np.any(scores < 0) or np.any(scores > 1):
                    iprint(
                        f"scores are not within range [0,1] (mn={mn}, mx={mx}) and normalization_method is None,",
                        "this may cause trouble for heatmap",
                        ident=ident_level + 2
                    )
                heatmap_vis_data["normalization_max"] = 1.0
                heatmap_vis_data["normalization_min"] = 0.0
        elif value_type == "pos_neg_two_parts":
            scores_pos_idx = (scores > 0).nonzero()[0]
            scores_neg_idx = (scores < 0).nonzero()[0]
            if normalization_method == "convert_to_percentiles":

                if len(scores_pos_idx) > 0:
                    scores[scores_pos_idx] = to_percentiles(scores[scores_pos_idx]) / 100
                if len(scores_neg_idx) > 0:
                    scores[scores_neg_idx] = -to_percentiles(np.abs(scores[scores_neg_idx])) / 100
                heatmap_vis_data["normalization_min_neg"] = -1
                heatmap_vis_data["normalization_max_neg"] = 0
                heatmap_vis_data["normalization_min_pos"] = 0
                heatmap_vis_data["normalization_max_pos"] = 1

                iprint("using percentile normalization, the percentiles are normalized to [-1,1]",
                       ident=ident_level + 2)

            elif normalization_method == "min_max":

                scores,heatmap_vis_data_ = self.two_parts_linear_normalization(
                    scores,normalization_method="min_max",ident_level=ident_level + 2
                )
                heatmap_vis_data.update(heatmap_vis_data_)
                iprint("using min_max normalization, the percentiles are normalized to [-1,1]", ident=ident_level + 2)

            elif normalization_method == "quantile":

                scores,heatmap_vis_data_ = self.two_parts_linear_normalization(
                    scores,normalization_method="quantile",ident_level=ident_level + 2
                )
                heatmap_vis_data.update(heatmap_vis_data_)
                iprint("using quantile normalization, the percentiles are normalized to [-1,1]", ident=ident_level + 2)

            elif normalization_method is None:
                mn = scores.min()
                mx = scores.max()
                if np.any(scores < -1) or np.any(scores > 1):
                    iprint(
                        f"scores are not within range [-1,1] (mn={mn}, mx={mx}) and normalization_method is None,",
                        "this may cause trouble for heatmap",
                        ident=ident_level + 2
                    )
                heatmap_vis_data["normalization_min_neg"] = -1
                heatmap_vis_data["normalization_max_neg"] = 0
                heatmap_vis_data["normalization_min_pos"] = 0
                heatmap_vis_data["normalization_max_pos"] = 1

        # calculate the heatmap of raw attention scores (before colormap)
        # by accumulating scores over overlapped regions ######

        # heatmap overlay: tracks attention score over each pixel of heatmap
        # overlay counter: tracks how many times attention score is accumulated over each pixel of heatmap
        overlay = np.full(np.flip(region_size), 0).astype(float)
        counter = np.full(np.flip(region_size), 0).astype(np.uint16)
        count = 0
        for idx in range(len(coords)):
            score = scores[idx]
            coord = coords[idx]
            if binarize:
                if score >= threshold:
                    score = 1.0
                    count += 1
                else:
                    score = 0.0
            else:
                if np.abs(score) < threshold:
                    score = 0.0
            # accumulate attention
            overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += score

            # accumulate counter
            counter[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += 1

        if binarize:
            print(f'binarized tiles based on cutoff of {threshold}',ident=ident_level + 1)
            print(f'identified {count}/{len(coords)} patches as positive', ident=ident_level + 1)

        # fetch attended region and average accumulated attention
        zero_mask = counter == 0

        if binarize:
            overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
        else:
            overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
        del counter
        if return_overlay:
            im = overlay
            if return_PIL:
                overlay = utils.normalize8(overlay)
                im = Image.fromarray(overlay)
            return im if not return_heatmap_vis_data else im,heatmap_vis_data

        if blur:
            overlay = cv2.GaussianBlur(overlay,tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1),0)

        if segment:
            tissue_mask = self.get_seg_mask(region_size, scale, use_holes=use_holes, offset=tuple(top_left))
            # return Image.fromarray(tissue_mask) # tissue mask

        if not blank_canvas:
            # downsample original image and use as canvas
            img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
        else:
            # use blank canvas
            img = np.array(Image.new(size=region_size, mode="RGB", color=(255,255,255)))

        # return Image.fromarray(img) #raw image

        iprint('computing heatmap image:', ident=ident_level + 1)
        iprint(f'total of {len(coords)} patches', ident=ident_level + 2)
        twenty_percent_chunk = max(1, int(len(coords) * 0.2))

        if isinstance(cmap_1, str):
            cmap_1 = plt.get_cmap(cmap_1)
        if isinstance(cmap_2,str):
            cmap_2 = plt.get_cmap(cmap_2)
        for idx in range(len(coords)):
            if (idx + 1) % twenty_percent_chunk == 0:
                iprint(f'progress: {idx}/{len(coords)}', ident=ident_level + 2)

            score = scores[idx]
            coord = coords[idx]
            if np.abs(score) >= threshold:

                # attention block
                raw_block = overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]]
                # image block (either blank canvas or orig image)
                img_block = img[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]].copy()

                if value_type == "one_part":
                    raw_block_cmap_mapped = cmap_1(raw_block)
                elif value_type == "pos_neg_two_parts":
                    raw_block_cmap_mapped = cmap_1(raw_block)
                    raw_block_cmap_mapped[raw_block < 0] = cmap_2(np.abs(raw_block[raw_block < 0]))

                # color block (cmap applied to attention block)
                color_block = (raw_block_cmap_mapped * 255)[:,:,:3].astype(np.uint8)

                if segment:
                    # tissue mask block
                    mask_block = tissue_mask[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]]
                    # copy over only tissue masked portion of color block
                    img_block[mask_block] = color_block[mask_block]
                else:
                    # copy over entire color block
                    img_block = color_block

                # rewrite image block
                img[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] = img_block.copy()

        # return Image.fromarray(img) #overlay
        iprint('Done',ident=ident_level + 1)
        del overlay

        if blur:
            img = cv2.GaussianBlur(img,tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1),0)

        if alpha < 1.0:  # The image and overlay are blended in this function
            img = self.block_blending(
                img, vis_level, top_left, bot_right,
                alpha=alpha, blank_canvas=blank_canvas, block_size=1024,ident_level=ident_level + 1
            )

        img = Image.fromarray(img)
        w, h = img.size

        if custom_downsample > 1:
            img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size / w if w > h else max_size / h
            img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

        if return_PIL:
            return img if not return_heatmap_vis_data else img,heatmap_vis_data
        else:
            return np.array(img) if not return_heatmap_vis_data else np.array(img),heatmap_vis_data

    def visHeatmap2(
        self,
        scores,
        coords,
        patch_size=(256,256),
        vis_level=-1
    ):
        if vis_level < 0:
            vis_level = self.wsi.get_best_level_for_downsample(32)
        downsample = self.level_downsamples[vis_level]
        scale = [1 / downsample[0],1 / downsample[1]]
        if len(scores.shape) == 2:
            scores = scores.flatten()

        patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
        coords = np.ceil(coords * np.array(scale)).astype(int)

    def block_blending(self, img, vis_level, top_left, bot_right, alpha=0.5, blank_canvas=False, block_size=1024,ident_level=0):
        iprint('computing blend:',ident=ident_level)
        downsample = self.level_downsamples[vis_level]
        w = img.shape[1]
        h = img.shape[0]
        block_size_x = min(block_size, w)
        block_size_y = min(block_size, h)
        iprint(f'using block size: {block_size_x} x {block_size_y}', ident=ident_level + 1)

        shift = top_left  # amount shifted w.r.t. (0,0)
        for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
            for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):
                # print(x_start, y_start)

                # 1. convert wsi coordinates to image coordinates via shift and scale
                x_start_img = int((x_start - shift[0]) / int(downsample[0]))
                y_start_img = int((y_start - shift[1]) / int(downsample[1]))

                # 2. compute end points of blend tile, careful not to go over the edge of the image
                y_end_img = min(h, y_start_img + block_size_y)
                x_end_img = min(w, x_start_img + block_size_x)

                if y_end_img == y_start_img or x_end_img == x_start_img:
                    continue
                # print('start_coord: {} end_coord: {}'.format((x_start_img, y_start_img), (x_end_img, y_end_img)))

                # 3. fetch blend block and size
                blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img]
                blend_block_size = (x_end_img - x_start_img, y_end_img - y_start_img)

                if not blank_canvas:
                    # 4. read actual wsi block as canvas block
                    pt = (x_start, y_start)
                    canvas = np.array(self.wsi.read_region(pt, vis_level, blend_block_size).convert("RGB"))
                else:
                    # 4. OR create blank canvas block
                    canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255,255,255)))

                # 5. blend color block and canvas block
                img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(
                    blend_block, alpha, canvas, 1 - alpha, 0, canvas
                )
        return img

    def get_seg_mask(self, region_size, scale, use_holes=False, offset=(0,0)):
        print('\ncomputing foreground tissue mask')
        tissue_mask = np.full(np.flip(region_size), 0).astype(np.uint8)
        contours_tissue = self.scaleContourDim(self.contours_tissue, scale)
        offset = tuple((np.array(offset) * np.array(scale) * -1).astype(np.int32))

        contours_holes = self.scaleHolesDim(self.holes_tissue, scale)
        contours_tissue, contours_holes = zip(
            *sorted(zip(contours_tissue, contours_holes), key=lambda x: cv2.contourArea(x[0]), reverse=True)
        )
        for idx in range(len(contours_tissue)):
            cv2.drawContours(
                image=tissue_mask, contours=contours_tissue, contourIdx=idx, color=(1), offset=offset, thickness=-1
            )

            if use_holes:
                cv2.drawContours(
                    image=tissue_mask, contours=contours_holes[idx], contourIdx=-1, color=(0), offset=offset, thickness=-1
                )
            # contours_holes = self._scaleContourDim(self.holes_tissue, scale, holes=True, area_thresh=area_thresh)

        tissue_mask = tissue_mask.astype(bool)
        print(f'detected {tissue_mask.sum()}/{tissue_mask.size} of region as tissue')
        return tissue_mask
