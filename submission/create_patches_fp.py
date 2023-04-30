# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df

# other imports
import os
import numpy as np
import time
import argparse
import pandas as pd
import joblib
import tempfile
from pathlib import Path


def stitching(file_path, wsi_object, downscale=64):
    start = time.time()
    heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0, 0, 0), alpha=-1, draw_grid=False)
    total_time = time.time() - start

    return heatmap, total_time


def segment(WSI_object, seg_params, filter_params):
    ### Start Seg Timer
    start_time = time.time()

    # Segment
    WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    ### Stop Seg Timers
    seg_time_elapsed = time.time() - start_time
    return WSI_object, seg_time_elapsed


def patching(WSI_object, **kwargs):
    ### Start Patch Timer
    start_time = time.time()

    # Patch
    file_path = WSI_object.process_contours(**kwargs)

    ### Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def seg_and_patch(
    source,
    save_dir,
    patch_save_dir,
    mask_save_dir,
    stitch_save_dir,
    patch_size=256,
    step_size=256,
    seg_params={
        'seg_level': -1,
        'sthresh': 8,
        'mthresh': 7,
        'close': 4,
        'use_otsu': False,
        'keep_ids': 'none',
        'exclude_ids': 'none',
    },
    filter_params={'a_t': 100, 'a_h': 16, 'max_n_holes': 8},
    vis_params={'vis_level': -1, 'line_thickness': 500},
    patch_params={'use_padding': True, 'contour_fn': 'four_pt'},
    patch_level=0,
    use_default_params=False,
    seg=False,
    save_mask=True,
    stitch=False,
    patch=False,
    auto_skip=True,
    process_list=None,
    use_parallelization=False,
):
    slides = sorted(os.listdir(source))

    slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)

    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)
    mask = df['process'] == 1
    process_stack = df[mask]
    total = len(process_stack)

    legacy_support = 'a' in df.keys()
    if legacy_support:
        print('detected legacy segmentation csv file, legacy support enabled')
        df = df.assign(
            **{
                'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
                'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
                'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
                'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
                'contour_fn': np.full((len(df)), patch_params['contour_fn']),
            }
        )

    seg_times = 0.0
    patch_times = 0.0
    stitch_times = 0.0

    def iter_fn(i, df):
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        print(f"\n\nprogress: {(i / total):.2f}, {i}/{total}")
        print(f'processing {slide}')

        df['process'] = 0
        slide_id, _ = os.path.splitext(slide)

        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            print(f'{slide_id} already exist in destination location, skipped')
            df['status'] = 'already_exist'
            return
        # Inialize WSI
        full_path = os.path.join(source, slide)
        WSI_object = WholeSlideImage(full_path)

        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()

        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}

            for key in vis_params.keys():
                if legacy_support and key == 'vis_level':
                    df[key] = -1
                current_vis_params[key] = df[key]

            for key in filter_params.keys():
                if legacy_support and key == 'a_t':
                    old_area = df['a']
                    seg_level = df['seg_level']
                    scale = WSI_object.level_downsamples[seg_level]
                    adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
                    current_filter_params[key] = adjusted_area
                    df[key] = adjusted_area
                current_filter_params[key] = df[key]

            for key in seg_params.keys():
                if legacy_support and key == 'seg_level':
                    df[key] = -1
                current_seg_params[key] = df[key]

            for key in patch_params.keys():
                current_patch_params[key] = df[key]

        if current_vis_params['vis_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_vis_params['vis_level'] = 0

            else:
                best_level = WSI_object.get_best_level_for_downsample(64)
                current_vis_params['vis_level'] = best_level

        if current_seg_params['seg_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params['seg_level'] = 0

            else:
                best_level = WSI_object.get_best_level_for_downsample(64)
                current_seg_params['seg_level'] = best_level

        keep_ids = str(current_seg_params['keep_ids'])
        if keep_ids not in ['none', ""]:
            str_ids = current_seg_params['keep_ids']
            current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['keep_ids'] = []

        exclude_ids = str(current_seg_params['exclude_ids'])
        if exclude_ids not in ['none', ""]:
            str_ids = current_seg_params['exclude_ids']
            current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['exclude_ids'] = []

        w, h = WSI_object.level_dim[current_seg_params['seg_level']]
        if w * h > 1e8:
            print(f'level_dim {w} x {h} is likely too large for successful segmentation, aborting')
            df['status'] = 'failed_seg'
            return

        df['vis_level'] = current_vis_params['vis_level']
        df['seg_level'] = current_seg_params['seg_level']

        seg_time_elapsed = -1
        if seg:
            WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params)

        if save_mask:
            mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id + '.jpg')
            mask.save(mask_path)

        patch_time_elapsed = -1  # Default time
        if patch:
            current_patch_params.update(
                {'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size, 'save_path': patch_save_dir}
            )

            file_path, patch_time_elapsed = patching(
                WSI_object=WSI_object,
                **current_patch_params,
            )

        stitch_time_elapsed = -1
        if stitch:
            file_path = os.path.join(patch_save_dir, slide_id + '.h5')
            if os.path.isfile(file_path):
                heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
                stitch_path = os.path.join(stitch_save_dir, slide_id + '.jpg')
                heatmap.save(stitch_path)

        print(f"segmentation took {seg_time_elapsed} seconds")
        print(f"patching took {patch_time_elapsed} seconds")
        print(f"stitching took {stitch_time_elapsed} seconds")
        df['status'] = 'processed'

        return seg_time_elapsed, patch_time_elapsed, stitch_time_elapsed, df

    tic = time.time()

    if use_parallelization:
        with joblib.Parallel(n_jobs=min(len(total), 50), backend="loky") as parallel:
            temp_folder = tempfile.mkdtemp()
            filename = os.path.join(temp_folder, 'df.mmap')
            if os.path.exists(filename):
                os.unlink(filename)
            _ = joblib.dump(df.values, filename)
            df_mmap = joblib.load(filename)
            results = parallel([joblib.delayed(iter_fn)(i, df.loc[i].copy()) for i in range(total)])
            for i in range(total):
                res = results[i]
                if res is not None and len(res) == 4:
                    seg_time_elapsed, patch_time_elapsed, stitch_time_elapsed, df_row = res
                    seg_times += seg_time_elapsed
                    patch_times += patch_time_elapsed
                    stitch_times += stitch_time_elapsed
                    df.loc[i] = df_row
    else:
        for i in range(total):
            res = iter_fn(i, df.loc[i].copy())
            if res is not None and len(res) == 4:
                seg_time_elapsed, patch_time_elapsed, stitch_time_elapsed, df_row = res
                seg_times += seg_time_elapsed
                patch_times += patch_time_elapsed
                stitch_times += stitch_time_elapsed
                df.loc[i] = df_row
    toc = time.time()
    print("total_time", toc - tic)

    seg_times /= total
    patch_times /= total
    stitch_times /= total
    df["slide_id"] = df["slide_id"].map(lambda x: os.path.splitext(x)[0])
    df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
    print(f"average segmentation time in s per slide: {seg_times}")
    print(f"average patching time in s per slide: {patch_times}")
    print(f"average stiching time in s per slide: {stitch_times}")

    return seg_times, patch_times


parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--data_dir', type=str, default='dataset_files/visiomel/', help='path to parent folder containing raw wsi image files')
parser.add_argument('--source', type=str, default='dataset_files/visiomel/WSI/')
parser.add_argument('--step_size', type=int, default=1024, help='step_size')
parser.add_argument('--patch_size', type=int, default=1024, help='patch_size')
parser.add_argument('--patch', default=True, action='store_true')
parser.add_argument('--seg', default=True, action='store_true')
parser.add_argument('--stitch', default=True, action='store_true')
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--save_dir', type=str, help='directory to save processed data')
parser.add_argument(
    '--preset', default=None, type=str, help='predefined profile of default segmentation and filter parameters (.csv)'
)
parser.add_argument('--patch_level', type=int, default=0, help='downsample level at which to patch')
parser.add_argument('--process_list', type=str, default=None, help='name of list of images to process with parameters (.csv)')

if __name__ == '__main__':
    args = parser.parse_args()
    DATA = Path(args.data_dir)
    args.save_dir = DATA / f'patches/patches--patch_size_{args.patch_size}'

    patch_save_dir = args.save_dir / 'patches'
    mask_save_dir = args.save_dir / 'masks'
    stitch_save_dir = args.save_dir / 'stitches'

    if args.process_list:
        process_list = os.path.join(args.save_dir, args.process_list)

    else:
        process_list = None

    print('source: ', args.source)
    print('patch_save_dir: ', patch_save_dir)
    print('mask_save_dir: ', mask_save_dir)
    print('stitch_save_dir: ', stitch_save_dir)

    directories = {
        'source': args.source,
        'save_dir': args.save_dir,
        'patch_save_dir': patch_save_dir,
        'mask_save_dir': mask_save_dir,
        'stitch_save_dir': stitch_save_dir,
    }

    for key, val in directories.items():
        print(f"{key}: {val}")
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)

    seg_params = {
        'seg_level': -1,
        'sthresh': 8,
        'mthresh': 7,
        'close': 4,
        'use_otsu': False,
        'keep_ids': 'none',
        'exclude_ids': 'none',
    }
    filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
    vis_params = {'vis_level': -1, 'line_thickness': 250}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if args.preset:
        preset_df = pd.read_csv(os.path.join('presets', args.preset))
        for key in seg_params:
            seg_params[key] = preset_df.loc[0, key]

        for key in filter_params:
            filter_params[key] = preset_df.loc[0, key]

        for key in vis_params:
            vis_params[key] = preset_df.loc[0, key]

        for key in patch_params:
            patch_params[key] = preset_df.loc[0, key]

    parameters = {
        'seg_params': seg_params,
        'filter_params': filter_params,
        'patch_params': patch_params,
        'vis_params': vis_params,
    }

    print(parameters)

    seg_times, patch_times = seg_and_patch(
        **directories,
        **parameters,
        patch_size=args.patch_size,
        step_size=args.step_size,
        seg=args.seg,
        use_default_params=False,
        save_mask=True,
        stitch=args.stitch,
        patch_level=args.patch_level,
        patch=args.patch,
        process_list=process_list,
        auto_skip=not args.no_auto_skip,
        use_parallelization=False,
    )
