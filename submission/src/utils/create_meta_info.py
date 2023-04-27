########################################
# this file is currently only for preparing meta dataset for survival analysis
# modification is required if the project is not survival analysis
########################################

from model.helper import get_filename_extensions
import argparse
import glob
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.append('./')

parser = argparse.ArgumentParser(description='Create meta information')
parser.add_argument('--ffpe-only',
                    action='store_true', default=False,
                    help='keep only ffpe slides')
parser.add_argument('--cancer',
                    type=str, default='COAD',
                    help='Cancer type')
parser.add_argument('--magnification',
                    type=int, default=20,
                    help='magnification level')
parser.add_argument('--patch-size',
                    type=int, default=224,
                    help='size of the extracted patch')
parser.add_argument('--random-seed',
                    type=int, default=1,
                    help='random seed for generating the Nested-CV splits')
parser.add_argument('--outer-fold',
                    type=int, default=5,
                    help='number of outer folds for the Nested-CV splits')
parser.add_argument('--inner-fold',
                    type=int, default=4,
                    help='number of inner folds for the Nested-CV splits')
parser.add_argument('--root',
                    type=str, default='./',
                    help='root directory')
parser.add_argument('--stratify',
                    type=str, default='',
                    help='when spliting the datasets, stratify on which variable')


args = parser.parse_args()
np.random.seed(args.random_seed)

EXT_DATA, EXT_EXPERIMENT, EXT_SPLIT = get_filename_extensions(args)

patch_dir = f'imgs/{args.cancer}/{args.magnification}_{args.patch_size}'

logging_file = (
    f'{args.root}/data/meta_log_surv_{EXT_DATA}_{EXT_EXPERIMENT}.csv'
)
handlers = [logging.FileHandler(
    logging_file, mode='w'), logging.StreamHandler()]
logging.basicConfig(format='%(message)s',
                    level=logging.INFO, handlers=handlers)

for arg, value in sorted(vars(args).items()):
    logging.info(f"Argument {arg}: {value}")


##################################################
# The basic survival information
##################################################

def parse_json(cancer, root_dir):
    fname = os.path.join(root_dir, f'data/meta_files/TCGA-{cancer}.json')
    with open(fname) as f:
        d = json.load(f)
    results = [
        [
            d[i]['demographic']['age_at_index'],
            d[i]['demographic']['race'],
            d[i]['demographic']['gender'],
            d[i]['demographic']['vital_status'],
            d[i]['demographic'].get('days_to_death', -1),
            d[i]['demographic']['submitter_id'],
            d[i]['case_id'],
            d[i]['diagnoses'][0]['tumor_stage'],
            d[i]['diagnoses'][0]['tissue_or_organ_of_origin'],
            d[i]['diagnoses'][0]['days_to_last_follow_up'],
            d[i]['diagnoses'][0].get('ajcc_pathologic_m', 'NA'),
            d[i]['diagnoses'][0].get('ajcc_pathologic_t', 'NA'),
            d[i]['diagnoses'][0].get('ajcc_pathologic_n', 'NA'),
            d[i]['diagnoses'][0].get('primary_diagnosis', '--'),
        ]
        for i in range(len(d))
        if len(d[i]) >= 4
    ]
    df = pd.DataFrame(results)
    df.columns = [
        'age_at_index',
        'race',
        'gender',
        'vital_status',
        'days_to_death',
        'submitter_id',
        'case_id',
        'tumor_stage',
        'tissue_or_organ_of_origin',
        'days_to_last_follow_up',
        'ajcc_pathologic_m',
        'ajcc_pathologic_t',
        'ajcc_pathologic_n',
        'primary_diagnosis']
    df.submitter_id = df.submitter_id.apply(lambda x: x.split('_')[0])

    logging.info(df.shape)
    logging.info(df.submitter_id.unique().shape)

    # preparing the survival outcome
    # filtering out patients without follow-up information

    df['time'] = 0
    df.loc[df.vital_status == 'Alive', 'time'] = df[df.vital_status == 'Alive'].days_to_last_follow_up / 365
    df.loc[df.vital_status == 'Dead', 'time'] = [
        np.NaN if x == '--' else int(x) / 365 for x in df[df.vital_status == 'Dead'].days_to_death.to_list()
    ]
    df['time'] = df.time - df.time.min() + 0.01

    df['status'] = 0
    df.loc[df.vital_status == 'Dead', 'status'] = 1
    df = df.loc[~df.time.isna()].copy().reset_index(drop=True)
    logging.info(
        f'number of participants after excluding missing time {df.shape[0]}'
    )
    logging.info(df.describe())
    return df[['case_id', 'submitter_id', 'vital_status', 'days_to_death', 'time', 'status']]


def get_patch_meta(patch_dir, ext, root_dir):

    # obtaining the list of the extracted patches
    if os.path.exists(f'{root_dir}/data/patches_meta_raw_{ext}.pickle'):
        df_cmb = pd.read_pickle(f'{root_dir}/data/patches_meta_raw_{ext}.pickle')
    else:
        df_cmb = _extracted_from_get_patch_meta_8(patch_dir, root_dir, ext)
    df_cmb_meta = df_cmb.drop_duplicates('file_original').copy()
    df_cmb_meta['slide_type'] = df_cmb_meta.file_original.apply(
        lambda x: x.split('-')[3])

    # if FFPE slide is available
    df_cmb_meta['ffpe_slide'] = 0
    df_cmb_meta.loc[df_cmb_meta.slide_type.isin(
        ['01Z', '02Z']), 'ffpe_slide'] = 1
    df_cmb = df_cmb.merge(df_cmb_meta[[
                          'file_original', 'slide_type', 'ffpe_slide']], on='file_original', how='inner')
    df_cmb['ffpe_slide_avail'] = df_cmb.groupby(
        'submitter_id').ffpe_slide.transform(max)

    if args.ffpe_only:
        df_cmb = df_cmb.loc[df_cmb.ffpe_slide_avail == 1].reset_index(drop=True)
    logging.info(
        f"Number of patients in the final dataset: {len(df_cmb.submitter_id.unique())}"
    )

    return df_cmb


def _extracted_from_get_patch_meta_8(patch_dir, root_dir, ext):
    patch_files = glob.glob(f'{patch_dir}/*/*/*/*.jpg')
    logging.info(f"Number of patch files: {len(patch_files)}")
    result = pd.DataFrame(columns=['file'])
    result['file'] = patch_files
    result['file_original'] = result.file.apply(lambda x: x.split('/')[-3])
    result['submitter_id'] = result.file_original.apply(lambda x: x[:12])
    result.to_pickle(f'{root_dir}/data/patches_meta_raw_{ext}.pickle')

    return result


# split the dataset into train-validation-test
def random_split_by_id(df_cmb, df_meta, ext, splits='32', root_dir='./'):
    vars_to_keep = ['submitter_id', 'stratify_var']

    if args.stratify:
        df_meta['stratify_var'] = df_meta[args.stratify]
    else:
        df_meta['stratify_var'] = np.random.randint(0, 2, df_meta.shape[0])

    df = df_cmb[['submitter_id']].merge(
        df_meta[vars_to_keep], on='submitter_id', how='inner')
    df = df.dropna()

    df_id = df.drop_duplicates('submitter_id').reset_index(
        drop=True).copy()[vars_to_keep]
    logging.info(f"Total number of patients: {df_id.shape[0]}")
    kf_outer = StratifiedKFold(
        args.outer_fold, random_state=args.random_seed, shuffle=True)

    df_id['split'] = 0
    df_id.reset_index(drop=True, inplace=True)

    for i, (tv_index, test_index) in enumerate(kf_outer.split(df_id, df_id['stratify_var'])):
        # outer loop
        logging.info("-" * 40)
        df_dev = df_id.loc[df_id.index.isin(tv_index)].reset_index(drop=True)
        df_test = df_id.loc[df_id.index.isin(test_index)]
        logging.info(
            f"Working on outer split {i} .... Dev: {df_dev.shape[0]}; Test: {df_test.shape[0]}"
        )

        df_test[['submitter_id', 'split']].merge(
            df_meta, on='submitter_id', how='inner'
        ).to_pickle(f'{root_dir}/data/meta_test_{EXT_SPLIT}_s-{i}.pickle')

        # inner loop
        kf_inner = StratifiedKFold(
            args.inner_fold, random_state=i, shuffle=True)
        for j, (train_index, val_index) in enumerate(kf_inner.split(df_dev, df_dev['stratify_var'])):
            df_train = df_dev.loc[df_dev.index.isin(train_index)]
            df_val = df_dev.loc[df_dev.index.isin(val_index)]
            logging.info("Working on inner split %02d .... Train: %s; Val: %s" % (
                (i * 10 + j), df_train.shape[0], df_val.shape[0]))

            df_train[['submitter_id', 'split']].\
                merge(df_meta, on='submitter_id', how='inner').\
                to_pickle('%s/data/meta_train_%s_s-%02d.pickle' %
                          (root_dir, EXT_SPLIT, i * 10 + j))
            df_val[['submitter_id', 'split']].\
                merge(df_meta, on='submitter_id', how='inner').\
                to_pickle('%s/data/meta_val_%s_s-%02d.pickle' %
                          (root_dir, EXT_SPLIT, i * 10 + j))


if __name__ == '__main__':
    # the standard approach
    ext = f'{EXT_DATA}_{EXT_EXPERIMENT}'

    # process meta information
    fname_meta = f'data/meta_clinical_{args.cancer}.csv'
    if os.path.isfile(fname_meta):
        df_meta = pd.read_csv(fname_meta)
        print(df_meta.head())
    else:
        df_meta = parse_json(args.cancer, args.root)
        df_meta.to_csv(fname_meta, index=False)
    logging.info(df_meta.describe())

    # process patch information
    patch_meta_file = f'{args.root}/data/patches_meta_{EXT_DATA}.pickle'
    if os.path.exists(patch_meta_file):
        logging.info(f"patch meta file {patch_meta_file} already exists!")
        df_cmb = pd.read_pickle(patch_meta_file)
    else:
        df_cmb = get_patch_meta(patch_dir, EXT_DATA, args.root)
        df_cmb.to_pickle(patch_meta_file)

    # obtain random splits
    random_split_by_id(df_cmb, df_meta, ext, args.root)
