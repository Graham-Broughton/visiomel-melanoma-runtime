import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips as pv
import tensorflow as tf

#from loguru import logger
from sklearn.preprocessing import OneHotEncoder

#logger.remove(0)
# logger.add()

TFRECORD_IMAGE_RATIO = {
    1: 1,
    2: 2,
    3: 4,
    4: 6
}

class CorruptDownload(Exception):
    pass


class MakeTFRecords:
    def __init__(self, df_path, page):
        self.page = page
        self.data_path = Path(df_path)
        if not os.path.exists(self.data_path / 'tfrecords'):
            os.makedirs(self.data_path / 'tfrecords')
            self.record_path = Path(self.data_path / 'tfrecords')

        train_meta = pd.read_csv(self.data_path / 'train_metadata.csv')
        train_labels = pd.read_csv(self.data_path / 'train_labels.csv')
        self.train = train_meta.merge(train_labels, on='filename')

        self.samples_per_rec = TFRECORD_IMAGE_RATIO[page]
        self.num_recs = self.train.shape[0] // self.samples_per_rec
        if self.train.shape[0] % self.samples_per_rec != 0:
            self.num_recs += 1

    def preprocess_feats_2_encode(self):
        feats = self.train.copy()

        enc = OneHotEncoder(drop="first", sparse_output=False)
        enc.fit(np.array(feats["melanoma_history"]).reshape(-1, 1))

        feats['age'] = feats['age'].apply(lambda x: x[1:3]).astype(int)

        feats['sex'] = feats['sex'].replace({1: 0}).replace({2: 1}).astype(int)

        feats = pd.concat([
            feats[['age', 'sex']],
            pd.DataFrame(
                enc.transform(np.array(feats["melanoma_history"]).reshape(-1, 1)),
                columns=enc.get_feature_names_out(),
                index=feats.index,
            ).astype(int),
            feats['relapse'].astype(int)
        ], axis=1)

        files = self.train[['filename', 'tif_cksum', 'tif_size', 'us_tif_url', 'eu_tif_url', 'as_tif_url']]
        return feats, files

    def dl_image(self, url, filename):
        subprocess.run(['s5cmd', '--no-sign-request', 'cp', url, '.'], check=True)
        output = subprocess.run(['cksum', filename], capture_output=True, check=True)
        cksum, filesize, _ = output.stdout.decode().split()
        return cksum, filesize

    def verify_dl(self, file_row, cksum, filesize):
        if file_row['tif_cksum'] != cksum or file_row['tif_size'] != filesize:
            raise CorruptDownload(f"Corrupt download: {file_row['filename']}")

    def create_example(self, image, feats_row, height, width):
        image_raw = image.write_to_memory()
        
    def main(self):
        
    @staticmethod
    def image_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value])
        )

    @staticmethod
    def int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=[value])
        )
