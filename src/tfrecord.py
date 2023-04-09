import gc
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips as pv
import tensorflow as tf
from loguru import logger
from sklearn.preprocessing import OneHotEncoder

# logger.remove(0)
logger.add('./logs/tfrecord.log', format='{time} {level} {message}', level='INFO')

TFRECORD_IMAGE_RATIO = {
    1: 1,
    2: 2,
    3: 4,
    4: 8
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
        train = train_meta.merge(train_labels, on='filename')

        # Sort by filesize to pair large files with small ones
        train = train.sort_values('tif_size').reset_index(drop=True)
        train1 = train.iloc[:len(train)//2].reset_index(drop=True)
        train2 = train.iloc[~train1.index].reset_index(drop=True)
        self.train = pd.concat([train1, train2]).sort_index().reset_index(drop=True)

        self.samples_per_rec = TFRECORD_IMAGE_RATIO[page]
        self.num_recs = self.train.shape[0] // self.samples_per_rec
        if self.train.shape[0] % self.samples_per_rec != 0:
            self.num_recs += 1

    def preprocess_feats_2_encode(self):
        feats = self.train.copy()

        feats["melanoma_history"] = feats["melanoma_history"].replace({'YES': 1, 'NO': 0}).fillna(-1).astype(int)

        feats['age'] = feats['age'].apply(lambda x: x[1:3]).astype(int)
        feats['sex'] = feats['sex'].replace({1: 0}).replace({2: 1}).astype(int)

        feats = pd.concat([
            feats[['age', 'sex']],
            feats['relapse'].astype(int)
        ], axis=1)

        files = self.train[['filename', 'tif_cksum', 'tif_size', 'us_tif_url', 'eu_tif_url', 'as_tif_url', 'local_url']]
        return feats, files

    def dl_image(self, url, filename, idx):
        if idx == 0:
            subprocess.run(['s5cmd', '--endpoint-url', 'https://storage.googleapis.com', 'cp', url, '.'], check=True)
            return self.extract_cksum(filename)
        subprocess.run(['s5cmd', '--no-sign-request', 'cp', url, '.'], check=True)
        return self.extract_cksum(filename)

    def extract_cksum(self, filename):
        output = subprocess.run(['cksum', filename], capture_output=True, check=True)
        cksum, filesize, _ = output.stdout.decode().split()
        return cksum, filesize

    def upload_tfrec(self, rec):
        subprocess.run(['s5cmd', '--endpoint-url', 'https://storage.googleapis.com', 'cp', f'{rec}', f's3://visiomel/page_{self.page}/'], check=True)

    def verify_dl(self, file_row, cksum, filesize):
        if file_row['tif_cksum'] != int(cksum) or file_row['tif_size'] != int(filesize):
            raise CorruptDownload(f"Corrupt download: {file_row['filename']}")

    def load_image(self, filename):
        img = pv.Image.new_from_file(filename, access='sequential', page=self.page)
        height, width = img.height, img.width
        img = img.write_to_buffer('.jpg', Q=100)
        os.remove(filename)
        return img, height, width

    def create_example(self, image, feats_row, files_row, height, width):
        key = hashlib.sha256(image).hexdigest()
        return {
            'image/encoded': self.image_feature(image),
            'image/height': self.int64_feature(height),
            'image/width': self.int64_feature(width),
            'image/filename': self.image_feature(files_row['filename'].encode()),
            'image/cksum': self.image_feature(key.encode()),
            'meta/age': self.int64_feature(feats_row['age']),
            'meta/sex': self.int64_feature(feats_row['sex']),
            'meta/melanoma_history_yes': self.int64_feature(feats_row['x0_YES']),
            'meta/melanoma_history_nan': self.int64_feature(feats_row['x0_nan']),
            'label': self.int64_feature(feats_row['relapse']),
        }

    def main(self):
        feats, files = self.preprocess_feats_2_encode()
        for rec in range(self.num_recs):
            logger.info(f"Creating record {rec}")
            with tf.io.TFRecordWriter(
                f'{self.record_path}/train_{rec}.tfrec'
            ) as writer:
                for i in range(self.samples_per_rec):
                    idx = rec * self.samples_per_rec + i
                    if idx >= self.train.shape[0]:
                        break
                    feats_row = feats.iloc[idx]
                    files_row = files.iloc[idx]
                    filename = files_row['filename']
                    for idx, url in enumerate(
                        [files_row['local_url'], files_row['us_tif_url'], files_row['eu_tif_url'], files_row['as_tif_url']]
                    ):
                        try:
                            cksum, filesize = self.dl_image(url, filename, idx)
                            self.verify_dl(files_row, cksum, filesize)
                            break
                        except CorruptDownload:
                            with logger.contextualize(filename=filename):
                                logger.info(
                                    f"Corrupt download: {filename} in {'local' if idx == 0 else 'us' if idx == 1 else 'eu' if idx == 2 else 'as'}"
                                )
                            continue
                    image, height, width = self.load_image(filename)
                    example = self.create_example(image, feats_row, files_row, height, width)
                    writer.write(tf.train.Example(features=tf.train.Features(feature=example)).SerializeToString())
                    gc.collect()
                self.upload_tfrec(f'{self.record_path}/train_{rec}.tfrec')
                os.remove(f'{self.record_path}/train_{rec}.tfrec')

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

    @staticmethod
    def float32_feature(value):
        """Returns an float32_list from a float"""
        return tf.train.Feature(
            float32_list=tf.train.FloatList(value=[value])
        )


class ParseTFRecords:
    