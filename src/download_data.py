import pyvips as pv
import pandas as pd
import os
import subprocess


def read_file(filepath):
    return pd.read_csv(filepath)


def dl_img(row, CFG):
    path = row[10]
    file = row[1]
    new = row[1].split('.')[0]

    subprocess.run(['s5cmd', '--no-sign-request', 'cp', '--no-clobber', path, '/mnt/disk1/visiomel-melanoma-runtime/data/images/'],
                   capture_output=True)
    img = pv.Image.new_from_file(f'{CFG.DATA_PATH}/images/{file}', page=4)
    img.write_to_file(f'{CFG.DATA_PATH}/images/{new}.jpg')
    del img
    os.remove(f'{CFG.DATA_PATH}/images/{file}')


def main(CFG):
    df = read_file(f'{CFG.DATA_PATH}/train_metadata.csv')

    for row in df.itertuples():
        dl_img(row, CFG)


if __name__ == '__main__':
    from config import CFG
    CFG = CFG()

    main(CFG)
