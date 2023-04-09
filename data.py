from src.tfrecord import MakeTFRecords
import argparse
from config import CFG

CFG = CFG()

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--page', required=True, type=int, choices=range(12))
args = parser.parse_args()

make_tf = MakeTFRecords(CFG.DATA_PATH, args.page)
make_tf.main()
