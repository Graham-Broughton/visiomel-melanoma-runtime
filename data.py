from src.tfrecord import main
import argparse
from config import CFG

CFG = CFG


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--page', required=True, type=int, choices=range(12))
args = parser.parse_args()

main(CFG, args)
