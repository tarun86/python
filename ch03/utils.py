import sys
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

print DATA_DIR

if not os.path.exists(DATA_DIR):
    print "error: data directory doesnt exist"
    sys.exit(0)

CHARTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "charts")

if not os.path.exists(CHARTS_DIR):
    os.mkdir(CHARTS_DIR)