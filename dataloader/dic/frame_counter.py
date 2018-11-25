#!/usr/bin/env python
"""Generate frame counts dict for a dataset.

Usage:
    frame_counter.py [options]

Options:
    -h, --help              Print help message
    --root=<str>            Path to root of dataset (should contain video folders that contain images)
        [default: /vision/vision_users/azou/data/hmdb51_flow/u/]
    --output=<str>          Output filename [default: hmdb_frame_count.pickle]
"""
from __future__ import print_function
from docopt import docopt
import os
import sys
import pickle

if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)

    # Final counts
    counts = {}
    min_count = sys.maxint

    # Generate list of video folders
    for root, dirs, files in os.walk(args['--root']):
        # Skip the root directory
        if len(dirs) != 0:
            continue

        # Process a directory and frame count into a dictionary entry
        name = os.path.basename(os.path.normpath(root))
        print('{}: {} frames'.format(name, len(files)))
        counts[name] = len(files)

        # Track minimum count
        if len(files) < min_count:
            min_count = len(files)

    with open(args['--output'], 'wb') as ofile:
        pickle.dump(counts, ofile)

    print('Minimum frame count = {}'.format(min_count))
