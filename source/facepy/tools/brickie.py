"""Script for visualizing dataset folders in file system.
"""
# MIT License
# 
# Copyright (c) 2018 Yichun Shi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import time
import argparse
from ..dataset import Dataset
from ..brickie import WebViewer

def main(args):
    dataset = os.path.abspath(args.dataset)
    print('Loading data from %s ...' % dataset)
    dt = Dataset(init_path=dataset)
    if dt.classes is not None:
        num = min(args.num_classes, len(dt.classes))
        if num < len(dt.classes):
            print('Only show the first %d folders, set "--num_classes" for more!' % num)
        image_groups = [c.images for c in dt.classes[:num]]
        descriptions = [c.label for c in dt.classes[:num]]
    else:
        num = min(args.num_images, len(dt.images))
        if num < len(dt.images):
            print('Only show the first %d images, set "--num_images" for more!' % num)
        image_groups = [[img] for img in dt.images[:num]]
        descriptions = None

    bc = WebViewer(image_groups = image_groups, descriptions=descriptions, port=args.port)
    bc.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str,
                    help='the port number to bind the server')
    parser.add_argument('-p', dest='port', type=int, default=8000,
                    help='the port number to bind the server')
    parser.add_argument('--num_classes', type=int, default=50,
                    help='Maximum number classes to show')
    parser.add_argument('--num_images', type=int, default=1000,
                    help='Maximum number images to show')
    args = parser.parse_args()
    main(args)