import argparse
import numpy as np
import sys
import os
import logging


def generate_random(args):
    if not args['noise_ratio']:
        args['noise_ratio'] = np.random.rand() / 10  # default generate number between (0,0.1)
        args['angle'] = np.random.rand() * 90  # default generate number between (0, 90)
    if not isinstance(args['erode_kernel'], tuple):
        args['erode_kernel'] = tuple(int(i) for i in args['erode_kernel'])
    if not isinstance(args['dilate_kernel'], tuple):
        args['dilate_kernel'] = tuple(int(i) for i in args['dilate_kernel'])


def args_parse():
    parser = argparse.ArgumentParser(description="Create sequence digits from MNIST")
    parser.add_argument("-c", "--cache", default=os.path.join(sys.path[0], "mnist_data"),
                        help="Directory to save data in")
    # basic argument
    parser.add_argument("-r", "--read_from_config", default=False, action='store_true',
                        help="True means read parameters from config, False will read from command line")
    parser.add_argument("-num", "--number", required=True, type=str, help="The sequence which need to be transformed")
    parser.add_argument("-iw", "--image_width", default=200, type=int, help="Width of output image")
    parser.add_argument("-min", "--min_spacing", default=0, type=int, help="Minimum space between two digits")
    parser.add_argument("-max", "--max_spacing", default=10, type=int, help="Maximum space between two digits")
    # data augmentation argument
    parser.add_argument("-da", "--data_augmentation", default=False, action='store_true',  help="Data augmentation")
    parser.add_argument("-ra", "--random_all", default=False, action='store_true',  help="Ture means random selection these data augmentation functions with 0.5 probability, ")
    parser.add_argument("-e", "--erode_kernel", default=(1,2),nargs=2, help="Image erosion requires accepting two parameters as erode kernel")
    parser.add_argument("-d", "--dilate_kernel", default=(2,1),nargs=2, help="Same with erode parameter")
    parser.add_argument("-a", "--angle", help="Rotation angle for every digit")
    parser.add_argument("-n", "--noise_ratio", help="Noise ratio for every digit")
    # multiprocessing argument
    parser.add_argument("-m", "--multi_core", default=0, type=int, help="Directory to save data in")
    parser.add_argument("-s", "--size", default=100, type=int, help="Directory to save data in")

    args = vars(parser.parse_args())
    num = args['number']
    if args['read_from_config']:
        from config import ImgProcessPara
        args = ImgProcessPara().__dict__
        if not args['number']:
            args['number'] = num
    generate_random(args)
    logging.info("Current arguments are : {}".format(args))
    return args
