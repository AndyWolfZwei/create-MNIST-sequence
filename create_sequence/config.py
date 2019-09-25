import sys
import os


class MNIST:
    URL = u"http://yann.lecun.com/exdb/mnist"
    TRAIN = u"{}/train-images-idx3-ubyte.gz".format(URL)
    TRAIN_LABELS = u"{}/train-labels-idx1-ubyte.gz".format(URL)
    TEST = u"{}/t10k-images-idx3-ubyte.gz".format(URL)
    TEST_LABELS = u"{}/t10k-labels-idx1-ubyte.gz".format(URL)


class ImgProcessPara:
    def __init__(self):
        self.cache = os.path.join(sys.path[0], "mnist_data")
        self.output_path = None
        # basic arguments
        self.number = None
        self.image_width = 200
        self.min_spacing = 0
        self.max_spacing = 10
        # data augmentation arguments
        self.data_augmentation = False
        self.random_all = False
        self.noise_ratio = None
        self.angle = None
        self.erode_kernel = (1, 2)
        self.dilate_kernel = (1, 2)
        # multiprocessing arguments
        self.multi_core = 0
        self.size = 100

