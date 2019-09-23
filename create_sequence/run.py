from src.download_read_MNIST import *
from src.process_image_function import *
from src.args_parse import args_parse
import sys
import multiprocessing


class CreateDigitSequence:
    def __init__(self):
        self.args = args_parse()
        self.data = get_mnist(self.args['cache'])

    def create_digit_sequence(self, number, image_width, min_spacing, max_spacing):
        """ A function that create an image representing the given number,
        with random spacing between the digits.
        Each digit is randomly sampled from the MNIST dataset.
        Returns an NumPy array representing the image.
        Parameters
        ----------
        number: str
        A string representing the number, e.g. "14543"
        image_width: int
        The image width (in pixel).
        min_spacing: int
        The minimum spacing between digits (in pixel).
        max_spacing: int
        The maximum spacing between digits (in pixel).
        """
        spacing = np.random.randint(min_spacing, max_spacing, size=len(number))
        reshape_width = (image_width - sum(spacing)) / len(number)
        return_np = None
        for idx, i in enumerate(number):
            n = np.random.randint(len(self.data[int(i)]))

            process_img = self.data[int(i)][n]  # random select a image from MNIST
            # data augmentation
            if self.args['data_augmentation']:
                process_img = data_augmentation(process_img, self.args)

            # combined it
            if not isinstance(return_np, np.ndarray):
                return_np = process_img
            else:
                return_np = np.concatenate([return_np, process_img], axis=1)
            if idx != len(number):
                return_np = np.concatenate([return_np, np.zeros(shape=(28, spacing[idx]))], axis=1)
        return return_np

    def workers(self, counter):
        self.save_sequence_digit(self.create_digit_sequence(self.args['number'], self.args['image_width'],
                                                            self.args['min_spacing'],self.args['max_spacing']), counter)

    def save_sequence_digit(self, processed_image, counter):
        save_path = os.path.join(sys.path[0], "output", self.args['number'], '{}.jpg'.format(counter))
        done = cv2.imwrite(save_path, processed_image)
        if not done:
            print('save failed, the path is {}'.format(save_path))
        else:
            print(' the path is {}'.format(save_path))


def main():
    CDS = CreateDigitSequence()
    if CDS.args['multi_core'] == 0:
        CDS.save_sequence_digit(CDS.create_digit_sequence(CDS.args['number'], CDS.args['image_width'],
                                                          CDS.args['min_spacing'], CDS.args['max_spacing']), '0')
    else:
        if not os.path.exists(os.path.join(sys.path[0], "output", CDS.args['number'])):
            os.makedirs(os.path.join(sys.path[0], "output", CDS.args['number']))
            counter = 1
        else:
            counter = max([int(i.split('.')[0]) for i in os.listdir(os.path.join(sys.path[0], "output",
                                                                                       CDS.args['number']))]) + 1
        pool = multiprocessing.Pool(processes=CDS.args['multi_core'])
        pool.map(CDS.workers, range(counter, counter + CDS.args['generated_number']))
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()