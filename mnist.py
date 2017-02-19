import struct
import numpy as np

def load():
    """ Loads MNIST data and returns it as:
        [training_data, labels], [test_data, labels] """

    training_images = _load_image_file("data/training_images")
    training_labels = _load_label_file("data/training_labels")
    test_images = _load_image_file("data/test_images")
    test_labels = _load_label_file("data/test_labels")

    return [training_images, training_labels], [test_images, test_labels]

def _load_image_file(filename):
    """ Loads an MNIST image file, returning a 3 dimensional numpy array """
    fin = open(filename)

    # Skip file description, this is static given the context
    __read_int(fin)

    num_images = __read_int(fin)
    num_rows = __read_int(fin)
    num_cols = __read_int(fin)

    pixels_per_image = num_rows * num_cols
    total_pixels = num_images * pixels_per_image

    pixel_data = np.fromfile(fin, dtype=np.uint8, count=total_pixels)
    images = pixel_data.reshape((num_images, num_rows, num_cols))

    return images

def _load_label_file(filename):
    """ Loads an MNIST label file, returning a 1 dimensional numpy array """
    fin = open(filename)

    # Skip file description, this is static given the context
    __read_int(fin)

    num_labels = __read_int(fin)

    labels = np.fromfile(fin, dtype=np.uint8, count=num_labels)

    return labels

def __read_int(fin):
    return struct.unpack('>i', fin.read(4))[0]

