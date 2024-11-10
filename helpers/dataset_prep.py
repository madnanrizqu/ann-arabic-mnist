from pathlib import Path
import cv2
import numpy as np
from keras.utils import to_categorical


def convert_dataset_to_numpy():
    # initialize lists to store images and labels
    images = []
    labels = []

    # walk through each digit folder (0-9)
    dataset_path = Path("dataset")
    for digit in range(10):
        digit_path = dataset_path / str(digit)

        # process each image in the digit folder
        for image_path in digit_path.glob("*"):
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # add to our lists
                images.append(img)
                labels.append(digit)

    # convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)

    return X, y


def distribute_class(dataset):
    X, y = dataset

    samples_per_class = 100
    num_of_distributions = 10
    num_of_class = 10
    res_X = []
    res_y = []

    # split by class
    # this ensures the training set / test set has equal distribution of each class
    for _ in range(num_of_distributions):
        for class_index in range(num_of_class):
            target_indices = np.where(y == class_index)[0]

            res_X.append(X[target_indices[:samples_per_class]])
            res_y.append(y[target_indices[:samples_per_class]])

    # flatten the list of numpy array
    res_X = np.concatenate(res_X)
    res_y = np.concatenate(res_y)

    return res_X, res_y


def split_dataset(dataset):
    X, y = dataset

    x_train = X[:9000]
    y_train = y[:9000]
    x_test = X[9000:]
    y_test = y[9000:]

    return x_train, y_train, x_test, y_test


def preprocess_dataset(dataset):
    x_train, y_train, x_test, y_test = dataset

    # reshape and normalize input data
    x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
    x_train = x_train.astype("float32")
    x_train /= 255

    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = to_categorical(y_train)

    # same for test data
    x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)
    x_test = x_test.astype("float32")
    x_test /= 255
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


# Uncomment to visualize result
# from visualizer import visualize_distribution, visualize_dataset_shapes

# x_train, y_train, x_test, y_test = preprocess_dataset(
#     split_dataset(distribute_class(convert_dataset_to_numpy()))
# )

# x_train should look like: (9000, 1, 784)
# y_train should look like: (9000, 10)
# x_text should look like: (1000, 1, 784)
# y_test should look like: (1000, 10)
# visualize_dataset_shapes(x_train, y_train, x_test, y_test)

# all class should be distributed evenly per 1000 datum
# so each 1000 datum, contains 100 class 0, 100 class 1, ..., 100 class 10
# visualize_distribution(y_train, y_test)
