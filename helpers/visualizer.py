import matplotlib.pyplot as plt
import numpy as np


def visualize_distribution(y_train, y_test):
    # convert one-hot encoded labels back to digits
    train_labels = np.argmax(y_train, axis=1)
    test_labels = np.argmax(y_test, axis=1)

    # get counts
    train_unique, train_counts = np.unique(train_labels, return_counts=True)
    test_unique, test_counts = np.unique(test_labels, return_counts=True)

    # set up the plot
    plt.figure(figsize=(12, 5))

    # plot training distribution
    plt.subplot(1, 2, 1)
    plt.bar(train_unique, train_counts)
    plt.title("Training Set Distribution")
    plt.xlabel("Digit")
    plt.ylabel("Number of Samples")
    plt.grid(True, alpha=0.3)

    # plot testing distribution
    plt.subplot(1, 2, 2)
    plt.bar(test_unique, test_counts)
    plt.title("Test Set Distribution")
    plt.xlabel("Digit")
    plt.ylabel("Number of Samples")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_dataset_shapes(x_train, y_train, x_test, y_test):
    # create figure
    plt.figure(figsize=(12, 6))

    # create a table of shapes
    shapes = [
        ["Dataset", "Shape", "Dimensions"],
        ["x_train", str(x_train.shape), f"{np.prod(x_train.shape):,} total values"],
        ["y_train", str(y_train.shape), f"{np.prod(y_train.shape):,} total values"],
        ["x_test", str(x_test.shape), f"{np.prod(x_test.shape):,} total values"],
        ["y_test", str(y_test.shape), f"{np.prod(y_test.shape):,} total values"],
    ]

    # hide axes
    plt.axis("off")

    # create table
    table = plt.table(
        cellText=shapes,
        colWidths=[0.2, 0.3, 0.3],
        cellLoc="center",
        loc="center",
        cellColours=[["lightgray"] * 3] + [["white"] * 3] * 4,
    )

    # style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # add title
    plt.title("Dataset Shapes Visualization", pad=20)

    plt.tight_layout()
    plt.show()
