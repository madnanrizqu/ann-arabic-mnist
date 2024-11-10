import numpy as np


def calculate_accuracy(predictions, y_test):
    # convert predictions to class labels
    predicted_classes = np.argmax(predictions, axis=2)
    predicted_classes = predicted_classes.flatten()
    actual_classes = np.argmax(y_test, axis=1)

    print("Sample predictions vs actuals:")
    for i in range(100):
        print(f"Prediction: {predicted_classes[i]}, Actual: {actual_classes[i]}")

    # calculate accuracy
    correct = np.sum(predicted_classes == actual_classes)
    total = len(y_test)
    accuracy = correct / total

    return accuracy
