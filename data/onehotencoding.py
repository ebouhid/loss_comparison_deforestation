import numpy as np

def encode(label_array, num_classes):
    label_array = np.uint8(label_array)
    flat_labels = label_array.flatten()
    one_hot_encoded = np.eye(num_classes)[flat_labels]
    one_hot_encoded = one_hot_encoded.reshape(label_array.shape + (num_classes, ))

    return one_hot_encoded

def decode(one_hot_array):
    # Find the argmax for each pixel
    decoded = np.argmax(one_hot_array, axis=0)

    return decoded
