import numpy as np


def rotate_orientation(volume_data, volume_label, orientation='coronal'):
    if orientation == 'coronal':
        return volume_data.transpose((2, 0, 1)), volume_label.transpose((2, 0, 1))
    elif orientation == 'axial':
        return volume_data.transpose((1, 2, 0)), volume_label.transpose((1, 2, 0))
    elif orientation == 'sagital':
        return volume_data, volume_label
    else:
        raise ValueError('Invalid value for orientation. Choices: coronal, axial, sagital.')


def estimate_weights_mfb(labels):
    class_weights = np.zeros_like(labels)
    unique, counts = np.unique(labels, return_counts=True)
    median_freq = np.median(counts)
    weights = np.zeros(len(unique))
    for i, (label, count) in enumerate(zip(unique, counts)):
        class_weights += (median_freq // count) * np.array(labels == label)
        weights[i] = median_freq // count

    grads = np.gradient(labels)
    edge_weights = (grads[0] ** 2 + grads[1] ** 2) > 0
    class_weights += 2 * edge_weights
    return class_weights, weights


def remap_labels(labels, remap_config):
    """
    Function to remap the label values into the desired range of algorithm
    """
    if remap_config == 'FS':
        label_list = [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50,
                      51, 52, 53, 54, 58, 60]
    elif remap_config == 'Neo':
        labels[(labels >= 100) & (labels % 2 == 0)] = 210
        labels[(labels >= 100) & (labels % 2 == 1)] = 211
        label_list = [45, 211, 52, 50, 41, 39, 60, 37, 58, 56, 4, 11, 35, 48, 32, 46, 30, 62, 44, 210, 51, 49, 40, 38,
                      59, 36, 57, 55, 47, 31, 23, 61]
    elif remap_config == 'FS_Combined':
        label_list = [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50,
                      51, 52, 53, 54, 58, 60]
        label_list_2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                        13, 14, 16, 17, 18]
    elif remap_config == 'Neo_Combined':
        labels[(labels >= 100) & (labels % 2 == 0)] = 210
        labels[(labels >= 100) & (labels % 2 == 1)] = 211
        label_list = [45, 211, 52, 50, 41, 39, 60, 37, 58, 56, 4, 11, 35, 48, 32, 46, 30, 62, 44, 210, 51, 49, 40, 38,
                      59, 36, 57, 55, 47, 31, 23, 61]
        label_list_2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                        13, 14, 16, 17, 18]
    else:
        raise ValueError('Invalid argument value for remap config, only valid options are FS and Neo')

    if remap_config == 'Neo_Combined' or remap_config == 'FS_Combined':
        new_labels = np.zeros_like(labels)
        for label1, label2 in zip(label_list, label_list_2):
            label_present = np.zeros_like(labels)
            label_present[labels == label1] = 1
            new_labels = new_labels + (label2 + 1) * label_present
    else:
        new_labels = np.sum([(labels == label) * idx for idx, label in enumerate(label_list, 1)], 0).astype(np.float)

    return new_labels


def reduce_slices(data, labels, skip_frame=40):
    """
    This function removes the useless black slices from the start and end. And then selects every even numbered frame.
    """
    num_slices, H, W = data.shape
    mask_vector = np.zeros(num_slices, dtype=int)
    mask_vector[::2], mask_vector[1::2] = 1, 0
    mask_vector[:skip_frame], mask_vector[-skip_frame:-1] = 0, 0

    data_reduced = np.compress(mask_vector, data, axis=0).reshape((-1, H, W))
    labels_reduced = np.compress(mask_vector, labels, axis=0).reshape((-1, H, W))

    return data_reduced, labels_reduced


def remove_black(volume, labelss, threshold=0.999):
    return map(np.array, zip(*[[slice, labels] for slice, labels in zip(volume, labelss) if
                               np.sum(labels == 0) / np.prod(labels.shape) < threshold]))
