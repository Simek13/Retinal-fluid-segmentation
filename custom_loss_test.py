import numpy as np


def weighted_dice_coef(y_true, y_pred, smooth=1e-7):
    # n_el = K.int_shape(K.flatten(y_pred))[0]
    w = np.sum(y_true, axis=(0, 1))
    w = np.sum(w) / (w + 1)
    w = w / np.max(w)
    y_pred = np.argmax(y_pred, axis=-1)
    y_pred_hot = np.eye(4)[y_pred]
    numerator = y_true * y_pred_hot
    numerator = w * np.sum(numerator, axis=(0, 1))
    numerator = np.sum(numerator)

    denominator = y_true + y_pred_hot
    denominator = w * np.sum(denominator, axis=(0, 1))
    denominator = np.sum(denominator)

    return (2. * numerator + smooth) / (denominator + smooth)


def weighted_mse(y_true, y_pred):
    n_el = y_true.size
    count_positive = np.count_nonzero(y_true)
    count_negative = n_el - count_positive
    w1 = w2 = 1
    if count_negative != 0 and count_positive != 0:
        if count_negative > count_positive:
            w1 = count_positive / count_negative
        else:
            w2 = count_negative / count_positive
    positive_indices = np.where(y_true != 0)
    negative_indices = np.where(y_true == 0)
    w = np.zeros(y_true.shape)
    w[positive_indices] = w2
    w[negative_indices] = w1

    error = (w * ((y_true - y_pred) ** 2)).mean()
    return error


if __name__ == '__main__':
    y_true = np.array([[[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
              [[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
              [[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
              [[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]],
              [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
              ])
    y_pred = np.array([[[0.5, 0.2, 0.1, 0.2], [0.4, 0.5, 0.1, 0.0], [0.7, 0.1, 0.1, 0.1]],
              [[0.6, 0.2, 0.1, 0.1], [0.0, 0.9, 0.1, 0.0], [0.6, 0.4, 0.0, 0.0]],
              [[0.2, 0.1, 0.5, 0.2], [0.0, 0.9, 0.1, 0.0], [0.0, 0.5, 0.4, 0.1]],
              [[0.0, 0.1, 0.5, 0.4], [0.3, 0.5, 0.1, 0.1], [0.9, 0.1, 0.0, 0.0]],
              [[0.5, 0.3, 0.2, 0.0], [0.7, 0.2, 0.1, 0.1], [0.6, 0.1, 0.1, 0.2]]
              ])

    y_true_mse = np.array([[33, 27, 26],
                  [44, 7, 8],
                  [20, 0, 0],
                  [3, 24, 0],
                  [33, 20, 17]])
    y_pred_mse = np.array([[27, 26, 33],
                  [43, 7, 8],
                  [23, 17, 8],
                  [2, 5, 0],
                  [30, 11, 16]])

    dice = weighted_dice_coef(y_true, y_pred)
    print(dice)
    mse = weighted_mse(y_true_mse, y_pred_mse)
    print(mse)
