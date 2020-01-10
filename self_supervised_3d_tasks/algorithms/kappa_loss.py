from keras import backend as K
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def quadratic_kappa(y_true, y_pred, N=5):
    actuals = K.eval(y_true)
    actuals = np.dot(actuals, np.array([x + 1 for x in range(N)])).astype(int)
    actuals = np.subtract(actuals, 1)
    preds = K.eval(y_pred)
    preds = np.dot(preds, np.array([x + 1 for x in range(N)])).astype(int)
    preds = np.subtract(preds, 1)
    """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition
    at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values
    of adoption rating."""
    w = np.zeros((N, N))
    O = confusion_matrix(actuals, preds)
    for i in range(N):
        for j in range(N):
            w[i][j] = float(((i - j) ** 2) / (N - 1) ** 2)

    act_hist = np.zeros([N])
    for item in actuals:
        act_hist[item] += 1

    pred_hist = np.zeros([N])
    for item in preds:
        pred_hist[item] += 1

    E = np.outer(act_hist, pred_hist);
    E = E / E.sum()
    O = O / O.sum()

    num = 0
    den = 0
    for i in range(N):
        for j in range(N):
            num += w[i][j] * O[i][j]
            den += w[i][j] * E[i][j]
    return (1 - (num / den))


if __name__ == "__main__":
    actuals = to_categorical(np.array([4, 4, 3, 4, 4, 4, 1, 1, 2, 1]), num_classes=5)
    preds = to_categorical(np.array([0, 2, 1, 0, 0, 0, 1, 1, 2, 1]), num_classes=5)
    F = quadratic_kappa(K.variable(actuals), K.variable(preds))
    print(F)
