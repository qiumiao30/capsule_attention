# -*- coding: utf-8 -*-
# @Time    : 2021/4/24 23:40
# @File    : loss_and_metries.py
import tensorflow as tf
from keras import backend as K
import numpy as np
from sklearn.metrics import f1_score


def get_loss(loss_name):
    if loss_name == "focal_bce":
        loss = focal_binarycrossentropy
    elif loss_name == "bce":
        loss = 'binary_crossentropy'
    elif loss_name == "mccfocal":
        loss = mccfaclloss
    elif loss_name == "dicebce":
        loss = DiceBCELoss
    elif loss_name == "combo":
        loss = Combo_loss
    else:
        print("loss name missing")
    return loss


############################ metrics ###############################
def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def mcc(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred = tf.convert_to_tensor(y_pred, np.float32)
    y_true = tf.convert_to_tensor(y_true, np.float32)

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


##########################################loss###########################################
import tensorflow as tf

ALPHA = 0.25
BETA = 0.5
GAMMA = 2
CE_RATIO = 0.5

def iouloss(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=-1)
    sum_ = K.sum(y_true + y_pred, axis=-1)
    jac = intersection / (sum_ - intersection)
    return 1 - jac

def focal_binarycrossentropy(y_true, y_pred):
    # focal loss with gamma 8
    t1 = K.binary_crossentropy(y_true, y_pred)
    alpha = 0.10
    lambd = 5
    t2 = tf.where(tf.equal(y_true, 0), alpha * t1 * (y_pred ** lambd), (1 - alpha) * t1 * ((1 - y_pred) ** lambd))
    return t2

def biased_crossentropy(y_true, y_pred):
    # apply 1000 times heavier punishment to ship pixels
    t1 = K.binary_crossentropy(y_true, y_pred)
    t2 = tf.where(tf.equal(y_true, 0), t1 * 0.001, t1)
    return t2

def mccloss(y_true, y_pred):
    #     loss1 = K.binary_crossentropy(y_true, y_pred)
    #     t1=K.binary_crossentropy(y_true, y_pred)
    #     loss1 = tf.where(tf.equal(y_true,0),t1*(y_pred**5),t1*((1-y_pred)**5))
    t1 = K.binary_crossentropy(y_true, y_pred)
    t2 = tf.where(tf.equal(y_true, 0), t1 * 0.01, t1)

    loss2 = 1 - matthews_correlation(y_true, y_pred)
    return t2 - loss2


def mccfocalloss(y_true, y_pred):
    t1 = focal_binarycrossentropy(y_true, y_pred)
    t2 = (1 + matthews_correlation(y_true, y_pred)) / 2
    return t1 - t2

def matthews_correlation(y_true, y_pred):

    y_pred_pos = y_pred
    y_pred_neg = 1 - y_pred_pos

    y_pos = y_true
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def mccfaclloss(y_true, y_pred):
    t1 = focal_binarycrossentropy(y_true, y_pred)
    t2 = (1 - matthews_correlation(y_true, y_pred))
    # t2 = mcc_loss(y_true, y_pred)
    return t1 + t2



def newmccfacl(y_true, y_pred):
    t1 = focal_binarycrossentropy(y_true, y_pred)
    t2 = 1 - matthews_correlation(y_true, y_pred)
    t3 = tf.exp(3 * t2)
    return t1 * t3

def mcccloss(y_true, y_pred, alpha=0.5, gamma=6):
    loss1 = K.binary_crossentropy(y_true, y_pred)
    loss1_EXP = K.exp(-loss1)
    mcccloss = K.mean(alpha * K.pow((1 - loss1_EXP), gamma) * loss1)

    return mcccloss


def mcc_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0) * 1e2
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0) / 1e2

    up = tp * tn - fp * fn
    down = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    mcc = up / (down + K.epsilon())
    # mcc = tf.where(tf.is_nan(mcc), tf.zeros_like(mcc), mcc)

    return 1 - K.mean(mcc)


def f1_weighted(y_true, y_pred):  # shapes (batch, 4)

    # for metrics include these two lines, for loss, don't include them
    # these are meant to round 'pred' to exactly zeros and ones
    # predLabels = K.argmax(pred, axis=-1)
    # pred = K.one_hot(predLabels, 4)

    ground_positives = K.sum(y_true, axis=0)  # = TP + FN
    pred_positives = K.sum(y_pred, axis=0)  # = TP + FP
    true_positives = K.sum(y_true * y_pred, axis=0)  # = TP
    # all with shape (4,)

    precision = (true_positives + K.epsilon()) / (pred_positives + K.epsilon())
    recall = (true_positives + K.epsilon()) / (ground_positives + K.epsilon())
    # both = 1 if ground_positives == 0 or pred_positives == 0
    # shape (4,)

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    # not sure if this last epsilon is necessary
    # matematically not, but maybe to avoid computational instability
    # still with shape (4,)

    weighted_f1 = f1 * ground_positives / K.sum(ground_positives)
    weighted_f1 = K.sum(weighted_f1)

    return 1 - weighted_f1  # for metrics, return only 'weighted_f1'


def DiceLoss(targets, inputs, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    intersection = K.sum(targets * inputs)
    dice = (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice


def DiceBCELoss(targets, inputs, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    BCE = K.binary_crossentropy(targets, inputs)
    intersection = K.sum(targets * inputs)
    dice_loss = 1 - (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss

    return Dice_BCE


def IoULoss(targets, inputs, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    intersection = K.sum(K.dot(targets, inputs))
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU


# Keras
ALPHA = 0.8
GAMMA = 2


def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1 - BCE_EXP), gamma) * BCE)

    return focal_loss


ALPHA = 0.5
BETA = 0.5


def Combo_loss(targets, inputs, smooth=1e-6):
    targets = K.flatten(targets)
    inputs = K.flatten(inputs)

    intersection = K.sum(targets * inputs)
    dice = (2. * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    inputs = K.clip(inputs, K.epsilon(), 1.0 - K.epsilon())
    out = - (ALPHA * ((targets * K.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * K.log(1.0 - inputs))))
    weighted_ce = K.mean(out, axis=-1)
    combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)

    return combo


def TverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    # True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1 - targets) * inputs))
    FN = K.sum((targets * (1 - inputs)))

    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return 1 - Tversky


ALPHA = 0.5
BETA = 0.8
GAMMA = 5


def FocalTverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    # True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1 - targets) * inputs))
    FN = K.sum((targets * (1 - inputs)))

    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    FocalTversky = K.pow((1 - Tversky), gamma)

    return FocalTversky
