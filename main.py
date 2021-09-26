# -*- coding: utf-8 -*-
# @Time    : 2021/4/24 18:23
# @File    : main.py
import numpy as np 
import os
import time
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, recall_score, precision_score

from keras.callbacks import * 
from keras.utils import to_categorical
import train_model
from loss_and_metries import mcc, recall, precision, f1
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import loss_and_metries
from data.dataload import loading
# from tf.keras.metrics import AUC, Precision, Recall, BinaryAccuracy
#     # TrueNegatives, TruePositives, FalseNegatives, FalsePositives

from config import args_parser
args = args_parser()

from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(120)

x_train, y_train, x_test, y_test = loading(args.data_name)

model_prefix = "%s" % (args.model)
loss_prefix = "%s" % (args.loss)
optimizer_prefix = "%s" % (args.optimizer)
data_prefix = "%s" % (args.data_name)

nb_iterations = args.iterations
batch_size = args.batch_size
nb_epochs = np.ceil(nb_iterations * (batch_size / x_train.shape[0])).astype(int)

input_shape = x_train.shape
model = train_model.get_model(args.model, input_shape)

def get_model_size(model):
	para_num = sum([np.prod(w.shape) for w in model.get_weights()])
	para_size = para_num * 4 / 1024 / 1024
	return para_size
para_size = get_model_size(model)
print("para_size: ",para_size)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=np.ceil(nb_epochs / 20.).astype(int),
                              verbose=args.verbose, mode='auto', min_lr=1e-5,
                              cooldown=np.ceil(nb_epochs / 40.).astype(int))
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
#                               verbose=args.verbose, mode='auto', min_lr=1e-5,
#                               cooldown=np.ceil(nb_epochs / 40.).astype(int))
early_stopping = EarlyStopping(monitor='val_loss', patience=150,)
# early_stopping = EarlyStopping(monitor='loss', patience=15,)

if args.save:
    if not os.path.exists(args.weight_dir):
        os.mkdir(args.weight_dir)
    if not os.path.exists(os.path.join(args.weight_dir, model_prefix)):
        os.mkdir(os.path.join(args.weight_dir, model_prefix))
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(os.path.join(args.log_dir, model_prefix)):
        os.mkdir(os.path.join(args.log_dir, model_prefix))
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, model_prefix)):
        os.mkdir(os.path.join(args.output_dir, model_prefix))
    model_checkpoint = ModelCheckpoint(os.path.join(args.weight_dir, model_prefix, "%s_%s_%s_%s_best_val_mcc_weights.h5" %
                                                    (data_prefix, model_prefix, loss_prefix, optimizer_prefix)),
                                                    verbose=1, monitor='val_mcc', mode='max', save_best_only=True)
    # if not os.path.exists(os.path.join(args.log_dir, model_prefix)):
    #     os.mkdir(os.path.join(args.log_dir, model_prefix))
    csv_logger = CSVLogger(os.path.join(args.log_dir, model_prefix, '%s_%s_%s_%s.csv' %
                                        (data_prefix, model_prefix, loss_prefix, optimizer_prefix+time.strftime('_%m%d_%H%M%S'))))

    # callback_list = [reduce_lr, csv_logger]
    callback_list = [model_checkpoint, reduce_lr, csv_logger, early_stopping]
else:
    callback_list = [model_checkpoint, reduce_lr, early_stopping]

if args.optimizer == "adam":
    from keras.optimizers import Adam
    optm = Adam(lr=args.lr)
elif args.optimizer == "nadam":
    from keras.optimizers import Nadam
    optm = Nadam(lr=args.lr)
elif args.optimizer == "adadelta":
    from keras.optimizers import Adadelta
    optm = Adadelta(lr=args.lr, rho=0.95, epsilon=1e-8)
else:
    from keras.optimizers import SGD
    optm = SGD(lr=args.lr, decay=5e-4, momentum=0.9)  # , nesterov=True)


model.compile(loss=loss_and_metries.get_loss(args.loss),
              optimizer = optm,
              metrics=[mcc, f1, recall, precision])
              # metrics=[mcc])

# train
if args.train:
    # splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=2019).split(x_train,y_train))
    # preds_val = []
    # y_val = []
    # for idx, (train_idx, val_idx) in enumerate(splits):
    #     print("Beginnig fold {}".format(idx+1))
    #     train_X, train_y, val_X, val_y = x_train[train_idx], y_train[train_idx], \
    #                                      x_train[val_idx], y_train[val_idx]
    model.fit(x_train, y_train, batch_size=batch_size, epochs=100, callbacks=callback_list,
               verbose=args.verbose, validation_split=args.validation_split)
    # model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs,
    #           verbose=args.verbose, validation_data=[x_val, y_val], callbacks=callback_list)
    if args.save:
        model.save_weights(os.path.join(args.weight_dir, model_prefix, "%s_final_weights.h5" % (model_prefix)))
else:
    model.load_weights(os.path.join(args.weight_dir, model_prefix, "%s_final_weights.h5" % (model_prefix)))

# if args.save:
#     if os.path.exists(os.path.join(args.weight_dir, model_prefix, "%s_%s_%s_%s_best_val_loss_weights.h5" %
#                                                                   (data_prefix, model_prefix, loss_prefix,
#                                                                    optimizer_prefix))):
#         model.load_weights(os.path.join(args.weight_dir, model_prefix, "%s_%s_%s_%s_best_val_loss_weights.h5" %
#                                         (data_prefix, model_prefix, loss_prefix, optimizer_prefix)))
#
#         y_preds = np.array(model.predict(x_test, batch_size=batch_size))
#         # y_preds = np.argmax(y_preds, axis=1)
#         y_preds = [1 if i >= 0.5 else 0 for i in y_preds]
#         np.savetxt(os.path.join(args.output_dir, model_prefix, "%s_%s_%s_%s_y_predicts.txt" %
#                                (data_prefix, model_prefix, loss_prefix, optimizer_prefix)), y_preds, fmt="%d")
#         f1 = f1_score(y_test, [1 if i >= 0.5 else 0 for i in y_preds])
#         mcc = matthews_corrcoef(y_test, [1 if i >= 0.5 else 0 for i in y_preds])
#         recall = recall_score(y_test, [1 if i >= 0.5 else 0 for i in y_preds])
#         precision = precision_score(y_test, [1 if i >= 0.5 else 0 for i in y_preds])
#         # print("F1 score: {}".format(f1_score(y_test, [1 if i >= 0.5 else 0 for i in y_preds])))
#
#         loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
#
#
#
#         pd.DataFrame([loss, mcc, f1, recall, precision, accuracy], index=['loss', 'mcc', 'f1', 'recall', 'precision', 'accuracy']).to_csv(
#             os.path.join(args.log_dir, model_prefix, '%s_%s_%s_%s_best_model_test.csv' %
#                          (data_prefix, model_prefix, loss_prefix, optimizer_prefix)), mode='a', header=False)



#     y_preds = np.array(model.predict(x_test, batch_size=batch_size))
#     # y_preds = np.argmax(y_preds, axis=1)
#     y_preds = [1 if i >= 0.5 else 0 for i in y_preds]
#     # print("F1 score: {}".format(f1_score(y_test, [1 if i >= 0.5 else 0 for i in y_preds])))
#     np.savetxt(os.path.join(args.output_dir, model_prefix, "%s_%s_%s_%s_y_predicts.txt" %
#                             (data_prefix, model_prefix, loss_prefix, optimizer_prefix)), y_preds, fmt="%d")
#
if os.path.exists(os.path.join(args.weight_dir, model_prefix, "%s_%s_%s_%s_best_val_mcc_weights.h5" %
                                                              (data_prefix, model_prefix, loss_prefix, optimizer_prefix))):
    model.load_weights(os.path.join(args.weight_dir, model_prefix, "%s_%s_%s_%s_best_val_mcc_weights.h5" %
                                                              (data_prefix, model_prefix, loss_prefix, optimizer_prefix)))
    test_evaluate = model.evaluate(x_test, y_test, batch_size=batch_size)
    pd.DataFrame(test_evaluate, index=['loss', 'mcc', 'f1', 'recall', 'precision']).to_csv(os.path.join(args.log_dir, model_prefix, '%s_%s_%s_%s_best_model_test.csv' %
                 (data_prefix, model_prefix, loss_prefix, optimizer_prefix)), mode='a', header=False)