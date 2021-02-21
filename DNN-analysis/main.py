import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from keras import backend as K
import tensorflow as tf
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
import warnings
from model import load_model
from NNutils import normarize, to_onehot_label, cal_accuracy
warnings.filterwarnings('ignore')


BATCH_SIZE = 4
EPOCHS = 100
SAVE_PATH = 'results'
NUM_CLS=10+1
PATH = '../sample_csv/iris.csv'


def train(input_dim, X_train, y_train, X_test, y_test, y_test_):
    os.makedirs(SAVE_PATH, exist_ok=True)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    K.set_session(sess)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=2)
    callback = [reduce_lr]
    model = load_model(input_dim, NUM_CLS, activations='sigmoid', opt='Adam')


    startTime1 = datetime.now() #DB
    hist1 = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE,epochs=EPOCHS, 
                      callbacks=callback, verbose=2)
    endTime1 = datetime.now()
    diff1 = endTime1 - startTime1
    print("\n")
    print("Elapsed time for Keras training (s): ", diff1.total_seconds())
    print("\n")

    for key in ["loss", "val_loss"]:
        plt.plot(hist1.history[key],label=key)
    plt.legend()

    plt.savefig(os.path.join(SAVE_PATH, "loss_curve.png"))

    model.save(os.path.join(SAVE_PATH, "ep" + str(EPOCHS) + "_trained_unet_model.hdf5"))
    print("\nEnd of UNET training\n")
    
    print('testing.....')
    cal_accuracy(model, NUM_CLS, X_test, y_test_, y_test)
    K.clear_session()


def pd_load(path=PATH):
    df = pd.read_csv(path, sep=',',header=0).rename({'Unnamed: 0':'column'},axis=1)
    print(df.shape)
    return df

 

def create_dataset(df, target_column_name='column'):
    tname=target_column_name
    X = df.drop(tname, axis=1).values # 説明変数(target以外の特徴量) 
    y = df[tname].values # 目的変数(target) 
    X = normarize(X)
    print(X.shape, y.shape, X.min(), X.max())
    return X, y


def main():
    df = pd_load(path=PATH)
    X, y = create_dataset(df, target_column_name='target')
    input_dim = X.shape[1]
    X_train, X_test, y_train_, y_test_ = train_test_split(X, y,test_size=0.20, random_state=2)
    y_train = to_onehot_label(y_train_, NUM_CLS)
    y_test = to_onehot_label(y_test_, NUM_CLS)
    train(input_dim, X_train, y_train, X_test, y_test, y_test_)
    
if __name__=='__main__':
    main()
