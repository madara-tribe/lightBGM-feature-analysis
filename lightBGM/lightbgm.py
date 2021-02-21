import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import warnings
import lightgbm as lgb 
from sklearn.preprocessing import StandardScaler
import cv2
from utils import display, on_docker_analysis, on_jupyter_analysis
warnings.filterwarnings('ignore')


DATA_FILE='../sample_csv/iris.csv'
def pd_load(path=DATA_FILE):
    df = pd.read_csv(path,sep=',',header=0).rename({'Unnamed: 0':'column'},axis=1)
    #df.to_csv('/Users/hagi/downloads/color_data.csv/color_data2.csv', index=False)
    print(df.shape)
    return df

def normarize(X):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    return Xs


def create_dataset(df):
    tname='target'
    X = df.drop(tname, axis=1).values # 説明変数(target以外の特徴量)
    y = df[tname].values # 目的変数(target)
    X = normarize(X)
    print(X.shape, y.shape, X.min(), X.max())
    return X, y

def lightbgm_analysis(jupyter_is):
    NUM_CLS=10+1
    tname='target'
    df = pd_load(path=DATA_FILE)
    X, y = create_dataset(df)
    on_docker_analysis(df, X, y, NUM_CLS, target_name=tname, on_jupyter=jupyter)
    if jupyter:
        on_jupyter_analysis(X, y, NUM_CLS)
        
if __name__=='__main__':
    jupyter=None
    lightbgm_analysis(jupyter)
