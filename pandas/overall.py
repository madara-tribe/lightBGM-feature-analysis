import pandas as pd
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np

DATA_FILE='../sample_csv/iris.csv'
def pd_load(path=DATA_FILE):
    df = pd.read_csv(path,sep=',',header=0).rename({'Unnamed: 0':'column'},axis=1)
    #df.to_csv('/Users/hagi/downloads/color_data.csv/color_data2.csv', index=False)
    print(df.shape)
    return df

def cv_save(path):
    imgs = [cv2.imread(os.path.join(path, im)) for im in os.listdir(path) if im!=".DS_Store"]
    imgs_ = np.hstack(imgs)
    cv2.imwrite(os.path.join(path, 'result.png'), imgs_)
    

def pandas_save_as_img(df, save_path='results'):
    path = save_path
    os.makedirs(path, exist_ok=True)
    plt.figure()

    df.plot.line()
    df.plot.line(subplots=True) 
    plt.savefig(os.path.join(path, 'line.png'))
    df.plot.bar()
    plt.savefig(os.path.join(path, 'bar.png'))
    df.plot.area()
    plt.savefig(os.path.join(path, 'area.png'))
    df.plot.bar(stacked=True)
    plt.savefig(os.path.join(path, 'bar.png'))
    df.plot.box()
    plt.savefig(os.path.join(path, 'box.png'))
    df.plot.kde()
    plt.savefig(os.path.join(path, 'kde.png'))
    plt.show()
    plt.close('all')
    print('concat image and save')
    cv_save(path)
  
def overall_analysis():
    df = pd_load(path=DATA_FILE)
    #df.head()
    target_name = 'target'
    path='results'
    os.makedirs(path, exist_ok=True)
    pandas_save_as_img(df.drop(target_name, axis=1), save_path=path)
    
if __name__=='__main__':
    overall_analysis()
