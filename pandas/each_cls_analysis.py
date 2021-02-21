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
    

def pd_column_name(df):
    return df.columns.values

def plot_kinds(types):
    if types=='scatter':
        plot_types='scatter'
    elif types=='hexbin':
        plot_types='hexbin'
    elif types=='pie':
        plot_types='pie'
    elif types=='box':
        plot_types='box'
    elif types=='area':
        plot_types='area'
    else:
        plot_types='hist'
    return plot_types


def pandas_linear_plot(df, path, yname, types='scatter'):
    os.makedirs(os.path.join(path, types), exist_ok=True)
    df = df.copy() 
    plot_type = plot_kinds(types)  
    names = pd_column_name(df)
    xs = yname
    df.drop(str(xs), axis=1)
    for i, name in enumerate(names):
        if name==xs:
            continue
        df.plot(kind=plot_type, x=xs, y=name, subplots=True, layout=(2, 2))
        plt.savefig(os.path.join(path, types, 'kde{}.png'.format(i)))
    cv_save(os.path.join(path, types)) 
   
def each_cls_analysis():
    df = pd_load(path=DATA_FILE)
    #df.head()
    target_name = 'target'
    PATH= 'results'
    pandas_linear_plot(df, PATH, yname=target_name, types='scatter')
    pandas_linear_plot(df, PATH, yname=target_name, types='hist')
    pandas_linear_plot(df, PATH, yname=target_name, types='area')

if __name__=="__main__":
    each_cls_analysis()
