import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import collections 

DATA_FILE= '../sample_csv/SalesJan2009.csv'
save_path='R2'
os.makedirs(save_path, exist_ok=True)

def pd_load(path=DATA_FILE):
    time_df = pd.read_csv(path,sep=',',header=0).rename({'Unnamed: 0':'column'},axis=1)
    print(time_df.shape)
    return time_df

def original_dataframe(df):
    csv=[[i.Transaction_date, i.Price, i.Name, i.City, i.State, i.Latitude] for idx, i in df.iterrows()]
    csv=pd.DataFrame(csv)
    listed=['Transaction_date', 'Price', 'Name', 'City', 'State', 'Latitude']
    csv.columns = listed
    return csv

def number_of_duplicate_hist(df, name):
    
    print('duplicates', list(collections.Counter(df[name]).keys()))
    
    plat = df[name].value_counts()
    print('number of duplicates')
    print(plat)
    
    
    print('\nUnique values of platform:', df[name].unique())
    df[name] = csv[name].astype(str)
    plat = csv[name].value_counts()

    plt.figure(figsize=(12,4))
    sns.barplot(plat.index, plat.values, alpha=0.8)
    plt.xlabel('Platform', fontsize=12)
    plt.ylabel('Occurence count', fontsize=12)
    plt.savefig(os.path.join(save_path, 'duplicate.png'))
    
def hour_hist(df, name):
    print('create hour column')
    df["hour"] = [i[10:].replace(':', '') for i in df[name]]

    # time histgram
    plt.figure(figsize=(12,4))
    df.hour.hist(bins=np.linspace(-0.5, 23.5, 25), label="train", alpha=0.7, density=True)
    plt.xlim(-0.5, 23.5)
    plt.legend(loc="best")
    plt.xlabel("Hour of Day")
    plt.ylabel("Fraction of Events")
    plt.savefig(os.path.join(save_path, 'time_hist.png'))
    return df

def time_data_analysis():
    df = pd_load(path=DATA_FILE)
    csv = original_dataframe(df)
    print('calcurate and plot duplicate')
    number_of_duplicate_hist(csv, name='Price')
    print('plot time histgram')
    hour_hist(csv, name='Transaction_date')
    
if __name__=='__main__':
    time_data_analysis()
