import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
DATA_FILE='../sample_csv/iris.csv'
def pd_load(path=DATA_FILE):
    df = pd.read_csv(path,sep=',',header=0).rename({'Unnamed: 0':'column'},axis=1)
    #df.to_csv('/Users/hagi/downloads/color_data.csv/color_data2.csv', index=False)
    print(df.shape)
    return df

def main():
    df = pd_load(path=DATA_FILE)
    # 散布図行列の出力
    sns.pairplot(data=df, hue="target", diag_kind="kde", kind="reg")
    # 線グラフ
    df.plot(figsize=(15,4), title="title")
    plt.show()
    
    
if __name__=='__main__':
    main()
