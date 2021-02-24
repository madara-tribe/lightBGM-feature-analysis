import numpy as np 
import matplotlib.pyplot as plt 
import os
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor as RFR

def RandomForestRegressor(X, y, state=2000):
    train_data_bs, test_data_bs, train_labels_bs, test_labels_bs = train_test_split(X, y, test_size=0.2)
    rg = RFR(n_jobs=-1, random_state=state)
    rg.fit(train_data_bs,train_labels_bs)
    predicted_labels_bs = rg.predict(test_data_bs)
    print('accuracy socore', rg.score(test_data_bs, test_labels_bs))
    
    
def create_new_column(df, new, name1, name2):
    df[new]=df[name1]-df[name2]
    return df

def normarize(X, MinMax=True):
    if MinMax:
        Xs = MinMaxScaler().fit_transform(X)
    else:
        Xs = preprocessing.StandardScaler().fit_transform(X)
    return Xs


def create_target(df, num_cls=10):
    vst=[]
    for cls in range(num_cls):
        if cls ==0 or cls==1 or cls==2:
            num = df[df.column == cls].shape[0]
            for _ in range(num):
                vst.append(np.array(df1.loc['L0']))
        elif cls ==3 or cls==4 or cls==5:
            num = df[df.column == cls].shape[0]
            for _ in range(num):
                vst.append(np.array(df1.loc['L1']))
        elif cls ==6 or cls==6 or cls==7:
            num = df[df.column == cls].shape[0]
            for _ in range(num):
                vst.append(np.array(df1.loc['L2']))
        elif cls >7:
            num = df[df.column == cls].shape[0]
            for _ in range(num):
                vst.append(np.array(df1.loc['l3']))
    return pd.DataFrame(np.vstack(vst))
    



def cal_accuracy(model, num_cls, X_test, y_label, y_onehot):
    _, acc =model.evaluate(X_test, y_onehot, verbose=0)
    print('\nTest accuracy: {0}'.format(acc))
    y_pred = model.predict(X_test, verbose=0)
    
    print('confusion matrix')
    f_pred=np.argmax(y_pred, axis=1)
    print('acuracy:{}'.format(accuracy_score(y_label, f_pred)))
    label_string = ['{}'.format(i) for i in range(num_cls-1)]
    print(classification_report(y_label, f_pred,target_names=label_string))
    
    cm = confusion_matrix(y_pred=f_pred, y_true=y_label)
    cmp = ConfusionMatrixDisplay(cm, display_labels=label_string)
    cmp.plot(cmap=plt.cm.Blues)
    

def normarize(X):
    mm = MinMaxScaler()
    return mm.fit_transform(X)
   

def to_onehot_label(y, num_cls): 
    label = np.eye(num_cls)[y]
    return np.reshape(label, (len(label), num_cls))

