import numpy as np 
import matplotlib.pyplot as plt 
import os
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler



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

