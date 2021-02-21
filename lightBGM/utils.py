import IPython
import pandas as pd
import os
import cv2
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split 

# データフレームを綺麗に出力する関数
def display(*dfs, head=True):
    for df in dfs:
        IPython.display.display(df.head() if head else df)


def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=2)
    return X_train, X_test, y_train, y_test



def plot_feature_importance(df, img_name): 
    n_features = len(df)                              
    df_plot = df.sort_values('importance')            
    f_importance_plot = df_plot['importance'].values  
    plt.barh(range(n_features), f_importance_plot, align='center') 
    cols_plot = df_plot['feature'].values             
    plt.yticks(np.arange(n_features), cols_plot)      
    plt.xlabel('Feature importance')                  
    plt.ylabel('Feature')   
    plt.savefig(img_name+'.png')
    plt.show()

def on_docker_analysis(df, X, y, num_cls, target_name, on_jupyter=True):
    def gain_feature_importance(df, model, target_name, jupyter):
        tname = target_name
        cols = list(df.drop(tname, axis=1).columns) # 特徴量名のリスト(目的変数target以外)
        # 特徴量重要度の算出方法 'gain'(推奨) : トレーニングデータの損失の減少量を評価
        f_importance = np.array(model.feature_importance(importance_type='gain')) # 特徴量重要度の算出 //
        f_importance = f_importance / np.sum(f_importance) # 正規化(必要ない場合はコメントアウト)
        df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})
        df_importance = df_importance.sort_values('importance', ascending=False) # 降順ソート
        if jupyter:
            display(df_importance)
            # 特徴量重要度の可視化
        plot_feature_importance(df_importance, 'importance_features')
        
    tname = target_name
    print('trin')
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # LightGBM parameters
    params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass', # 目的 : 多クラス分類
            'num_class': num_cls,
            'metric': {'multi_error'}, # 評価指標 : 誤り率(= 1-正答率)
            #　他には'multi_logloss'など
    }
    # モデルの学習
    model = lgb.train(params, train_set=lgb_train, # トレーニングデータの指定
                      valid_sets=lgb_eval, # 検証データの指定
                      )
    print('saving importance features as gain.png')
    gain_feature_importance(df, model, target_name=tname, jupyter=on_jupyter)

    print('calculating accuracy')
    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)
    # テストデータを予測する
    y_pred = model.predict_proba(X_test)
    y_pred_max = np.argmax(y_pred, axis=1)  # 最尤と判断したクラスの値にする
    # 精度 (Accuracy) を計算する
    accuracy = sum(y_test == y_pred_max) / len(y_test)
    print('AUC (accuracy) :{}'.format(accuracy))


def on_jupyter_analysis(X, y, num_cls):
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
    params = {
        'num_leaves': 5,
        'metric': ['l1', 'l2'],
        'verbose': -1
    }

    evals_result = {}  # to record eval results for plotting
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100,
                    valid_sets=[lgb_train, lgb_test],
                    feature_name=['f' + str(i + 1) for i in range(X_train.shape[-1])],
                    categorical_feature=[num_cls],
                    evals_result=evals_result,
                    verbose_eval=10)

    def render_metric(metric_name):
        ax = lgb.plot_metric(evals_result, metric=metric_name, figsize=(10, 5))
        plt.show()
        
    try:
        # To enable interactive mode you should install ipywidgets
        # https://github.com/jupyter-widgets/ipywidgets
        from ipywidgets import interact, SelectMultiple
        INTERACTIVE = True
    except ImportError:
        INTERACTIVE = False

    print('loss curve while training')
    if INTERACTIVE:
        # create widget to switch between metrics
        interact(render_metric, metric_name=params['metric'])
    else:
        render_metric(params['metric'][0])


    print('plot decision tree')

    def render_tree(tree_index, show_info, precision=3):
        show_info = None if 'None' in show_info else show_info
        return lgb.create_tree_digraph(gbm, tree_index=tree_index,
                                       show_info=show_info, precision=precision)

    if INTERACTIVE:
        # create widget to switch between trees and control info in nodes
        interact(render_tree,
                 tree_index=(0, gbm.num_trees() - 1),
                 show_info=SelectMultiple(  # allow multiple values to be selected
                     options=['None',
                              'split_gain',
                              'internal_value',
                              'internal_count',
                              'internal_weight',
                              'leaf_count',
                              'leaf_weight',
                              'data_percentage'],
                     value=['None']),
                 precision=(0, 10))
        tree = None
    else:
        tree = render_tree(53, ['None'])
    tree
