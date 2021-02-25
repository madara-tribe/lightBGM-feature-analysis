from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def normarize(X, MinMax=True):
    if MinMax:
        Xs = MinMaxScaler().fit_transform(X)
    else:
        sc = preprocessing.StandardScaler()
        Xs = sc.fit_transform(X)
    return Xs



def MultiOutputRegressor(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=2)

    params={'learning_rate': 0.5,
        'objective':'mae', 
        'metric':'mae',
        'num_leaves': 9,
        'verbose': 0,
        'bagging_fraction': 0.7,
        'feature_fraction': 0.7
       }
    reg = MultiOutputRegressor(lgb.LGBMRegressor(**params, n_estimators=500))

    reg.fit(X_train, y_train)
    y_pred =reg.predict(X_test)
    print(reg.score(X_test, y_test))
    
    print('evaluation')
    # 完全に一致する場合に 1 となり、1 に近いほど精度の高い予測が行えていることを表し
    r2 = r2_score(y_test, y_pred)
    print('R**2 score', r2)
    # RMSE が 0 に近いほど見積もられる予測誤差が小さい、すなわち予測精度が高いことを表す
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE score', rmse)

    # # RMSE が 0 に近いほど見積もられる予測誤差が小さい、すなわち予測精度が高いことを表し
    mae = mean_absolute_error(y_test, y_pred)
    print('MAE score', mae)
    
def main():
  # 予測変数
  y = dft.values 
  print(y.shape)
  # 説明変数
  tname='column'
  X = df.drop(tname, axis=1)  #.values # 説明変数(target以外の特徴量) 
  X = normarize(X, MinMax=None)
  print(X.shape, y.shape, X.min(), X.max(), y.max(), y.min())
  MultiOutputRegressor(X, y)
