Code written by: Shuya Guo


from datetime import datetime
start_time = datetime.now() ## start of training time

## Import the required libraries
from lightgbm import LGBMRegressor as lgb
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold,cross_validate as CVS,train_test_split as TTS
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

## Load the dataset
data=pd.read_excel("AllMOFD.xlsx")
data=data.dropna(axis=0) ## Missing value handling

##Divide the data set
X = data.iloc[:,np.r_[3:12]].values
X = X.astype(np.float64)
Y = data.iloc[:,13].values
Y = Y.astype(np.float64)
X_train, X_test, y_train, y_test = TTS(X, Y, test_size=0.3, random_state=42)

## Standardization
transfer=StandardScaler()
X_train=transfer.fit_transform(X_train)
X_test=transfer.transform(X_test)

## Training LGBM Algorithm model
model = lgb(objective='regression'
                     ,num_leaves=400
                     ,n_estimators=620
                     ,learning_rate=0.1                 
                     ,force_col_wise=True
                     ,colsample_bytree=0.8
                     ,min_child_samples=20
                     ,reg_alpha=0.6
                     ,reg_lambda=0.7
                     ,subsample_for_bin=90000
                     ,random_state=1314
                     ,n_jobs= -1
                     ).fit(X_train, y_train)

## save model
joblib.dump(model, "model/lgbm.pt")

## model prediction
Y_predict1 = model.predict(X_train)
Y_predict2 = model.predict(X_test)

## model evaluation
#training set
MSE=metrics.mean_squared_error(y_train,Y_predict1)
print('MSE_train={}'.format(MSE))
MAE=metrics.mean_absolute_error(y_train,Y_predict1)
print('MAE_train={}'.format(MAE))
RMSE=np.sqrt(metrics.mean_squared_error(y_train,Y_predict1))  
print('RMSE_train={}'.format(RMSE))
R2=metrics.r2_score(y_train,Y_predict1)
print('R2_train={}'.format(R2))

## testing set
MSE=metrics.mean_squared_error(y_test,Y_predict2)
print('MSE_test={}'.format(MSE))
MAE=metrics.mean_absolute_error(y_test,Y_predict2)
print('MAE_test={}'.format(MAE))
RMSE=np.sqrt(metrics.mean_squared_error(y_test,Y_predict2))  
print('RMSE_test={}'.format(RMSE))
R2=metrics.r2_score(y_test,Y_predict2)
print('R2_test={}'.format(R2))

end_time = datetime.now() ##End of training time

## Model training time
print('Duration: {}'.format(end_time - start_time))

## k-fold cross-validation
cv = KFold(n_splits=10 ,shuffle=True,random_state=1402)
scores= CVS(model,X,Y,cv=cv,return_train_score=True
                   ,scoring=('r2', 'neg_mean_squared_error')
                   ,verbose=True,n_jobs=-1)
train_CVS_=(scores['train_neg_mean_squared_error'])
test_CVS_=(scores['test_neg_mean_squared_error'])
train_CVS_r2=scores['train_r2']
test_CVS_r2=scores['test_r2']
train_CVS_RMSE=(abs(train_CVS_)**0.5)
test_CVS_RMSE=(abs(test_CVS_)**0.5)

## Model rendering
ax = plt.scatter(y_test,Y_predict2)
plt.scatter(Y_predict1,y_train)
plt.scatter(Y_predict2,y_test)
