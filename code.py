# --------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score,mean_squared_error

## Load the data
data=pd.read_csv(path)

## Split the data and preprocess
data.head(2)
train=data[data.source=='train']
test=data[data.source=='test']
train_sub=train[['Item_Weight', 'Item_MRP', 'Item_Visibility','Item_Outlet_Sales']]
test_sub=test[['Item_Weight', 'Item_MRP', 'Item_Visibility','Item_Outlet_Sales']]
## Baseline regression model
X_sub_train, X_sub_test, y_sub_train, y_sub_test=train_test_split(train_sub.iloc[:,:-1],train_sub.Item_Outlet_Sales,test_size=0.3,random_state=0)
Lingreg=LinearRegression()
Lingreg.fit(X_sub_train,y_sub_train)
y_sub_pred=Lingreg.predict(X_sub_test)
rmse=np.sqrt(mean_squared_error(y_sub_test,y_sub_pred))
r2=r2_score(y_sub_test,y_sub_pred)
## Effect on R-square if you increase the number of predictors
print(rmse)
print(r2)
## Effect of decreasing feature from the previous model
train.drop(['Item_Identifier'],inplace=True,axis=1)
train=train[['Unnamed: 0', 'Item_MRP', 'Item_Visibility',
'Item_Weight', 'source', 'Outlet_Years', 'Item_Fat_Content_0',
'Item_Fat_Content_1', 'Item_Fat_Content_2', 'Outlet_Location_Type_0',
'Outlet_Location_Type_1', 'Outlet_Location_Type_2', 'Outlet_Size_0',
'Outlet_Size_1', 'Outlet_Size_2', 'Outlet_Type_0', 'Outlet_Type_1',
'Outlet_Type_2', 'Outlet_Type_3', 'Item_Type_Combined_0',
'Item_Type_Combined_1', 'Item_Type_Combined_2', 'Outlet_Identifier_0',
'Outlet_Identifier_1', 'Outlet_Identifier_2', 'Outlet_Identifier_3',
'Outlet_Identifier_4', 'Outlet_Identifier_5', 'Outlet_Identifier_6',
'Outlet_Identifier_7', 'Outlet_Identifier_8', 'Outlet_Identifier_9','Item_Outlet_Sales']]
train.drop(['source'],inplace=True,axis=1)
X_train, X_test, y_train, y_test=train_test_split(train.iloc[:,:-1],train.Item_Outlet_Sales,test_size=0.3,random_state=0)
Lingreg2=LinearRegression()
Lingreg2.fit(X_train,y_train)
y_pred=Lingreg2.predict(X_test)
rmse2=np.sqrt(mean_squared_error(y_test,y_pred))
r22=r2_score(y_test,y_pred)
## Effect on R-square if you increase the number of predictors
print(rmse2)
print(r22)
## Detecting hetroskedacity


## Model coefficients


## Ridge regression


## Lasso regression


## Cross vallidation



