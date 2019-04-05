import pandas as pd
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
column_names=["Sample code number","Clump Thickness","Uniformity of Cell Size" ,
              "Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size",
              "Bare Nuclei","Bland Chromatin",   "Normal Nucleoli", "Mitoses","Class"]
data=pd.read_csv(r"https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",names=column_names)
#data=pd.read_csv(r"C:\Users\dell\Desktop\test\breastCancer.csv",names=column_names)
data=data.replace(to_replace="?",value=np.nan)
data.head(20)
data=data.dropna(how='any')
data.shape
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)
y_test.value_counts()
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
## standardize the data
## for training data,fit the parameters
## for test data,use the parameters fiteds directly
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
# use the logistic Regression to get the parameters 
lr=LogisticRegression()
lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)
# use the sgdc methode to get the parameters
sgdc=SGDClassifier()
sgdc.fit(X_train,y_train)
sgdc_y_predict=sgdc.predict(X_test)
# Analyse the result of the calssificaiton: Precision and Recall
from sklearn.metrics import classification_report
print("Accuracy of lr classifier",lr.score(X_test,y_test))
print(classification_report(y_test,lr_y_predict,target_names=["bengin","malignant"]))
# Analyse the result of SGD classfier
print("Accuarcy of SGD Classfier:",sgdc.score(X_test,y_test))
print(classification_report( y_test,sgdc_y_predict,target_names=["benign","malignant"]))
"""
The first methode can give the analytical solution of the parameters lgistic regression model
which is more accurate but it takes more time 
The seconde methde can give a solution calculted with the methode sgd 
which is more powerfull for the big data
"""

## The first methode can give the analytical solution of the parameters lgistic regression model
## which is more accurate but it takes more time 
## The seconde methde can give a solution calculted with the methode sgd 
## which is more powerfull for the big data

