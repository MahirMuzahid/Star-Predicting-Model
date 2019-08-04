#import module
from sklearn import svm
import pandas as pd

#Getting Data
data = pd.read_csv('pulsar_stars.csv')
col= [' Mean of the integrated profile',' Standard deviation of the integrated profile',' Excess kurtosis of the integrated profile',
      ' Skewness of the integrated profile',' Mean of the DM-SNR curve',' Standard deviation of the DM-SNR curve',
      ' Excess kurtosis of the DM-SNR curve',' Skewness of the DM-SNR curve']

#Spliting data in X and y
X = data[col]
y = data['target_class']

#Splitting data for test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=0)

#Useing SVM to train model
model = svm.SVC(kernel = 'linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Getting accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)



