import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


iris = datasets.load_iris()

#iris
#iris.keys()

#iris.data
#iris.target
#len(iris.data)
#len(iris.target)

#a1=[[1,2,3],[4,5,6]]
#len(a1) #output: 2

#import numpy as np
#a1=np.array([[1,2,3],[4,5,6]])


print(iris);

x=iris.data
y=iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#Define model
knn=KNeighborsClassifier()

#Train model with the trainning data
knn.fit(x_train, y_train)

#Predict data with the trainned model
y_predict=knn.predict(x_test)

#Check the predict result
#1. the kind of result
np.unique(y_predict)
#2. Group by to count
Counter(y_test)
Counter(y_predict)
#3. Accuracy
accuracy_score(y_test, y_predict)
accuracy_score(y_predict, y_test)
accuracy_score(y_test, y_predict, normalize=True)
accuracy_score(y_test, y_predict, normalize=False)

xindex = np.arange(50);

plt.subplot(2,2,1);
plt.plot(xindex, y_predict, label='Predit', color='red');
plt.title('Predit Result');
plt.legend(loc='upper left')

plt.subplot(2,2,2);
plt.plot(xindex, y_test, label='Real', color='green');
plt.title('Real Result');
plt.legend(loc='upper left')

plt.subplot(2,2,3);
plt.plot(xindex, y_predict, label='Predict', color='red');
plt.plot(xindex, y_test, label='Real', color='green');
plt.title('Compare Predict and Real Result');
plt.legend(loc='upper left')

plt.show();
