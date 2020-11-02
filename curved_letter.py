from sklearn.ensemble import RandomForestClassifier as rc
from sklearn.svm import SVC as s
from sklearn.linear_model import  SGDClassifier
import pandas as pd
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score


data = pd.read_csv('C://Users//User//Curved.csv')

feed = pd.read_csv('C://Users//User//First_feed11.csv')

#train_X, test_x,train_y, test_y = train_test_split(data.iloc[:,1:],data.iloc[:,0],test_size=0.4, random_state=7)


clf3 = neighbors.KNeighborsClassifier(2)


model3 = clf3.fit(data.iloc[:,1:], data.iloc[:,0])


pred3 = model3.predict(feed)

df = pd.DataFrame(pred3)

df.to_csv('c1.csv')









