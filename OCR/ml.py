from sklearn.ensemble import RandomForestClassifier as rc
from sklearn.svm import SVC as s
from sklearn.linear_model import  SGDClassifier
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC


wordlist = pd.read_csv('C://Users//User//Word_list.csv', engine='python')

data = pd.read_csv('C://Users//User//data_image_reco.csv')

feed = pd.read_csv('C://Users//User//Final_feed11.csv')

data = data.sample(frac=1).reset_index(drop=True)


clf1 = OneVsOneClassifier(SGDClassifier(random_state=48))
clf2 = GaussianNB()
#clf3 = neighbors.KNeighborsClassifier(7)
clf4 = rc()
clf5 = SVC(kernel='rbf',C=2.75, gamma=0.005, class_weight='balanced')

model1 = clf1.fit(data.iloc[:,1:], data.iloc[:,0])
model2 = clf2.fit(data.iloc[:,1:], data.iloc[:,0])
#model3 = clf3.fit(data.iloc[:,1:], data.iloc[:,0])
model4 = clf4.fit(data.iloc[:,1:], data.iloc[:,0])
model5 = clf5.fit(data.iloc[:,1:], data.iloc[:,0])
pred1 = model1.predict(feed)
pred2 = model2.predict(feed)
#pred3 = model3.predict(feed)
pred4 = model4.predict_proba(feed)
pred5 = model4.predict(feed)


word = pd.DataFrame()
word.loc[0,"Word"] = str(pred5)
wordlist = wordlist.append(word, sort=False)

wordlist.to_csv('Word_list.csv')





