import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

yorumlar = pd.read_csv('Restaurant_Reviews_delimiter_adjusted.csv', sep=';')

import re
import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

#nltk.download('stopwords')
from nltk.corpus import stopwords

#Preprocessing (Önişleme)
derlem = []
for i in range(1000):
    yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)
    
#Feautre Extraction ( Öznitelik Çıkarımı)
#Bag of Words (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(derlem).toarray() # bağımsız değişken
y = yorumlar.iloc[:,1].values # bağımlı değişken
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print('gausianNB')
print(cm)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('KNN')
print(cm)

from sklearn.svm import SVC
svc = SVC(kernel='rbf',probability=True)
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=20, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)

#Visualizer


from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
y_test = lb.fit_transform(y_test)

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
#Prediction probabilities
r_probs = [0 for _ in range(len(y_test))]
rfc_probs = rfc.predict_proba(X_test)
knn_probs = knn.predict_proba(X_test)
svc_probs = svc.predict_proba(X_test)
dtc_probs = dtc.predict_proba(X_test)
gnb_probs = gnb.predict_proba(X_test)


#Probabilities for the positive outcome is kept.
rfc_probs = rfc_probs[:, 1]
knn_probs = knn_probs[:, 1]
svc_probs = svc_probs[:, 1]
dtc_probs = dtc_probs[:, 1]
gnb_probs = gnb_probs[:, 1]


r_probs = pd.DataFrame(r_probs)
rfc_probs = pd.DataFrame(rfc_probs)
knn_probs = pd.DataFrame(knn_probs)
svc_probs = pd.DataFrame(svc_probs)
dtc_probs = pd.DataFrame(dtc_probs)
gnb_probs = pd.DataFrame(gnb_probs)

y_test = pd.DataFrame(y_test)
#Calculate AUROC
r_auc = roc_auc_score(y_test, r_probs, multi_class = 'ovr')
rfc_auc = roc_auc_score(y_test, rfc_probs, multi_class = 'ovr')
knn_auc = roc_auc_score(y_test, knn_probs, multi_class = 'ovr')
svc_auc = roc_auc_score(y_test, svc_probs, multi_class = 'ovr')
dtc_auc = roc_auc_score(y_test, dtc_probs, multi_class = 'ovr')
gnb_auc = roc_auc_score(y_test, gnb_probs, multi_class = 'ovr')


#Calculate ROC curve
r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
rfc_fpr, rfc_tpr, _ = roc_curve(y_test, rfc_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
svc_fpr, svc_tpr, _ = roc_curve(y_test, svc_probs)
dtc_fpr, dtc_tpr, _ = roc_curve(y_test, dtc_probs)
gnb_fpr, gnb_tpr, _ = roc_curve(y_test, gnb_probs)


#Plot the ROC curve
plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
plt.plot(rfc_fpr, rfc_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % rfc_auc)
plt.plot(knn_fpr, knn_tpr, marker='.', label='KNeighbors (AUROC = %0.3f)' % knn_auc)
plt.plot(svc_fpr, svc_tpr, marker='.', label='Support Vector (AUROC = %0.3f)' % svc_auc)
plt.plot(dtc_fpr, dtc_tpr, marker='.', label='Decision Tree (AUROC = %0.3f)' % dtc_auc)
plt.plot(gnb_fpr, gnb_tpr, marker='.', label='Naive Bayes (AUROC = %0.3f)' % gnb_auc)

# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()











