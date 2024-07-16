
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
diabetes = pd.read_csv("diabetes.csv")
diabetes.head()
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import classification_report

x = diabetes.iloc[:, :8].values
y = diabetes['y'].values
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.2, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

def KNN(x, y, x_query, k=3):
    m = x.shape[0]
    distances = []
    for i in range(m):
        dis = distance(x_query, x[i])
        distances.append((dis, y[i]))
    distances = sorted(distances)
    distances = distances[:k]
    distances = np.array(distances)
    labels = distances[:, 1]
    uniq_label, counts = np.unique(labels, return_counts=True)
    pred = uniq_label[counts.argmax()]
    return pred

predictions = []
for i in range(len(x_test)):
    p = KNN(x_train, y_train, x_test[i], k=3)
    predictions.append(p)


print(classification_report(y_test, predictions))

def distance(pa,pb):
    return np.sum((pa-pb)**2)**0.5
