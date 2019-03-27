
import numpy as np
import os 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


le = preprocessing.LabelEncoder()

cwd = os.getcwd()
cwd = cwd +'/'
print('Current Working Directory:\t' + cwd)

def what_is_in_file(filename = 'car.txt'):
    with open(cwd+filename, 'r') as file:
        return file.read()

data = what_is_in_file()


data = data.splitlines()

dataset = []
for line in data:
    temp = line.split(',')
    dataset.append(temp)
    

car_count_4 = 0
unacc_cnt = 0
acc_cnt = 0
good_cnt = 0
vgood_cnt = 0


for car in dataset:
    #print(car[6]) 
    if car[2] == '4':
        car_count_4 += 1
    if(car[6] == 'unacc'):
        unacc_cnt +=1
    elif(car[6] == 'acc'):
        acc_cnt+=1
    elif(car[6] == 'good'):
        good_cnt += 1
    elif(car[6] == 'vgood'):
        vgood_cnt += 1
    
    
print(car_count_4)
print(unacc_cnt, acc_cnt, good_cnt, vgood_cnt)


data_array = np.array(dataset)
print(data_array.shape)

data_array_trans = data_array.reshape(1728*7)
data_array_encode = []
print(data_array[:,0])
for idx in range(7):
    le.fit(data_array[:,idx])
    temp = le.transform(data_array[:,idx])
    data_array_encode.append(temp.tolist())
    print(temp.shape)

data_array_encode = np.array(data_array_encode )
data_array_encode = data_array_encode.transpose()
print(data_array_encode.shape)
print('Labels\n',data_array)
print('Encoded\n',data_array_encode)


data = data_array_encode[:, 0:6]
label = data_array_encode[:, 6]

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)


from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
GaussianNB(priors=None, var_smoothing=1e-09)
y_pred = clf.predict(X_test)
print("Gaussian accuracy : ", accuracy_score(y_test, y_pred) )

from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X_train, y_train)
GaussianNB(priors=None, var_smoothing=1e-09)
y_pred = clf.predict(X_test)
print("Bernoulli accuracy : ", accuracy_score(y_test, y_pred) )   
print('confusion matrix\n', confusion_matrix(y_test, y_pred))

from sklearn.naive_bayes import ComplementNB
clf = ComplementNB()
clf.fit(X_train, y_train)
GaussianNB(priors=None, var_smoothing=1e-09)
y_pred = clf.predict(X_test)
print( "Complement accuracy : ", accuracy_score(y_test, y_pred) )

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)
GaussianNB(priors=None, var_smoothing=1e-09)
y_pred = clf.predict(X_test)
print("Multinomial accuracy : ", accuracy_score(y_test, y_pred) )