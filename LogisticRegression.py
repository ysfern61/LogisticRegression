#sınıflandırma modelidir.
#iki seçenek için kullanılır
#1 ve 0 lar kullanılır.
#sigmoid() function kullanır.

import pandas as pd
import numpy
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#veri setimizi pandas kullanarak yükleme
df=pd.read_csv("logistic_regression.csv",sep=";")

#verileri ekrana çizdirme
plt.xlabel('Yas')
plt.ylabel('Sigorta (yok:0 / var:1)')
plt.scatter(df.yas,df.sigorta,color='red',marker='+')


#train ve test verilerimizi ana veri setimizi kullanıcaz
#train:%80 test:%20

X_train,X_test,Y_train,Y_test=train_test_split(df[['yas']],df.sigorta,train_size=0.8)


#model nesnesini oluşturma
model=LogisticRegression()
model.fit(X_train,Y_train)


#tahmin yapma
X_test
Y_predicted=model.predict(X_test)
model.predict_proba(X_test)












