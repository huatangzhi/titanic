
# Imports
import pandas as pd
import numpy as np
from pandas import Series,DataFrame

data_train = pd.read_csv("train.csv")
#print data_train.columns
#print data_train.info()
#print data_train.describe()

import matplotlib.pyplot as plt
fig = plt.figure()
fig.set(alpha=0.3)

plt.subplot2grid((2,3), (0,0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u"Survive(1,Survived)")
plt.ylabel(u'Count')
#plt.show()

plt.subplot2grid((2,3), (0,1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.ylabel(u'Count')
plt.title(u'Prank')

plt.subplot2grid((2,3), (0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u'Age')
plt.grid(b=True, which='major', axis='y')
plt.title(u'Survived by age(1, Survived)')

plt.subplot2grid((2,3), (1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u'Age')
plt.ylabel(u'density')
plt.title(u'Age of all Pclass')
plt.legend((u'class_1', u'class_2',u'class_3'), loc='best')

plt.subplot2grid((2,3), (1, 2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u'COunt of Embarked')
plt.ylabel(u'COunt')
#plt.show()

fig = plt.figure()
fig.set(alpha=0.2)

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'Survived':Survived_1, 'unsurvived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u'Pclass of all')
plt.xlabel(u'Survive of all')
plt.ylabel(u'Count')
#plt.show()

fig = plt.figure()
fig.set(alpha=0.2)
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'Survived':Survived_1, u'Unsurvived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u'Survive of all Embarked')
plt.xlabel(u'Embarked')
plt.ylabel(u'Count')
#plt.show()

fig = plt.figure()
fig.set(alpha=0.2)
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({u'male':Survived_m, u'female':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title('Survive by Sex')
plt.xlabel('Count')
plt.show()

fig = plt.figure()
fig.set(alpha=0.65)
plt.title(u'Survive by Pclass and Sex')

ax1 = fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass",  color='#FA2479')
ax1.set_xticklabels([u'Survived', u'Unsurvived'], rotation=0)
ax1.legend([u'female/highclass'], loc='best')

ax2 = fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels([u"Unsurvived", u"Survived"], rotation=0)
plt.legend([u"female/lowclass"], loc='best')
