import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
dataset = pd.read_csv('Downloads/Titanic-Dataset.csv')
dataset
dataset.info()
dataset.corr()
sns.heatmap(dataset.corr());
plt.figure(figsize=(16,6))
heatmap=sns.heatmap(dataset.corr(),vmin=-1,vmax=1,annot=True)
heatmap.set_title('Correlation Heatmap',fontdict={'fontsize':12},pad=12);
plt.figure(figsize=(5,5))
dataset.Sex.value_counts().plot(kind='pie')
dataset.Sex.value_counts()
dataset.groupby('Sex').Survived.mean().plot(kind='bar')
print(dataset.groupby('Sex').Survived.value_counts())
not_survived_fare=dataset['Fare'][dataset['Survived']==0]
survived_fare=dataset['Fare'][dataset['Survived']==1]

plt.figure(figsize=(12,6))
plt.subplot(121)
not_survived_fare.plot(kind='hist',title='People who didn\'t Survived')

plt.subplot(122)
survived_fare.plot(kind='hist',title='People who Survived')

print('Mean ticket fare of people who didn\'t survived: ',not_survived_fare.mean())
print('Mean ticket fare of people who survived: ',survived_fare.mean())
dataset.plot(kind='box')
