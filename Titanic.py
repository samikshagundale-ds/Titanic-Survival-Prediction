print("Titanic Srvival Prediction Project")

import pandas as pd
my_data = pd . read_csv("Titanic-Dataset.csv")
print("DATA LOADED")
print(my_data . head())
print(my_data.isnull().sum())
my_data['Age'] = my_data['Age'].fillna(my_data['Age'].mean())
my_data['Embarked'] = my_data['Embarked'].fillna(my_data['Embarked'].mode()[0])

my_data['FamilySize'] = my_data['SibSp'] + my_data['Parch'] + 1

my_data['IsAlone'] = my_data['FamilySize'].apply(lambda x: 1 if x == 1 else 0)

my_data['IsChild'] = my_data['Age'].apply(lambda x: 1 if x < 18 else 0)

import matplotlib.pyplot as plt
import seaborn as sns
sns.barplot(x='Sex', y='Survived', data=my_data)
plt.title("Survival based on Gender")
plt.savefig("graph1.png")
plt.clf()

my_data['Age'] = my_data['Age'].fillna(my_data['Age'].mean())
my_data['Embarked'] = my_data['Embarked'].fillna(my_data['Embarked'].mode()[0])

my_data['FamilySize'] = my_data['SibSp'] + my_data['Parch'] + 1
my_data['IsAlone'] = my_data['FamilySize'] . apply(lambda x: 1 if x == 1 else 0)
my_data['Isc'] = my_data['FamilySize'] . apply(lambda x: 1 if x == 1 else 0)
my_data['IsChild'] = my_data['Age'].apply(lambda x: 1 if x < 18 else 0)
features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'IsChild']
X = my_data[features]
y = my_data['Survived']

X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,y_train)


y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))

import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x='Pclass', y='Survived', data=my_data)
plt.title("Survival based on Class")
plt.savefig("graph2.png")
plt.clf()

print("Final Model Accuracy:", accuracy_score(y_test, y_pred))
print("Titanic Project Completed Successfully")