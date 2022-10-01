#подключение необходимых модулей
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')
print(df1.head())
print(df2.head())

#работа с данными
def fill_genre(Genre):
    if Genre == 'Male':
        return 1
    return 0
df1['Genre'] = df1['Genre'].apply(fill_genre)
df2['Genre'] = df2['Genre'].apply(fill_genre)


def fill_age(Age):
    if Age >= 40:
        return 1
    return 0
df1['Age'] = df1['Age'].apply(fill_age)
df2['Age'] = df2['Age'].apply(fill_age)


def fill_income(Income):
    if Income >= 20:
        return 1
    return 0
df1['Income'] = df1['Income'].apply(fill_income)
df2['Income'] = df2['Income'].apply(fill_income)

df1.info()
df2.info()

#обучение

X_train = df1.drop('Spending Score', axis=1)
y_train = df1['Spending Score']
X_test = df2

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
ID = df2['CustomerID']
result = pd.DataFrame({'CustomerID': ID, 'Spending Score': y_pred})
result.to_csv('answer.csv', index=False)

