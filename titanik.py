import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import numpy as np


def data_unload():

    train = pd.read_csv("train.csv", delimiter = ",")
    test = pd.read_csv("test.csv", delimiter = ",")

    # test["Survived"] = np.zeros(len(test))
    # test["Survived"] = -1

    # info = [train, test]
    # DataAll = pd.concat(info, ignore_index = "True")
    return train, test

train, test = data_unload()

res = {}
passenger_id = []


train_copy = train.copy()


for index, row in train_copy.iterrows():
    if(row["Sex"] == "female"):
        row["Sex"]=1
    else:
        row["Sex"]=0
for index, row in train_copy.iterrows():
    if(row["Embarked"] == "S"):
        row["Embarked"] = 0
    elif (row["Embarked"] == "C"):
        row["Embarked"] = 1
    elif(row["Embarked"] == "Q"):
        row["Embarked"] = 2
    else:
        row["Embarked"] = 0
train_features = train_copy[["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"]]
train_target = train_copy["Survived"]

for index, row in train_features.iterrows():
    if(row["Sex"] == "female"):
        sex_val=1
    else:
        sex_val=0
    train_features.set_value(index,'Sex',sex_val)
for index, row in train_features.iterrows():
    if(row["Embarked"] == "S"):
        embark_val = 0
    elif (row["Embarked"] == "C"):
        embark_val = 1
    elif(row["Embarked"] == "Q"):
        embark_val = 2
    else:
        embark_val = 0
    train_features.set_value(index,'Embarked',embark_val)

test_features = test[["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"]]
for index, row in test_features.iterrows():
    if(row["Sex"] == "female"):
        sex_val=1
    else:
        sex_val=0
    test_features.set_value(index,'Sex',sex_val)

test_features["Fare"] = test_features["Fare"].fillna(0.0)

for index, row in test_features.iterrows():
    if(row["Embarked"] == "S"):
        embark_val = 0
    elif (row["Embarked"] == "C"):
        embark_val = 1
    elif(row["Embarked"] == "Q"):
        embark_val = 2
    else:
        embark_val = 0
    test_features.set_value(index,'Embarked',embark_val)

logistic_regression_model = GradientBoostingClassifier()
logistic_regression_model.fit(train_features, train_target)


passengerss = test.iloc[:,[0]]
#
# print(passengerss)

for index, row in passengerss.iterrows():
    passenger_id.append(row["PassengerId"])

passenger_id = np.array(passenger_id)



if_alive = []
if_alive = (logistic_regression_model.predict(test_features))

if_alive = np.array(if_alive)

passenger_id.reshape(1,418)
if_alive.reshape(1,418)

print(passenger_id.shape)
print(if_alive.shape)

res["PassengerId"] = passenger_id
res["Survived"] = if_alive

print(res)

df = pd.DataFrame(res, columns = ["PassengerId", "Survived"])
export = df.to_csv("submissions.csv", index = None, header = "True")

print(df)
