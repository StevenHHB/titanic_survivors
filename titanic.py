# 先import基础的库
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


# 看一眼test data的格式
test_data = pd.read_csv('test.csv')
# print(test_data.head())
train_data = pd.read_csv('train.csv')


# 开始看数据column，很蒙的话直接从gender——submission给的假设开始。假设女生都存活
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
# 得到结果，0.742
print("% of women who survived:", rate_women)
# 假设男生都存活
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
# 得到0.189
print(" % of men survived:", rate_men)


# 是用基础random forest预测
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame(
    {'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
