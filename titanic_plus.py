# 更高难度一些，使用神经网络试一试
# 先引入基础的库，mac使用pytorch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# 再入测试和训练数据
test_data = pd.read_csv('test.csv')
train_data = pd.read_csv('train.csv')

# 完善缺失column
train_data['Age'].fillna(train_data['Age'].median())
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
train_data.drop('Cabin', axis=1)

test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'].fillna(test_data['Fare'].median())
test_data.drop('Cabin', axis=1)

# 创建新的feature和自变因素
train_data['Title'] = train_data['Name'].apply(
    lambda x: x.split(',')[1].split('.')[0].strip())
test_data['Title'] = test_data['Name'].apply(
    lambda x: x.split(',')[1].split('.')[0].strip())
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']

train_data = pd.get_dummies(
    train_data, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
test_data = pd.get_dummies(
    test_data, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

# normalize量化数据
scaler = StandardScaler()
num_features = ['Age', 'Fare', 'FamilySize']
train_data[num_features] = scaler.fit_transform(train_data[num_features])
test_data[num_features] = scaler.transform(test_data[num_features])

# 准备训练
X_train = train_data.drop(
    ['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1)
y_train = train_data['Survived']
X_test = test_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1).reindex(
    columns=X_train.columns, fill_value=0)

# 确定所有数据都是numeric属性
X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

# print(X_train.dtypes)
# print(X_test.dtypes)
# 将boolean变量编程numeric
for col in X_train.columns:
    if X_train[col].dtype == 'bool':
        X_train[col] = X_train[col].astype(int)
    if X_test[col].dtype == 'bool':
        X_test[col] = X_test[col].astype(int)


# 转化pytorch tensor
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

# 创建data loader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 创建model class


class TitanicSurvivalNN(nn.Module):
    def __init__(self, num_features):
        super(TitanicSurvivalNN, self).__init__()
        self.layer1 = nn.Linear(num_features, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.layer3(x))
        return x


# x train features编辑，model initialize
num_features = X_train.shape[1]
model = TitanicSurvivalNN(num_features)

# 损失
criterion = nn.BCELoss()

# 优化
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型


def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            # Make sure labels are the correct shape for BCELoss
            labels = labels.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')


# 训练模型
train_model(model, train_loader, criterion, optimizer)

# 预测


def predict(model, X_test_tensor):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
    predicted_labels = (predictions.squeeze() > 0.5).long()
    return predicted_labels


predictions = predict(model, X_test_tensor)

# 生成提交结果
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions.numpy()
})

submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv('submission.csv', index=False)
print("Submission file created.")
