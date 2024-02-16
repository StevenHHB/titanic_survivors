
---

<div align="center">

# 新手教学演示：泰坦尼克号生存预测项目

## Demo Tutorial：Titanic Survival Prediction with Neural Networks

</div>

---

### 📝 项目介绍 | Project Description

本项目旨在通过使用神经网络对泰坦尼克号乘客的生存情况进行预测。通过分析乘客的个人信息（如性别、年龄、票价等），我们构建了一个模型来预测他们是否能在这场灾难中幸存下来。

This project aims to predict the survival of passengers on the Titanic by using neural networks. By analyzing personal information of passengers such as gender, age, fare, etc., we have built a model to predict their survival in this disaster.

---

### 🛠 技术栈 | Tech Stack

- **PyTorch**: 用于构建和训练神经网络模型。
- **Pandas**: 数据处理和分析。
- **NumPy**: 数值计算。
- **scikit-learn**: 数据预处理。
- **PyTorch**: For building and training the neural network model.
- **Pandas**: For data processing and analysis.
- **NumPy**: For numerical computation.
- **scikit-learn**: For data preprocessing.

---

### 🚀 快速开始 | Quick Start

1. **克隆仓库 | Clone the repository**

```bash
git clone https://your-repository-link
cd your-project-folder
```

2. **安装依赖 | Install Dependencies**

确保已安装上述提到的所有库。

Make sure you have all the mentioned libraries installed.

3. **运行模型 | Run the Model**

```bash
python your-model-script.py
```

---

### 🔍 数据预处理 | Data Preprocessing

- 填补缺失值
- 特征工程（如家庭成员数、头衔等）
- 对类别特征进行独热编码
- 数值特征标准化
- Fill missing values
- Feature engineering (e.g., family size, title)
- One-hot encode categorical features
- Standardize numerical features

---

### 📊 模型架构 | Model Architecture

`TitanicSurvivalNN` 类定义了神经网络的架构，包含三个全连接层和ReLU激活函数，以及Dropout以防过拟合。

The `TitanicSurvivalNN` class defines the architecture of the neural network, containing three fully connected layers with ReLU activation functions, and Dropout to prevent overfitting.

---

### 📈 训练与评估 | Training and Evaluation

- 使用BCELoss作为损失函数。
- Adam优化器。
- 训练100个epochs。
- Uses BCELoss as the loss function.
- Adam optimizer.
- Trains for 100 epochs.

---

### ✅ 预测与提交 | Prediction and Submission

模型训练完成后，使用测试数据进行预测，并生成符合提交格式的CSV文件。

After the model is trained, it makes predictions on the test data and generates a CSV file in the submission format.

---

### 📝 注意事项 | Notes

Mac电脑跑Tensoflow可能会出问题，本次版本中使用Pytorch

You might run into issues runnign Tensorflow on Mac. This tutorial uses Pytorch for the NN

---
