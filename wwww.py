import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# 读取数据
data = pd.read_csv('./data/TrainingData.txt')

# 分离特征和标签
X = data.iloc[:, :-1].values  # 所有行，除了最后一列之外的所有列（特征）
y = data.iloc[:, -1].values   # 所有行，最后一列（标签）

# 数据标准化
sc = StandardScaler()
X_std = sc.fit_transform(X)  # 标准化

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

# 训练线性支持向量机
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train, y_train)

# 预测测试集结果
y_pred = svm.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")
# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 使用seaborn绘制混淆矩阵的热力图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 假设你的新数据集是new_data.csv，并且它的格式与训练数据集相同
new_data = pd.read_csv('./data/TestingData.txt')

# 分离特征
new_X = new_data.iloc[:, :].values  # 所有行

# 数据标准化，使用之前训练时的StandardScaler实例
new_X_std = sc.transform(new_X)  # 注意这里使用的是transform，而不是fit_transform

# 使用训练好的模型进行预测
new_y_pred = svm.predict(new_X_std)

# 将预测结果与新数据集的索引对应起来，以便知道每个预测结果对应的是哪个数据
predictions = pd.DataFrame(new_y_pred, index=new_data.index, columns=['Predicted_Label'])

# 打印预测结果
print(predictions)

# 如果需要，可以将预测结果保存到CSV文件
predictions.to_csv('./data/predictions.txt')
test_new = pd.DataFrame(new_X, columns=data.columns[:-1])  # 假设新数据集的特
test_new['Predicted_Label'] = new_y_pred  # 在DataFrame中添加预测标签列

# 保存到CSV文件
test_new.to_csv('./data/test_new.csv', index=False)  # 保存时不包含行索引
