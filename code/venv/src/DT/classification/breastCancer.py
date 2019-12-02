#b1: load_breast_cancer => dataset
#b2: train_test_split => chia dữ liệu
#b3: độ chính xác của
#b4: tạo file dot
#b5: tạo file png
#b6: mức độ quan trọng của thuộc tính


#import thư viện
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer #dataset

cancer = load_breast_cancer()
print('cancer.keys():\n{}'.format(cancer.keys()))
print('Kích thước dữ liệu:\n{}'.format(cancer.data.shape))
print('Các thuộc tính:\n{}'.format(cancer.feature_names))
print('Các lớp:\n{}'.format(cancer.target_names))


#print('data:\n{}'.format(cancer.data))
#print('target:\n{}'.format(cancer.target))

#chia dữ liệu
#trainning 80%
#testing 20%
from sklearn.model_selection import train_test_split
x=cancer.data #dữ liệu đầu vào
y=cancer.target #kết quả
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)
#stratify: phân tầng ???
#random_state  whenever you execute your code a new random value is generated and the train and test datasets would have different values each time.
#However, if you use a particular value for random_state(random_state = 1 or any other value) everytime the result will be same,i.e, same values in train and test datasets. Refer below code:
print('x_train:{}\n'.format(x_train.shape))
print('x_test:{}\n'.format(x_test.shape))

#import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import accuracy_score
from sklearn.metrics import  accuracy_score #accuracy_score: độ chính xác
# Cây quyết định
#tree=DecisionTreeClassifier(max_depth=2,random_state=42)
tree = DecisionTreeClassifier(random_state=42) #30
#tree=DecisionTreeClassifier(max_depth=4,random_state=42)
print('{}\n'.format(tree.fit(x_train,y_train)))

#y_pred = tree.predict(x_test)  #predict: dự đoán
print('Độ chính xác của tập huấn luyện: {:.4f}'.format(tree.score(x_train,y_train)))
print('Độ chính xác của tập kiểm tra: {:.4f}'.format(tree.score(x_test,y_test)))


#biểu thị cây nhị phân
from sklearn.tree import export_graphviz
import  graphviz
export_graphviz(tree, out_file='tree_classifier.dot',
                    feature_names=cancer.feature_names, class_names=cancer.target_names,
                    filled=True,  impurity=False)
#chuyển file dot sang file ảnh
import pydot
(graph,)=pydot.graph_from_dot_file('tree_classifier.dot')
graph.write_png('tree_classifier.png')

#mức độ quan trọng của các thuộc tính
print(tree.feature_importances_)

# Import matplotlib
import matplotlib.pyplot as plt
features = cancer.feature_names
n = len(features)
plt.figure(figsize=(8,10))
plt.barh(range(n),tree.feature_importances_)
plt.yticks(range(n),features)
plt.title('Muc do quan trong cac thuoc tinh')
plt.ylabel('Cac thuoc tinh')
plt.xlabel('Muc do')
plt.show()