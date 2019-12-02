import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("./data/auto-mpg.csv")
print(data.head())

x=data['displacement']
y=data['mpg']

plt.scatter(x,y,c='blue')
plt.xlabel('displacement')
plt.ylabel('mpg')
plt.title('Hồi quy')
plt.show()

from sklearn.tree import  DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

x = x.values.reshape(-1,1) # chỗ này để làm gì????
y = y.values.reshape(-1,1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=3)
dt = DecisionTreeRegressor(max_depth=3, min_samples_leaf=1, random_state=3)
print(dt.fit(x_train,y_train))

y_pred = dt.predict(x_test)
mse_dt = MSE(y_test,y_pred)
print(np.sqrt(mse_dt))

xx= np.linspace(min(x),max(x),400).reshape(-1,1)

plt.scatter(x,y,c="blue")
plt.plot(xx,dt.predict(xx), color="red", linewidth=2)
plt.xlabel('displacement')
plt.ylabel('mpg')
plt.title('Hồi quy')
plt.show()

from sklearn.tree import export_graphviz
export_graphviz(dt, out_file='tree.dot', feature_names=['displacement'])
import pydot
(graph,)=pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')

#from IPython.display import Image
#Image(filename='tree.png')


#ví dụ 2

data = pd.read_csv('./data/autompg.csv')
features = data.columns.values[1:][:-1]
print(features)
x = data[features].values
y = data['mpg']
regressor = DecisionTreeRegressor(max_depth=3)
print(regressor.fit(x, y))
from sklearn.tree import export_graphviz
export_graphviz(regressor, out_file='tree_multi.dot', feature_names=features)
import pydot
(graph,)=pydot.graph_from_dot_file('tree_multi.dot')
graph.write_png('tree_multi.png')

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y, regressor.predict(x)))

print(MSE(y, regressor.predict(x)))