import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
Iris_data = pd.read_csv('Iris.csv')
Iris_data.head(10)
Iris_data.info()
Iris_data.describe()
Iris_data.Species.value_counts()
plt.scatter(Iris_data['SepalLengthCm'],Iris_data['SepalWidthCm'])
plt.show()
sns.set_style('whitegrid')
sns.FacetGrid(Iris_data, hue = 'Species')    .map(plt.scatter, 'SepalLengthCm','SepalWidthCm')    .add_legend()
plt.show()
sns.pairplot(Iris_data.drop(['Id'],axis=1), hue='Species')
plt.show()
Iris_data['Sepal_diff'] = Iris_data['SepalLengthCm']-Iris_data['SepalWidthCm']
Iris_data['petal_diff'] = Iris_data['PetalLengthCm']-Iris_data['PetalWidthCm']
Iris_data
sns.set_style('whitegrid')
sns.FacetGrid(Iris_data,hue='Species')   .map(plt.scatter,'Sepal_diff','petal_diff')   .add_legend()
plt.show()  
sns.set_style('whitegrid')
sns.FacetGrid(Iris_data,hue='Species')   .map(sns.distplot,'petal_diff')   .add_legend()
plt.show() 
Iris_data['Sepal_petal_len_diff'] = Iris_data['SepalLengthCm']-Iris_data['PetalLengthCm']
Iris_data['Sepal_petal_width_diff'] = Iris_data['SepalWidthCm']-Iris_data['PetalWidthCm']
Iris_data
sns.set_style('whitegrid')
sns.FacetGrid(Iris_data,hue='Species')   .map(plt.scatter,'Sepal_petal_len_diff','Sepal_petal_width_diff')   .add_legend()
plt.show()
sns.set_style('whitegrid')
sns.FacetGrid(Iris_data,hue='Species')   .map(sns.distplot,'PetalLengthCm')   .add_legend()
plt.show()
Iris_data['Sepal_petal_len_wid_diff'] = Iris_data['SepalLengthCm']-Iris_data['PetalWidthCm']
Iris_data['Sepal_petal_wid_len_diff'] = Iris_data['SepalWidthCm']-Iris_data['PetalLengthCm']
Iris_data
sns.set_style('whitegrid')
sns.FacetGrid(Iris_data,hue='Species')   .map(plt.scatter,'Sepal_petal_wid_len_diff','Sepal_petal_len_wid_diff')   .add_legend()
plt.show()
sns.set_style('whitegrid')
sns.FacetGrid(Iris_data,hue='Species')   .map(sns.distplot,'Sepal_petal_wid_len_diff')   .add_legend()
plt.show()
sns.pairplot(Iris_data[['Species', 'Sepal_diff', 'petal_diff', 'Sepal_petal_len_diff',       'Sepal_petal_width_diff', 'Sepal_petal_len_wid_diff',       'Sepal_petal_wid_len_diff']], hue='Species')
plt.show()
Iris_data.drop(['Id'],axis=1,inplace=True)
for i in Iris_data.columns:
    if i == 'Species':
        continue
    sns.set_style('whitegrid')
    sns.FacetGrid(Iris_data,hue='Species')    .map(sns.distplot,i)    .add_legend()
    plt.show()

# # Building Classification Model

from sklearn import tree
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
X = Iris_data[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm', 'PetalWidthCm','Sepal_petal_wid_len_diff','Sepal_petal_width_diff']]
y = Iris_data['Species']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.30, random_state=42)
Xt, Xcv, Yt, Ycv = train_test_split(Xtrain, Ytrain, test_size=0.10, random_state=42)
Iris_clf = DecisionTreeClassifier(criterion='gini',min_samples_split=2)
Iris_clf.fit(Xt, Yt)
tree.plot_tree(Iris_clf)
dot_data = tree.export_graphviz(Iris_clf, out_file=None)
graph = graphviz.Source(dot_data)
graph
print('Accuracy score is:',cross_val_score(Iris_clf, Xt, Yt, cv=3, scoring='accuracy').mean())
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
Y_hat = Iris_clf.predict(Xcv)
print('Accuracy score for validation test data is:',accuracy_score(Ycv, Y_hat))
multilabel_confusion_matrix(Ycv , Y_hat)
YT_hat = Iris_clf.predict(Xtest)
YT_hat
print('Model Accuracy Score on totally unseen data(Xtest) is:',accuracy_score(Ytest, YT_hat)*100,'%')
multilabel_confusion_matrix(Ytest , YT_hat)
Iris_Fclf = DecisionTreeClassifier(criterion='gini',min_samples_split=2)
Iris_Fclf.fit(Xtrain, Ytrain)
tree.plot_tree(Iris_Fclf)
dot_data = tree.export_graphviz(Iris_Fclf, out_file=None)
graph = graphviz.Source(dot_data)
graph
YT_Fhat = Iris_Fclf.predict(Xtest)
YT_Fhat
print('Model Accuracy Score on totally unseen data(Xtest) is:',accuracy_score(Ytest, YT_Fhat)*100,'%')
multilabel_confusion_matrix(Ytest , YT_Fhat)
Test_point = [[5.4,3.0,4.5,1.5,-1.5,1.5],
             [6.5,2.8,4.6,1.5,-1.8,1.3],
             [5.1,2.5,3.0,1.1,-0.5,1.4],
             [5.1,3.3,1.7,0.5,1.6,2.8],
             [6.0,2.7,5.1,1.6,-2.4,1.1],
             [6.0,2.2,5.0,1.5,-2.8,0.7]]
print(Iris_Fclf.predict(Test_point))
