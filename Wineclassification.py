import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import time

white_wine = pd.read_csv("winequality-white.csv")
red_wine = pd.read_csv("winequality-red.csv")

plt.figure(figsize=(16, 7))

# Finding the correlation between the columns in the dataset
sns.heatmap(white_wine.corr(), annot=True, fmt='0.2g', linewidths=1)

sns.heatmap(red_wine.corr(), annot=True, fmt='0.2g', linewidths=1)

grid_dict = {}
acc_dict = {}

# Dropping residual sugar
white_wine = white_wine.drop('residual sugar', axis=1)
# Assigning X to data and y to target
y = white_wine['quality']
X = white_wine.drop('quality', axis=1)


# Splitting the data into training and test set with a 3:1 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Normalizing the data
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)

# K Nearest Neighbour Classifier
start_time = time.time()
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
# Accuracy for pre GridSearchCV classifier
knn_acc1 = accuracy_score(y_test, knn.predict(X_test))
# Parameters for GridSearchCV
param_grid = {'n_neighbors': [1, 2, 5, 10, 20],
			  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
			  'weights': ['uniform', 'distance']
			  }

grid = GridSearchCV(knn, param_grid, refit=True, verbose=3, n_jobs=-1)
grid.fit(X_train, y_train)
grid_predictions = grid.predict(X_test)
knn2 = KNeighborsClassifier(n_neighbors=grid.best_estimator_.n_neighbors, algorithm=grid.best_estimator_.algorithm,
							weights=grid.best_estimator_.weights)
knn2.fit(X_train, y_train)
# Accuracy for pre GridSearchCV classifier
knn_acc2 = accuracy_score(y_test, knn2.predict(X_test))
grid_dict["knn_acc"] = grid.best_params_
# Inserting both accuracy scores for comparison
acc_dict["KNN"] = [knn_acc1, knn_acc2]

# DecisionTree Classifier
start_time = time.time()
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
# Accuracy for pre GridSearchCV classifier
dt_acc1 = accuracy_score(y_test, dt.predict(X_test))
# Parameters for GridSearchCV
param_grid = {'criterion': ['gini', 'entropy'],
			  'splitter': ['best', 'random'],
			  'max_depth': [4, 5, 7, 9, 10, 11, None]}

grid = GridSearchCV(dt, param_grid, refit=True, verbose=3, n_jobs=-1)
grid.fit(X_train, y_train)
grid_predictions = grid.predict(X_test)
dt2 = DecisionTreeClassifier(criterion=grid.best_estimator_.criterion, splitter=grid.best_estimator_.splitter,
							 max_depth=grid.best_estimator_.max_depth)
dt2.fit(X_train, y_train)
# Accuracy for post GridSearchCV classifier
dt_acc2 = accuracy_score(y_test, dt2.predict(X_test))
grid_dict["dtc_acc"] = grid.best_params_
# Inserting both accuracy scores for comparison
acc_dict["DT"] = [dt_acc1, dt_acc2]

# RandomForrest Classifier
start_time = time.time()
rf1 = RandomForestClassifier()
rf1.fit(X_train, y_train)
# Accuracy for pre GridSearchCV classifier
rf_acc1 = accuracy_score(y_test, rf1.predict(X_test))
# Parameters for GridSearchCV
param_grid = {'n_estimators': [70, 80, 100, 130, 150],
			  'criterion': ['gini', 'entropy'],
			  'max_depth': [4, 5, 7, 9, 10, 11, None]}

grid = GridSearchCV(rf1, param_grid, refit=True, verbose=3, n_jobs=-1)
grid.fit(X_train, y_train)
grid_predictions = grid.predict(X_test)
rf2 = RandomForestClassifier(criterion=grid.best_estimator_.criterion, n_estimators=grid.best_estimator_.n_estimators,
							 max_depth=grid.best_estimator_.max_depth)
rf2.fit(X_train, y_train)
# Accuracy for post GridSearchCV classifier
rf_acc2 = accuracy_score(y_test, rf2.predict(X_test))
grid_dict["rf_acc"] = grid.best_params_
# Inserting both accuaracy scores for comparison
acc_dict["RF"] = [rf_acc1, rf_acc2]

# AdaBoost Classifier
start_time = time.time()
ada1 = AdaBoostClassifier(base_estimator=rf2)
ada1.fit(X_train, y_train)
# Accuracy for pre GridSearchCV classifier
ada_acc1 = accuracy_score(y_test, ada1.predict(X_test))
# Parameters for GridSearchCV
grid_param = {'n_estimators': [40, 50, 60, 65, 70, 80, 100],
			  'learning_rate': [0.01, 0.1, 0.05, 0.5, 1, 10],
			  'algorithm': ['SAMME', 'SAMME.R']
			  }
grid = GridSearchCV(ada1, grid_param, refit=True, verbose=3, n_jobs=-1)
grid.fit(X_train, y_train)
ada2 = AdaBoostClassifier(base_estimator=rf2, n_estimators=grid.best_estimator_.n_estimators,
						  learning_rate=grid.best_estimator_.learning_rate, algorithm=grid.best_estimator_.algorithm)
ada2.fit(X_train, y_train)
# Accuracy for post GridSearchCV classifier
ada_acc2 = accuracy_score(y_test, ada2.predict(X_test))
grid_dict["ada_acc"] = grid.best_params_
# Inserting both accuaracy scores for comparison
acc_dict["ADA"] = [ada_acc1, ada_acc2]


print("-----------BEST HYPER-PARAMETERS-----------")
print(grid_dict)

print("-----------BEFORE AND AFTER ACCURACY-----------")
print(acc_dict)

# Creating two dataframes pre and post GridSearchCV hyper-parameter search
models_pre = pd.DataFrame({
	'Model': ['KNN', 'Decision Tree', 'Random Forest', 'Ada Boost'],
	'Score': [knn_acc1, dt_acc1, rf_acc1, ada_acc1]
})

models_post = pd.DataFrame({
	'Model': ['KNN', 'Decision Tree', 'Random Forest', 'Ada Boost'],
	'Score': [knn_acc2, dt_acc2, rf_acc2, ada_acc2]
})

# Accuracy without hyper-parameters
plt.figure(figsize=(20, 8))
sns.lineplot(x='Model', y='Score', data=models_pre)

# Accuracy using best hyper-parameters from GridSearchCV
plt.figure(figsize=(20, 8))
sns.lineplot(x='Model', y='Score', data=models_post)
plt.show()