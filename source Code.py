from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import tree
from sklearn.model_selection import train_test_split

excelfile = pd.ExcelFile('TakenMind-Python-Analytics-Problem-case-study-1-1.xlsx')

Exst_Emp_data = excelfile.parse('Existing employees')

Left_Emp_data = excelfile.parse('Employees who have left')

#print(Exst_Emp_data.info())
#print(Left_Emp_data.info())
print('Average/Mean of all the features of Employees')
print('\nExisting Employees')
print(Exst_Emp_data.mean())
print('\nEmployees Left')
print(Left_Emp_data.mean())

Exst_Emp_data['left'] = 0
Left_Emp_data['left'] = 1

dataset = pd.concat([Exst_Emp_data, Left_Emp_data])
#print(dataset.head())
#print(dataset.info())

print(pd.crosstab(dataset.dept, dataset.left))
fig = pd.crosstab(dataset.dept, dataset.left).plot(kind = 'bar').get_figure()

fig.savefig('departmentcrosstab.png')

#sns.set(font_scale = 2)
#plt.figure(figsize = (34, 34))
#sns.heatmap(dataset.corr(), annot = True).get_figure().savefig('Correlation matrix')

pd.crosstab(dataset.salary, dataset.left).plot(kind = 'bar', stacked = True).get_figure().savefig('salary.png')

fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
ax1.set_title('Existing Employees')
ax2.set_title('Left Employees')
sns.boxplot(Exst_Emp_data['average_montly_hours'], ax = ax1, color = 'green').get_figure().savefig('monthlyhoursboxplot')
sns.boxplot(Left_Emp_data['average_montly_hours'],ax = ax2, color = 'red').get_figure().savefig('monthlyhoursboxplot')

fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
ax1.set_title('Existing Employees')
ax2.set_title('Left Employees')
sns.boxplot(Exst_Emp_data['satisfaction_level'], ax = ax1, color = 'green').get_figure().savefig('satisfaction_levelboxplot')
sns.boxplot(Left_Emp_data['satisfaction_level'],ax = ax2, color = 'red').get_figure().savefig('satisfaction_levelboxplot')

sns.countplot(x = 'Work_accident', data = dataset).get_figure().savefig('Work_accident count plot')
fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
ax1.set_title('Existing Employees')
ax2.set_title('Left Employees')
sns.countplot(x = 'Work_accident', data = Exst_Emp_data, ax = ax1).get_figure().savefig('Work_accident count plot')
sns.countplot(x = 'Work_accident', data = Left_Emp_data, ax = ax2).get_figure().savefig('Work_accident count plot')

dataset['dept'], _ = pd.factorize(dataset['dept'])
dataset['salary'], _ = pd.factorize(dataset['salary'])

#print(dataset['dept'].unique())
#print(dataset['salary'].unique())

sns.set(font_scale = 2)
plt.figure(figsize = (34, 34))
sns.heatmap(dataset.corr(), annot = True).get_figure().savefig('Correlation matrix')

X = dataset.iloc[:, : - 1]
Y = dataset.iloc[:, -1]

features = list(X.columns)
print(features)



#print(X.shape[1])
#print([variance_inflation_factor(X.values, i) for i in np.arange(X.shape[1])])
#print(variance_inflation_factor(X.values, 0))
#vif = [variance_inflation_factor(X.values, i) for i in np.arange(X.shape[1])]
#print(vif)

#for i in np.arange(X.shape[1]):
#    print(i)

##
    #removing collinearity, this helps the model to give accurate prediction by removing biasing
    #setting variance inflation factor value threshold to 5
##
threshold = 5
for i in np.arange(0, len(features)):
    vif = [variance_inflation_factor(X[features].values, ix) for ix in np.arange(X[features].shape[1])]
    if(max(vif) > threshold):
        maxIndex = vif.index(max(vif))
        print('Dropping :', features[maxIndex], 'having variance inflation factor value : ', max(vif))
        del features[maxIndex]
    else:
        break

print('Final Features are : \n', features)

X = X[features]

for i in np.arange(1, 6) :
    print('\nFor Decision Tree Classifier with maximum depth : ', i)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    dTreeClassifier = tree.DecisionTreeClassifier(criterion='gini', random_state=0, max_depth=i)
    dTreeClassifier.fit(X_train, Y_train)

    Y_pred = dTreeClassifier.predict(X_test)
    print(list(Y_pred))
    # Accuracy = np.mean(Y_pred == Y_test) * 100
    print('Accuracy : ', int(metrics.accuracy_score(Y_test, Y_pred) * 100), '%')

    ##
        # Accuracy is not always correct perfformance of model specially in case of classification
        # confusion matrix has values True positive, True negative, False, Positive, False Negative
    ##
    print('Confusion matrix : \n', metrics.confusion_matrix(Y_test, Y_pred))

    ##
        # this give information like precision, recall, f1-score which are calculated using values present in the confusion matrix
        # These score correctly measure the performance of a classification model
    ##
    print('Classification Report : \n', metrics.classification_report(Y_test, Y_pred))
