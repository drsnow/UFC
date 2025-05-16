#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from pandas import set_option
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, classification_report
import joblib as jb
from joblib import dump
import sklearn.impute as ski
import warnings
warnings.filterwarnings("ignore")


archivo = 'data.csv'
ufc = pd.read_csv(archivo)


# ***PRE-PROCESAMIENTO***

# In[2]:


ufc_n = ufc.select_dtypes(include=['float'])
print(ufc_n.shape)
set_option('precision',2)
ufc_n = ufc_n.copy()
ufc_n.head()


# In[3]:


corr_matrix = ufc_n.corr(method='pearson').abs()
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
cax  = ax.matshow(corr_matrix,vmin=-1,vmax=1)
fig.colorbar(cax)
plt.show()


# In[4]:


ganador = list(ufc['Winner'])
ufc_n.insert(134,'Ganador',ganador,True)
ufc_n.head()


# In[5]:


col_num = ['R_Weight_lbs','R_Height_cms','B_Height_cms','R_age','B_age','R_Reach_cms','B_Reach_cms']

imp = ski.SimpleImputer(missing_values=np.nan , strategy='median')

for col in col_num:
    imputer_mediana = imp.fit_transform(ufc_n[col].values.reshape(-1,1))
    ufc_n[col] = imputer_mediana

ufc_n = ufc_n[ufc_n['Ganador'] != 'Draw']    
ufc_n = ufc_n.dropna()
ufc_n = ufc_n.copy()
print(ufc_n.isna().sum().sum())
print(ufc_n.shape)


# In[6]:


le = LabelEncoder()
ufc_n['Ganador'] = le.fit_transform(ufc_n.Ganador)
ufc_n.head()


# ***EVALUACIÓN ALGORITMOS***

# In[7]:


array = ufc_n.values
x = array[:,:134]
Y = array[:,134]
Y=Y.astype('int')
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledx = scaler.fit_transform(x)
X= rescaledx
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# In[8]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[9]:


modelos = []
modelos.append(('LR', LogisticRegression(solver='lbfgs', max_iter=500)))
modelos.append(('LDA', LinearDiscriminantAnalysis()))
modelos.append(('KNN', KNeighborsClassifier()))
modelos.append(('CART', DecisionTreeClassifier()))
modelos.append(('NB', GaussianNB()))
modelos.append(('SVM', SVC()))


# In[10]:


resultados = []
nombres = []
scoring = 'accuracy'
for nombre, modelo in modelos:
	kfold = KFold(n_splits=10, random_state=None)
	cv_results = cross_val_score(modelo, X_train, Y_train, cv=kfold, scoring=scoring)
	resultados.append(cv_results)
	nombres.append(nombre)
	msg = "%s: %f (%f)" % (nombre, cv_results.mean(), cv_results.std())
	print(msg)


# In[11]:


fig = plt.figure()
fig.suptitle('Comparación de los algoritmos')
ax = fig.add_subplot(111)
plt.boxplot(resultados)
ax.set_xticklabels(nombres)
plt.show()


# ***RECORDEMOS QUE HEMOS CODIFICADO EL COLOR DEL LUCHADOR COMO:***
# 
# 1-->ROJO
# 
# 
# 0-->AZUL

# ***Escogemos los 3 algoritmos con mejor precisión para medir su curva ROC-AUC***

# ***LogisiticRegression***

# In[12]:


lr = LogisticRegression(solver='lbfgs', max_iter=500)
lr.fit(X_train, Y_train)
lr_pred = lr.predict(X_test)
cm = confusion_matrix(Y_test, lr_pred) 
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax, fmt = "d" ,cmap='Oranges')
ax.set_title("Matriz de confusión", color='orange')
ax.xaxis.set_ticklabels(['Azul', 'Rojo'])
ax.yaxis.set_ticklabels(['Azul', 'Rojo'])


# In[13]:


lr.fit(X_test,Y_test)
probs = lr.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.title('CURVA AUC-ROC LR')
plt.plot(fpr, tpr, 'orange',marker='.', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'b--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Verdaderos positivos')
plt.xlabel('Falsos positivos')
plt.show()


# In[14]:


print(classification_report(Y_test,lr_pred))


# ***LinearDiscriminantAnalysis***

# In[15]:


lda = LinearDiscriminantAnalysis()
lda.fit(X_train,Y_train)
lda_pred = lda.predict(X_test)
cm = confusion_matrix(Y_test, lda_pred) 
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax, fmt = "d" ,cmap='Purples')
ax.set_title("Matriz de confusión", color='purple')
ax.xaxis.set_ticklabels(['Azul', 'Rojo'])
ax.yaxis.set_ticklabels(['Azul', 'Rojo'])


# In[16]:


print(classification_report(Y_test,lda_pred))


# In[17]:


lda.fit(X_test,Y_test)
probs = lda.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.title('CURVA AUC-ROC LDA')
plt.plot(fpr, tpr, 'purple',marker='.', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'b--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Verdaderos positivos')
plt.xlabel('Falsos positivos')
plt.show()


# ***SupportVectorMachines***

# In[18]:


svc = SVC(gamma='scale', probability=True)
svc.fit(X_train,Y_train)
svc_pred = svc.predict(X_test)
cm = confusion_matrix(Y_test, svc_pred) 
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax, fmt = "d" ,cmap='Greens')
ax.set_title("Matriz de confusión", color='Green')
ax.xaxis.set_ticklabels(['Azul', 'Rojo'])
ax.yaxis.set_ticklabels(['Azul', 'Rojo'])


# In[19]:


print(classification_report(Y_test,svc_pred))


# In[20]:


svc.fit(X_test,Y_test)
probs = svc.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.title('CURVA AUC-ROC SVC')
plt.plot(fpr, tpr, 'green',marker='.', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'b--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Verdaderos positivos')
plt.xlabel('Falsos positivos')
plt.show()


# ***ESCOGEMOS EL MODELO QUE MEJOR PRESTACIONES NOS DA PARA MEJORARLO(TUNNING) Y GUARDAR EL MODELO***

# In[21]:


model = SVC() 
model.fit(X_train, Y_train) 
predictions = model.predict(X_test) 


# In[22]:


param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 0) 
  

grid_result = grid.fit(X_train, Y_train) 

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[23]:


print(grid_result.best_score_)
print(grid.best_params_) 
print(grid.best_estimator_) 


# In[31]:


modelo = SVC(C=1, gamma=1 , kernel='rbf')
fichero_modeloJoblib = 'modeloUFC_finalizado_joblib.sav'
dump(modelo, fichero_modeloJoblib)


# In[ ]:


modelo_cargado = load(fichero_modeloJoblib)
resultado = modelo_cargado.score(X_test, Y_test)
print("Resultado: " + str(resultado))

