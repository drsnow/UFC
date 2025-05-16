#!/usr/bin/env python
# coding: utf-8

# ***Fase inicial general:***

# In[1]:


import pandas as pd 
from pandas import set_option
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.impute as ski

archivo = 'data.csv'
archivo_2 = 'fighters_data.csv'

ufc = pd.read_csv(archivo)
ufc_f = pd.read_csv(archivo_2)


# In[2]:


ufc.head()


# In[3]:


ufc.info()


# In[4]:


print(ufc.shape)


# In[5]:


set_option('precision',2)
ufc.describe()


# In[6]:


columnas = ufc.columns
print(list(columnas))


# In[7]:


alt = pd.read_csv('raw_total_fight_data.csv',sep=';')

ganador = list(alt['Winner'])
ufc.insert(2,'Ganador',ganador,True)

method = list(alt['win_by'])
ufc.insert(3,'win_by',method,True)

Formato = list(alt['Format'])
ufc.insert(4,'Formato',Formato,True) 

ult_asalto = list(alt['last_round'])
ufc.insert(5,'ult_asalto',ult_asalto,True)


# In[8]:


ufc['Formato'] = ufc['Formato'].apply(lambda x : x.split('(')[0])
ufc['Formato'] = ufc['Formato'].apply(lambda x : x.split('+')[0])
ufc['location'] = ufc['location'].apply(lambda x : x.split(',')[-1])
ufc['date'] = ufc['date'].apply(lambda x: x.split('-')[0])


# In[9]:


ufc.head()


# In[10]:


ufc.isna().sum().sum()


# In[11]:


null = []
for index, col in enumerate(ufc):
    null.append((index,ufc[col].isna().sum()))
null.sort(key = lambda x : x[1])
for i in range(len(ufc.columns)):
    print("En la columna",ufc.columns[null[i][0]],"hay;",null[i][1],"valores IsNull")


# In[12]:


col_num = ['R_Weight_lbs','R_Height_cms','B_Height_cms','R_age','B_age','R_Reach_cms','B_Reach_cms']

imp = ski.SimpleImputer(missing_values=np.nan , strategy='median')

for col in col_num:
    imputer_mediana = imp.fit_transform(ufc[col].values.reshape(-1,1))
    ufc[col] = imputer_mediana



imp_ = ski.SimpleImputer(missing_values=np.nan,strategy='most_frequent')

imp_r = imp_.fit_transform(ufc['R_Stance'].values.reshape(-1,1))
imp_b = imp_.fit_transform(ufc['B_Stance'].values.reshape(-1,1))
ufc['R_Stance'] = imp_r
ufc['B_Stance'] = imp_b



ufc['Referee'].fillna(value='NO DATA', inplace=True)
ufc['Ganador'].fillna(value='NO WINNER', inplace=True)

print(ufc.isna().sum().sum())


# In[13]:


ufc_f.head()


# In[14]:


ufc_f.shape


# In[15]:


col = ufc_f.columns

print(list(col))
print(list(ufc_f.dtypes))


# In[16]:


ufc_f = ufc_f.drop(['url','fid','locality','nick'], axis=1)
ufc_f


# In[17]:


ufc_f.isna().sum().sum()


# In[18]:


null = []
for index, col in enumerate(ufc_f):
    null.append((index,ufc_f[col].isna().sum()))
null.sort(key = lambda x : x[1])
for i in range(len(ufc_f.columns)):
    print("En la columna",ufc_f.columns[null[i][0]],"hay;",null[i][1],"valores IsNull")


# In[19]:


col_num = ['height','weight']

imp = ski.SimpleImputer(missing_values=np.nan , strategy='median')

for col in col_num:
    imputer_mediana = imp.fit_transform(ufc_f[col].values.reshape(-1,1))
    ufc_f[col] = imputer_mediana
    
ufc_f['name'].fillna(value='NO DATA', inplace=True)
ufc_f['country'].fillna(value='NO DATA', inplace=True)
ufc_f['class'].fillna(value='NO DATA', inplace=True)
ufc_f['association'].fillna(value='NO DATA', inplace=True)
ufc_f['birth_date'].fillna(value='NO DATA' , inplace=True)

print(ufc_f.isna().sum().sum())


# In[20]:


ufc_f['birth_date'] = ufc_f['birth_date'].apply(lambda x : x.split('/')[-1])

X = []
for f in ufc_f['height']:
    X.append(f*0.0254)
        
Y = []
for p in ufc_f['weight']:
    Y.append(p*0.45359237)


ufc_f['height_'] = pd.Series(X, dtype='float')
ufc_f['weight_'] = pd.Series(Y, dtype='float')
ufc_f = ufc_f.drop(['height','weight'],axis =1)

pd.set_option('precision',2)

ufc_f.head(5)


# In[21]:


valores = ufc['date'].value_counts().sort_values().sort_index()
eventos = valores.index

plt.figure(figsize=(15,8))
fig = sns.barplot(x=eventos,y=valores, palette='gist_rainbow')
plt.plot(eventos,valores, color='Yellow', linewidth=2.5)
for p in fig.patches:
    fig.annotate(format(p.get_height(),'.2f'), (p.get_x() + p.get_width() / 2.,
                                                p.get_height()), ha = 'center', xytext = (0, 10),
                                                 textcoords = 'offset points')
    
plt.xlabel('Años',color='blue')
plt.ylabel('Nº eventos',color='blue')
plt.title('Nº de eventos por año',color='blue')


# In[22]:


valores = ufc['location'].value_counts()
eventos = valores.index

plt.figure(figsize=(15,7))
plt.xticks(rotation=90)
plt.title('Número de eventos por país',color='blue')
fig = sns.barplot(x=eventos,y=valores, palette='husl')


# In[23]:


plt.figure(figsize=(15,10))
sns.countplot(y=ufc['weight_class'], palette='brg')
plt.xlabel('Nº combates')
plt.ylabel('Categorías')
plt.title('Nº de combates por categoría',color='blue')
plt.show()


# In[24]:


valores = ufc['win_by'].value_counts()
metodo = valores.index

plt.figure(figsize=(20,8))
plt.title('Combates por método de finalización', color='blue')
plt.xlabel('Método')
fig = sns.barplot(x=metodo,y=valores, palette='pastel')


# In[25]:


plt.figure(figsize=(5,5))
colores=["#FF9B85","#AAF683"]
ufc['title_bout'].value_counts().plot.pie(explode=[0.05,0.05],autopct='%1.1f%%',shadow=True, colors=colores)
plt.title('Combates por título', color='blue')
plt.show()


# In[26]:


ufc['Formato'].value_counts()


# In[27]:


plt.figure(figsize=(20,10))
colores = colores = ["#FFD97D","#EE6055","#60D394","#AAF683","#FF9B85"]
ufc['Formato'].value_counts().plot.pie(explode=[0.05,0.05,0.05,0.05,0.05],autopct=None,shadow=False, 
                                            colors=colores)
plt.title('Duración de los combates por nº de asaltos', color='blue')
plt.ylabel('nº asaltos')
plt.show()


# In[28]:


for_data = ufc.groupby(['date','Formato']).size().reset_index().pivot(
    columns='Formato',index='date', values=0)
for_data.plot(kind='barh',alpha=1, width=0.5, stacked=True,figsize=(8,5))
plt.xlabel('Nº combates')
plt.ylabel('Años')
plt.title('Evolución de formatos en combates de la UFC',color='blue')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.show()


# In[29]:


plt.figure(figsize=(5,5))
colores=["#F44336","#29B6F6","#FFEB3B"]
ufc['Winner'].value_counts().plot.pie(explode=[0.05,0.05,0.05],
                                      autopct='%1.1f%%',shadow=True, colors=colores)
plt.title('Ganador del combate por color de esquina', color='blue')
plt.ylabel('Ganador')
plt.show()


# In[30]:


cat_met = ufc.groupby(['weight_class','win_by']).size().reset_index().pivot(
    columns='win_by',index='weight_class', values=0)
cat_met.plot(kind='barh',alpha=1, width=0.5, stacked=True,figsize=(8,5))
plt.xlabel('Nº de combates finalizados')
plt.ylabel('Categorías')
plt.title('Métodos de finalización por categoría',color='blue')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.show()


# In[31]:


cat_fecha = ufc.groupby(['date','win_by']).size().reset_index().pivot(columns='win_by',index='date', values=0)
cat_fecha.plot(kind='barh',alpha=1, width=0.5, stacked=True,figsize=(8,8))
plt.title('Evolución de los métodos de finalización con el paso de los años', color='blue')
plt.xlabel('Nº de combates finalizados')
plt.ylabel('Años')
plt.show()


# In[32]:


plt.figure(figsize=(5,5))
sns.stripplot(x='no_of_rounds',y='weight_class',data=ufc)
plt.title('Distribución de la duración en asaltos por categoría', color='blue')
plt.xlabel('Asaltos')
plt.ylabel('Categorías')
plt.show()


# In[33]:


valores = ufc['R_Stance'].value_counts().sort_values(ascending=False)
posturas_r = valores.index

plt.figure(figsize=(10,6))
plt.title('Posturas luchadores esquina roja', color='blue')

sns.barplot(x=posturas_r,y=valores, palette='Reds')


# In[34]:


valores = ufc['B_Stance'].value_counts().sort_values(ascending=False)
posturas_r = valores.index
plt.figure(figsize=(10,6))
plt.title('Posturas luchadores esquina azul', color='blue')
sns.barplot(x=posturas_r,y=valores, palette='Blues')


# In[35]:


plt.figure(figsize=(6,6))
colores=["#A569BD","#38E1E1","#FFEB3B"]
ufc['R_Stance'].value_counts().plot.pie(explode=[0.05,0.05,0.05,0.05,0.05],
                                      autopct='%.0f%%',shadow=True, 
                                        colors=colores, pctdistance=0.5)
plt.title('Posturas luchadores esquina roja', color='blue')


# ANALISIS LUCHADORES

# In[36]:


ufc_f.head()


# In[37]:


paises = ufc_f['country'].value_counts().sort_values(ascending=False)
num = paises.index
plt.figure(figsize=(20,6))
plt.xticks(rotation=90)
plt.yticks(rotation=90)
sns.barplot(x=num, y=paises, palette='cool')


# In[38]:


paises_top = ufc_f['country'].value_counts().sort_values(False).sort_index()
paises_top = pd.DataFrame(data=paises_top)
paises_top = paises_top.loc[paises_top.country>5]
paises_top['paises_top'] = paises_top.index
paises_top['n_luchadores'] = paises_top['country']
del paises_top['country']
paises_top.reset_index(drop=True,inplace=True)
paises_top.drop([9,15],inplace=True, axis=0)
paises_top


# In[39]:


paises_t = paises_top['paises_top']
n_luchadores = paises_top['n_luchadores']
plt.figure(figsize=(10,10))
fig= sns.barplot(x=n_luchadores, y=paises_t,palette='Paired')


# In[40]:


ufc_f['association'].unique()


# In[41]:


escuelas = ufc_f['association'].value_counts()
escuelas = pd.DataFrame(data=escuelas)
escuelas = escuelas.loc[escuelas.association>10]
escuelas['escuelas'] = escuelas.index
escuelas['luchadores']=escuelas['association']
del escuelas['association']
escuelas.reset_index(drop=True, inplace=True)
escuelas.drop([0],inplace=True, axis=0)
escuelas


# In[42]:


ass = ufc_f.groupby(['association','country']).size().reset_index().pivot(columns='association',index='country', values=0)
ass.plot(kind='barh',alpha=1, width=0.5, stacked=True,figsize=(20,20),legend=False)
sns.set_context('paper',font_scale=1.5)


# In[43]:


escuelas_t = escuelas['escuelas']
n_luchadores = escuelas['luchadores']
plt.figure(figsize=(10,10))
fig= sns.barplot(x=n_luchadores, y=escuelas_t)
plt.title('Principales equipos historia UFC', color='blue')


# In[44]:


ufc_f['class'].value_counts()


# In[45]:


plt.figure(figsize=(20,12))
plt.title('Distribución luchadores por categorías', color='blue')
colores=['#800080','#FF00FF','#56AEC4','#0000FF','#008080','#00FFFF',
         '#008000','#00FF00','#808000','#FFFF00','#800000','#FF0000']
ufc_f['class'].value_counts().plot.pie(explode=[0.05,0.05,0.05,
                                                0.05,0.05,0.05,
                                                0.05,0.05,0.05,
                                                0.05,0.05,0.05],
                                       autopct='%1.1f%%',shadow=True,colors=colores)


# In[46]:


col = list(ufc.columns)
filtro = [col for col in ufc if col.find('TOTAL')>=0]
totales = ufc[filtro]
totales = totales.dropna()
print(totales.isna().sum().sum())


# In[47]:


plt.figure(figsize=(50,50))
corr_matrix = totales.corr(method = 'pearson').abs()
sns.set(font_scale=5)
sns.heatmap(corr_matrix, annot=True, cmap="YlGn")


# In[48]:


sns.set(font_scale=1.5)
sns.set_style('white')
fig,ax=plt.subplots(1,2,figsize=(10,5))
sns.histplot(ufc['R_age'], ax=ax[0], color='red')
sns.histplot(ufc['B_age'], ax=ax[1], color='blue')
ax[0].set_title('R_age', color='red')
ax[1].set_title('B_age', color='blue')
plt.show()


# In[57]:


sns.set_style('white')
sns.jointplot(x='R_Reach_cms', y='R_Height_cms', data=ufc, kind='hex', color='green')
sns.despine(left=True, bottom=True)


# In[58]:


sns.set_style('white')
sns.jointplot(x='R_Reach_cms', y='R_Weight_lbs',
              data=ufc, kind='hex', color='orange')
sns.despine(left=True, bottom=True)


# In[59]:


sns.set_style('ticks')
sns.relplot(x='R_Height_cms', y='R_avg_TOTAL_STR_att',
            data=ufc, kind='line', color='gold')


# In[60]:


sns.set_style('ticks')
sns.relplot(x='R_Height_cms', y='R_avg_TOTAL_STR_landed',
            data=ufc, kind='line', color='purple')


# In[61]:


sns.set_style('white')
sns.jointplot(x='R_age', y='R_avg_TOTAL_STR_att',
            data=ufc, kind='scatter', color='pink')


# In[62]:


sns.set_style('white')
sns.jointplot(x='R_age', y='R_avg_TOTAL_STR_landed',
            data=ufc, kind='scatter', color='yellow')


# In[ ]:




