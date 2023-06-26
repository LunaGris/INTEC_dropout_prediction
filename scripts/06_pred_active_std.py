# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:21:58 2023

@author: DMatos

Script for processing and predicting drop out risk for active students
"""

#*****************************************************************************#
#***************************Setting up environment****************************#
#*****************************************************************************#

#*LIBRARIES*#
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

"""
#*WORKING DIRECTORY*#
import os
cwd = os.getcwd() #for getting current working directory
os.chdir('/tmp') #for changing working directory 
"""

#*DATA*# 
path = "C:/Users/DMatos/OneDrive - INTEC/proyecto_final/" #cambie PC
#path = "C:/Users/deya1/OneDrive - INTEC/proyecto_final/"
dt = pd.read_csv(path + "03_data/02_prep/dt_std_active.csv")


#*****************************************************************************#
#*************************Vars prep for y prediction**************************#
#*****************************************************************************#

# VARS PREP MODEL 1 ###########################################################
""" Modelo para predecir probabilidad de abandono en el primer año, a la 
entrada del estudiante """ 

#******************************************************************************

# Selecting vars to include in dataset for training model #1
vars_list1 = ['id', 'trim_quant', 'age_in', 'year_in', 'trim_in', 
              'program_in_desc', 'admin_type', 'credit_conv', 'cred_recon', 
              'year_birth', 'month_birth', 'day_birth', 'dayweek_birth', 'sex',
              'nationality', 'birth_city', 'birth_country', 'paa_verb_fin', 
              'paa_mat_fin', 'paa_redact_fin',  'area_in_name', 'y', 'y_m1']

# Dataset for model #1
dt1 = dt[vars_list1]

# Valores negativos 
des = dt1.describe()

# Missings 
dt1.isna().sum() 
#los estudiantes con solo un trimestre sin matricularse no estan siendo considerados como abandonos
#en este punto no sabemos nada sobre si el estudiante abandona o no, predecir esto es lo que buscamos, y y y_m1 no nos aportan nada

"""
# Saving dataset 
dt1.to_csv(path+'02_prep/dt_std_active_m1.csv', index=False)
"""

# Outliers
dt1[['age_in', 'year_in', 'trim_in', 'program_in_desc', 'admin_type', 
 'credit_conv', 'cred_recon', 'year_birth', 'month_birth', 'day_birth', 
 'dayweek_birth', 'sex', 'nationality', 'birth_city', 'birth_country', 
 'paa_verb_fin', 'paa_mat_fin', 'paa_redact_fin',  'area_in_name', 'y']].boxplot()


# VARS PREP MODEL 2 ###########################################################
""" Modelo para predecir probabilidad de abandono en el segundo año incluyendo 
datos academicos del primer año"""

## Selecting vars that will be used
var_list2 = [x for x in dt.columns if 'trim01_' in x or 'trim02_' in x \
               or 'trim03_' in x or 'trim04_' in x]

## Removing some variables that will not be used
for var in ['_nivel', '_grades_status_list', '_select_date_set', 
            '_teachid_list', '_termid', '_grades_list', ]:
    var_list2 = [x for x in var_list2 if var not in x ]

## FILTER variables and students to keep in dataset
dt2 = dt[dt.trim_quant >= 4][['id', 'date_in_year', 'date_in_month', 
                              'date_in_day','y','y_m2'] + var_list2]

## Checking missings 
temp = dt2.isna().sum() 

# Dropeando missingns que no sean acad condition en primer trimestre
dt2 = dt2[~dt2.trim01_course_quant.isna()]
dt2 = dt2[~dt2.trim01_area_name.isna()]

## Filling nan acad_condition 
vars_filled = []
vars_X = []

for i in range(1, 5):
    # Variable a rellenar
    var = 'trim'+str(i).zfill(2)+'_acad_condition'
    
    # filling missings just when needed (whene nan present)
    if dt2[var].isna().sum() > 0: 
    
        # Seleccionando variables predictivas del trimestre 1
        # TODO asegurarme que no entren las variables de course_list
        vars_list = [x for x in dt2.columns if 'trim'+str(i).zfill(2) in x]
        for v in ['_acad_condition']:
            vars_list = [x for x in vars_list if v not in x ]
        
        ## Original Data
        dt_pred = dt2[vars_X + vars_list + vars_filled + [var]].copy()
        dt_pred = dt_pred.select_dtypes(exclude=['object'])
        
        ## Process informacion with EXISTANT data ONLY
        dt_pred_complete = dt_pred.drop(index = dt_pred\
                                    .loc[pd.isna(dt_pred[var]), :].index)
        X = dt_pred_complete.drop(columns=[var])
        Y = dt_pred_complete[var]
        reg = LogisticRegression(random_state=0).fit(X, Y)
        
        ## Get missing data and predict model
        dt_pred_miss = dt_pred[dt_pred[var].isna()]
        dt_pred_index = dt_pred[dt_pred[var].isna()].index
        X = dt_pred_miss.drop(columns=[var])
        resultado = reg.predict(X)
    
        ## Rellenar missing data con prediction
        dt_pred.loc[dt_pred_index, var] = resultado
        dt2[var] = dt_pred[var].astype(int)
        
        vars_filled.append(var)
        vars_X = vars_X + vars_list
    
    
# Codificar from vector as vector como categorias: trim course_list
        
dt_temp = dt[['trim01_course_list', 'trim02_course_list', 'trim03_course_list',
              'trim04_course_list']].fillna('')
dt['year01_course_list'] = dt_temp.apply(lambda x: " ".join(x), 
                                                    axis=1)
## Hacer copia
df_exploded = dt[["year01_course_list"]].dropna().copy()
## Lista
df_exploded['year01_course_list'] = df_exploded['year01_course_list']\
                                    .apply(lambda x: x.split(" "))
## Explotar listas 
df_exploded = df_exploded.explode(column=["year01_course_list"])
## Crear lista a partir de elementos unicos
course_list = list(set(df_exploded["year01_course_list"]))
## one hot encodingg
trimestres = ['trim01_course_list', 'trim02_course_list', 'trim03_course_list', 
              'trim04_course_list']
for c in course_list:
    dt2["course_" + c] = [" ".join(df_row).split().count(c) for df_row in \
                          dt2[trimestres].values]

## Eliminando variables que siguen como strings 
dt2 =  dt2.select_dtypes(exclude=['object'])

## Eliminando course_
dt2.drop(columns='course_', inplace=True)

# Reseting index 
dt2.reset_index(inplace = True)

"""
# Saving datos listos para el modelo 2
dt2.to_csv(path+'02_prep/dt_std_active_m2.csv', index=False)
"""

# Aplicar PCA en courses exploded 
# PCA para reducir vars courses # Estaremos usando PCA para cursos de frente 
pca = PCA(n_components = 125, random_state=42)
X2_courses = dt2[[x for x in dt2.columns if 'course_' in x and '_quant'\
                      not in x]]
X2_train_pca = pd.DataFrame(pca.fit_transform(X2_courses), columns=None)
X2_train_pca = X2_train_pca.add_prefix('pca_') #arreglando nombre columns
dt2_pca = dt2.drop(columns = X2_courses.columns, inplace=False)
dt2_pca = pd.concat([dt2_pca, X2_train_pca], axis=1)
explained_variance = pca.explained_variance_ratio_.sum() #para seleccionar el n_components nos basamos en que este valor estuviera por encima de 0.95


#TODO posiblemente predecir y aqui para unir hacer la cascada y ya filtrar en dataset final para entrega al final

# VARS PREP MODEL 3 ###########################################################
""" Modelo para predecir probabilidad de abandono en el 3er año incluyendo 
datos academicos del segundo año """

## Selecting vars that will be used
var_list3 = [x for x in dt.columns if 'trim05_' in x or 'trim06_' in x \
               or 'trim07_' in x or 'trim08_' in x]

## Removing some variables that will not be used
for var in ['_nivel', '_grades_status_list', '_select_date_set', 
            '_teachid_list', '_termid']:
    var_list3 = [x for x in var_list3 if var not in x ]

## FILTER variables and students to keep in dataset
dt3 = dt[dt.trim_quant >= 8][['id', 'y', 'y_m3'] + var_list3]

## Checking missings 
temp = dt3.isna().sum()

## Filling nan acad_condition 
vars_filled = []
vars_X = []

for i in range(5, 9):
    # Variable a rellenar
    var = 'trim'+str(i).zfill(2)+'_acad_condition'
    
    # filling missings just when needed (whene nan present)
    if dt3[var].isna().sum() > 0: 
    
        # Seleccionando variables predictivas del trimestre 1
        # TODO asegurarme que no entren las variables de course_list
        vars_list = [x for x in dt3.columns if 'trim'+str(i).zfill(2) in x]
        for v in ['_acad_condition']:
            vars_list = [x for x in vars_list if v not in x ]
        
        ## Original Data
        dt_pred = dt3[vars_X + vars_list + vars_filled + [var]].copy()
        dt_pred = dt_pred.select_dtypes(exclude=['object'])
        
        ## Process informacion with EXISTANT data ONLY
        dt_pred_complete = dt_pred.drop(index = dt_pred\
                                    .loc[pd.isna(dt_pred[var]), :].index)
        X = dt_pred_complete.drop(columns=[var])
        Y = dt_pred_complete[var]
        reg = LogisticRegression(random_state=0).fit(X, Y)
        
        ## Get missing data and predict model
        dt_pred_miss = dt_pred[dt_pred[var].isna()]
        dt_pred_index = dt_pred[dt_pred[var].isna()].index
        X = dt_pred_miss.drop(columns=[var])
        resultado = reg.predict(X)
    
        ## Rellenar missing data con prediction
        dt_pred.loc[dt_pred_index, var] = resultado
        dt3[var] = dt_pred[var].astype(int)
        
        vars_filled.append(var)
        vars_X = vars_X + vars_list
    
# Codificar from vector as vector como categorias: trim course_list

dt_temp = dt[['trim05_course_list', 'trim06_course_list', 
                   'trim07_course_list', 'trim08_course_list']].fillna('')

dt['year02_course_list'] = dt_temp.apply(lambda x: " ".join(x), axis=1)


# Hacer copia
df_exploded = dt[["year02_course_list"]].dropna().copy()
# Lista
df_exploded['year02_course_list'] = df_exploded['year02_course_list']\
                                    .apply(lambda x: x.split(" "))
# Explotar listas
df_exploded = df_exploded.explode(column=["year02_course_list"])
# Crear lista a partir de elementos unicos
course_list = list(set(df_exploded["year02_course_list"]))

# one hot encodingg
trimestres = ['trim05_course_list', 'trim06_course_list', 'trim07_course_list', 
              'trim08_course_list']
for c in course_list:
    dt3["course_" + c] = [" ".join(df_row).split().count(c) for df_row in \
                          dt3[trimestres].values]

# Eliminando variables que siguen como strings 
dt3 =  dt3.select_dtypes(exclude=['object'])

# Eliminando course_
dt3.drop(columns='course_', inplace=True)

# Reseting index 
dt3.reset_index(inplace = True)

"""
# Saving datos listos para el modelo 2
dt3.to_csv(path+'02_prep/dt_std_active_m3.csv', index=False)
"""

# Aplicar PCA en courses exploded 
# PCA para reducir vars courses # Estaremos usando PCA para cursos de frente 
X3_courses = dt3[[x for x in dt3.columns if 'course_' in x and '_quant'\
                      not in x]]
X3_train_pca = pd.DataFrame(pca.fit_transform(X3_courses), columns=None)
X3_train_pca = X3_train_pca.add_prefix('pca_') #arreglando nombre columns
dt3_pca = dt3.drop(columns = X3_courses.columns, inplace=False)
dt3_pca = pd.concat([dt3_pca, X3_train_pca], axis=1)
explained_variance = pca.explained_variance_ratio_.sum() #para seleccionar el n_components nos basamos en que este valor estuviera por encima de 0.95


#TODO posiblemente predecir y aqui para unir hacer la cascada y ya filtrar en dataset final para entrega al final



# VARS PREP MODEL 4 ###########################################################
""" Modelo para predecir probabilidad de abandono luego del 4to año incluyendo 
datos academicos del tercer año """

## Selecting vars that will be used
var_list4 = []
for i in range(9,13):
    l = [x for x in dt.columns if 'trim'+str(i).zfill(2) in x]
    var_list4 = var_list4 + l

## Removing some variables that will not be used
for var in ['_nivel', '_grades_status_list', '_select_date_set', 
            '_teachid_list', '_termid']:
    var_list4 = [x for x in var_list4 if var not in x ]

## FILTER variables and students to keep in dataset
dt4 = dt[dt.trim_quant >= 12][['id', 'y', 'y_m4'] + var_list4]

## Checking missings 
temp = dt4.isna().sum()


## Filling nan acad_condition 
vars_filled = []
vars_X = []

for i in range(9, 13):
    # Variable a rellenar
    var = 'trim'+str(i).zfill(2)+'_acad_condition'
    
    # filling missings just when needed (whene nan present)
    if dt4[var].isna().sum() > 0: 
    
        # Seleccionando variables predictivas del trimestre 1
        # TODO asegurarme que no entren las variables de course_list
        vars_list = [x for x in dt4.columns if 'trim'+str(i).zfill(2) in x]
        for v in ['_acad_condition']:
            vars_list = [x for x in vars_list if v not in x ]
        
        ## Original Data
        dt_pred = dt4[vars_X + vars_list + vars_filled + [var]].copy()
        dt_pred = dt_pred.select_dtypes(exclude=['object'])
        
        ## Process informacion with EXISTANT data ONLY
        dt_pred_complete = dt_pred.drop(index = dt_pred\
                                    .loc[pd.isna(dt_pred[var]), :].index)
        X = dt_pred_complete.drop(columns=[var])
        Y = dt_pred_complete[var]
        reg = LogisticRegression(random_state=0).fit(X, Y)
        
        ## Get missing data and predict model
        dt_pred_miss = dt_pred[dt_pred[var].isna()]
        dt_pred_index = dt_pred[dt_pred[var].isna()].index
        X = dt_pred_miss.drop(columns=[var])
        resultado = reg.predict(X)
    
        ## Rellenar missing data con prediction
        dt_pred.loc[dt_pred_index, var] = resultado
        dt4[var] = dt_pred[var].astype(int)
        
        vars_filled.append(var)
        vars_X = vars_X + vars_list
    
    
# Codificar from vector as vector como categorias: trim course_list
dt_temp = dt[['trim09_course_list', 'trim10_course_list', 
                   'trim11_course_list', 'trim12_course_list']].fillna('')

dt['year03_course_list'] = dt_temp.apply(lambda x: " ".join(x), 
                                                    axis=1)


# Hacer copia
df_exploded = dt[["year03_course_list"]].dropna().copy()
# Lista
df_exploded['year03_course_list'] = df_exploded['year03_course_list']\
                                    .apply(lambda x: x.split(" "))
# Explotar listas
df_exploded = df_exploded.explode(column=["year03_course_list"])
# Crear lista a partir de elementos unicos
course_list = list(set(df_exploded["year03_course_list"]))

# one hot encodingg
trimestres = ['trim09_course_list', 'trim10_course_list', 'trim11_course_list', 
              'trim12_course_list']
for c in course_list:
    dt4["course_" + c] = [" ".join(df_row).split().count(c) for df_row in \
                          dt4[trimestres].values]

# Eliminando variables que siguen como strings 
dt4 =  dt4.select_dtypes(exclude=['object'])

# Eliminando course_
dt4.drop(columns='course_', inplace=True)

# Reseting index 
dt4.reset_index(inplace = True)

"""
# Saving datos listos para el modelo 2
dt4.to_csv(path+'02_prep/dt_std_active_model4.csv', index=False)
"""

# Aplicar PCA en courses exploded 
# PCA para reducir vars courses # Estaremos usando PCA para cursos de frente 
X4_courses = dt4[[x for x in dt4.columns if 'course_' in x and '_quant'\
                      not in x]]
X4_train_pca = pd.DataFrame(pca.fit_transform(X4_courses), columns=None)
X4_train_pca = X4_train_pca.add_prefix('pca_') #arreglando nombre columns
dt4_pca = dt4.drop(columns = X4_courses.columns, inplace=False)
dt4_pca = pd.concat([dt4_pca, X4_train_pca], axis=1)
explained_variance = pca.explained_variance_ratio_.sum() #para seleccionar el n_components nos basamos en que este valor estuviera por encima de 0.95



#*****************************************************************************#
#***************************Dropout risk prediction***************************#
#*****************************************************************************#

#TODO hacer un solo dataset de todos los estudiantes para predecir Y, que incluya su id y todas las variables preocesadas en dt1, 2, 3, y 4


# MODELO 1 ********************************************************************

# loading saved model
mod1 = joblib.load(path + '05_models/decision_tree_year1.joblib')

# desertion prediction
## selecting vars for model
dt1_X = dt1[['age_in', 'year_in', 'trim_in', 'program_in_desc', 'admin_type',
               'credit_conv', 'cred_recon', 'year_birth', 'month_birth', 
               'day_birth', 'dayweek_birth', 'sex', 'nationality', 
               'birth_city', 'birth_country', 'paa_verb_fin', 'paa_mat_fin', 
               'paa_redact_fin', 'area_in_name']]
## model prediction
pred_y1 = mod1.predict(dt1_X)
## saving results to dataset for model 1
dt1['y_y1_pred'] = pd.Series(pred_y1, dtype=int)
print('## YEAR 1 PREDICTION ##')
print(dt1['y_y1_pred'].value_counts())

# predicting desertion probability 
prob_desert_y1 =  mod1.predict_proba(dt1_X)
dt1['y_y1_prob'] = [sum(x[1:]) for x in prob_desert_y1]

### funcion de costo
sns.displot(dt1['y_y1_prob'], kde=True)
plt.show()

# Appending y_pred and y_prob to dt1 
dt = dt.merge(dt1[['id', 'y_y1_pred', 'y_y1_prob']], how='left', on='id')


# MODELO 2 ********************************************************************

"""Los modelos iniciales para año 2 en adelante no incluyeron los nombres de 
las variables durante el entrenamiento porque las variables arrojadas por el 
PCA tenían números como nombres. Para corregir esto se reentrenaron los modelos
con los mismos hiperparámetros corrigiendo los nombres de las variables 
arrojadas por el PCA antes de entrenar el modelo. Esto se hizo en un script a 
parte nombrado 'including_varnames_mod2_3_4'. También fueron corregidos los 
nombres de las vars arrojadas por PCA en este script. """

# loading saved model
mod2 = joblib.load(path + '05_models/decision_tree_year2_withvarnames.joblib')

# appending prob year 1 to dt2_pca
dt2_pca = dt2_pca.merge(dt[['id', 'y_y1_prob']], how='left', on='id')
# corrigiendo nombre prob year 1 para que coincida con feature names modelo
dt2_pca.rename(columns={"y_y1_prob": "prob_m1_year1"}, inplace=True)

# predicting desertion
dt2_X = dt2_pca.drop(columns=['y', 'id', 'y_m2', 'index'])
#dt2_X.columns = dt2_X.columns.astype(str)
pred_y2 = mod2.predict(dt2_X)
dt2['y_y2_pred'] = pd.Series(pred_y2, dtype=int)
print('## YEAR 2 PREDICTION ##')
print(dt2['y_y2_pred'].value_counts())

# predicting desertion probability 
prob_desert_y2 =  mod2.predict_proba(dt2_X)
dt2['y_y2_prob'] = [sum(x[1:]) for x in prob_desert_y2]

### funcion de costo
sns.displot(dt2['y_y2_prob'], kde=True)
plt.show()

# Appending y_pred and y_prob to dt1 
dt = dt.merge(dt2[['id', 'y_y2_pred', 'y_y2_prob']], how='left', on='id')


# MODELO 3 ####################################################################

# loading saved model
mod3 = joblib.load(path + '05_models/decision_tree_year3_withvarnames.joblib')

# appending prob year 1 to dt3_pca #TODO ojo que no esta entrando y_y2_prob a dt3_X
dt3_pca = dt3_pca.merge(dt[['id', 'y_y2_prob']], how='left', on='id')
# corrigiendo nombre prob year 1 para que coincida con feature names modelo
dt3_pca.rename(columns={"y_y2_prob": "prob_m2_m2_smote"}, inplace=True)
# dropping cases with missing prob year 2 OJO 
dt3_pca.dropna(axis=0, subset = ['prob_m2_m2_smote'], inplace=True)

# predicting desertion
dt3_X = dt3_pca.drop(columns=['y', 'id', 'y_m3'])
#dt3_X.columns = dt3_X.columns.astype(str)
pred_y3 = mod3.predict(dt3_X)
dt3_pca['y_y3_pred'] = pd.Series(pred_y3, dtype=int)
print('## YEAR 3 PREDICTION ##')
print(dt3_pca['y_y3_pred'].value_counts())

# predicting desertion probability 
prob_desert_y3 =  mod3.predict_proba(dt3_X)
dt3_pca['y_y3_prob'] = [sum(x[1:]) for x in prob_desert_y3]

### funcion de costo
sns.displot(dt3_pca['y_y3_prob'], kde=True)
plt.show()

# Appending y_pred and y_prob to dt1 
dt = dt.merge(dt3_pca[['id', 'y_y3_pred', 'y_y3_prob']], how='left', on='id')

"""
NOTA RESULTADO: 
- Arroja una probabilidad prácticamente en 0 para todos los 
casos. 
- Estamos teniendo 3 casos con missings con la probabilidad de abandono para el
segundo año, esto es inesperado. Por cuestiones de tiempo solo se procedió a 
dropear los missing y continuar
"""

# MODELO 4 ####################################################################

""" 
Esta parte queda pendiente de ser implementada. Este es el modelo que menos 
información útil aporta, en parte porque ya en este punto la población objetivo
(estudiantes que abandonan) es la minoría. 
"""

##para mod 4 hay que predecir prob mod 3, aplicar PCA para reducir vars courses

#TODO predecir y aplicando modelos de decision tree segun trim_quant 

#TODO preparar dataset para entrega que contenga vars_list_1, prob de desercion y aoutcome predicho para pasar a INTEC 


#*****************************************************************************#
#*************************** Dataset para entregar ***************************#
#*****************************************************************************#

dt_fin = dt[['id', 'age_in', 'age_out', 'date_in', 'grad_date', 'honor', 
             'nivel', 'program', 'program_desc', 'active', 'year_in', 
             'trim_in', 'program_in', 'program_in_desc', 'admin_type', 
             'credit_conv', 'cred_recon', 'grad', 'program_change', 'dob', 
             'sex', 'nationality', 'birth_city', 'birth_country', 'address',
             'city', 'y_y1_pred', 'y_y1_prob', 'y_y2_pred', 'y_y2_prob', 
             'y_y3_pred', 'y_y3_prob']]
