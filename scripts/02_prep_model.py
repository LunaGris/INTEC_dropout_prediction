# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 16:18:04 2023
After first preprocessing and visualization. This script organizes the data for
the modelling and converts all string variables to numeric. 

@author: DMatos
"""

#*****************************************************************************#
#***************************Setting up environment****************************#
#*****************************************************************************#

#*LIBRARIES*#
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from tensorflow.keras import Sequential
from tensorflow.keras.layers import TextVectorization, Embedding

"""
#*WORKING DIRECTORY*#
import os
cwd = os.getcwd() #for getting current working directory
os.chdir('/tmp') #for changing working directory 
"""

#*DATA*# 
path = "C:/Users/DMatos/OneDrive - INTEC/proyecto_final/03_data/"
dt_main = pd.read_csv(path + "02_prep/dt_main_pii.csv")


#*****************************************************************************#
#********************************* Cleaning **********************************#
#*****************************************************************************#

# Creando year en que abandona
conditions = [(dt_main.y.isin(['involuntary desertion','voluntary desertion']))
              & (dt_main.trim_quant < 4), 
              (dt_main.y.isin(['involuntary desertion','voluntary desertion'])) 
              & (dt_main.trim_quant >= 4) & (dt_main.trim_quant < 8),
              (dt_main.y.isin(['involuntary desertion','voluntary desertion'])) 
              & (dt_main.trim_quant >= 8) & (dt_main.trim_quant < 12), 
              (dt_main.y.isin(['involuntary desertion','voluntary desertion'])) 
              & (dt_main.trim_quant >= 12) & (dt_main.trim_quant < 16),
              (dt_main.y.isin(['involuntary desertion','voluntary desertion'])) 
              & (dt_main.trim_quant >= 16)]
choices = ['1st year', '2nd year', '3rd year', '4th year', '5th year +']
dt_main['year_desert'] = np.select(conditions, choices, default=np.nan)

""" Ire separando el dataset segun al modelo al que van orientados  para 
manejar mas especificamente los missings y que sea mas facil de manejar en 
general, porque tendre menos variables por dataset """

#Estas vars se quedan fuera
['age_out','grad_date', 'honor', 'nivel', 'active','program', 'program_desc',
 'grad','program_change', 'trim_quant', 'last_acad_condition', 
 'last_year_enroll', 'last_trim_enroll', 'last_termid', 'trim_unenroll_quant', 
 'prev_trims_out_sum', 'program_change_quant','program_in.1', 'area_name',
 'last_program', 'last_program_name', 'last_area', 'last_area_name']
#OJO no se si sea necesario crear una var trim_unenroll_quant por trimestre
#TODO quizas sea interesante agregar edad por trimestre, aunque ya estamos 
#agregando cantidad de trimestres out que es equivalente me parece 

# Cleaning by admin_type
## Dropeando MOVILIDAD ESTUDIANTIL, EGRESADO INTEC y PROFESIONAL
dt_main = dt_main[dt_main.admin_type != "MOVILIDAD ESTUDIANTIL"]
dt_main = dt_main[dt_main.admin_type != "EGRESADO INTEC"]
dt_main = dt_main[(dt_main.admin_type != "PROFESIONAL")]

# Cleaning by program_in, eliminando 3 maestrias coladas
dt_main = dt_main[~dt_main.program_in.isin(['MPM', 'NMC', 'MCS', 'ME'])]
dt_main = dt_main[~dt_main.last_program.isin(['ME'])]
dt_main = dt_main[~dt_main.trim02_program.isin(['ME'])]
dt_main = dt_main[~dt_main.trim03_program.isin(['ME'])]
dt_main = dt_main[~dt_main.trim04_program.isin(['ME'])]
dt_main = dt_main[dt_main.trim02_program_name \
                  != "PROGRAMA MOVILIDAD ESTUDIANTIL HUMANIDADES"]

# Dropping nan sex
dt_main.drop(index = dt_main.loc[pd.isna(dt_main["sex"]), :].index, 
             inplace = True)


# DATETIME DATA ***************************************************************

# DOB 
## La edad esta rara
""" Por ahora eliminare age_in (quizas puedo hacer una para el modelo 2, y en 
este punto solo usare los datos de DOB """

# Adaptando DOB 
dt_main['dob'] = pd.to_datetime(dt_main['dob'], infer_datetime_format=True)
dt_main['year_birth'] = dt_main['dob'].dt.year
dt_main['month_birth'] = dt_main['dob'].dt.month
dt_main['day_birth'] = dt_main['dob'].dt.day
dt_main['dayweek_birth'] = dt_main['dob'].dt.dayofweek

# Estimando age_in con dob 
dt_main['age_in_estim'] = dt_main['year_in'] - dt_main['year_birth']

## Eliminando valores negativos
dt_main['age_in_estim'] = np.where(dt_main['age_in'] <= 10, np.nan, 
                                   dt_main['age_in'])
dt_main['age_in_estim'] = np.where(dt_main['age_in'] >= 70, np.nan, 
                                   dt_main['age_in'])

## Eliminando birth info for age_in_estim eliminated
## Las edades estan mal porque la fecha de nacimiento no hacen sentido
for var in ['year_birth', 'month_birth', 'day_birth', 'dayweek_birth']:
    dt_main[var] = np.where(dt_main['age_in_estim'].isna(), np.nan, 
                          dt_main[var])
    
## Dropping nan year_birth 
dt_main.drop(index = dt_main.loc[pd.isna(dt_main["year_birth"]), :].index, 
             inplace = True)
    
# DATE_IN 
dt_main['date_in'] = pd.to_datetime(dt_main['date_in'], 
                                    infer_datetime_format=True)

# replacing weird years in date_in with year_in
for i in dt_main[(dt_main.date_in.dt.year < 2007) | \
                 (dt_main.date_in.dt.year > 2022)].index:
    dt_main.at[i, 'date_in'] = dt_main.at[i, 'date_in']\
                               .replace(year = dt_main.at[i, 'year_in'])

# replacing nan in date_in according to year and trim_in 
## calculating mode by year_in + trim_in
mode_year_trim_in = dt_main[['date_in', 'year_in', 'trim_in']]\
                    .groupby(['year_in','trim_in'])['date_in']\
                    .agg(pd.Series.mode)
                    
## replacing nan according to mode for year_in + trim_in                     
for i in dt_main[dt_main.date_in.isna()].index: 
    year = dt_main.at[i, 'year_in']
    trim = dt_main.at[i, 'trim_in']
    dt_main.at[i, 'date_in'] = mode_year_trim_in[year, trim]
    
# Adaptando date_in 
dt_main['date_in_year'] = dt_main['date_in'].dt.year
dt_main['date_in_month'] = dt_main['date_in'].dt.month
dt_main['date_in_day'] = dt_main['date_in'].dt.day

# GEO DATA ********************************************************************
""" Por ahora omitiremos datos de direcciones porque lleva un procesado mas 
complejo """

# Converting to lower
geo_vars = ['nationality', 'birth_city', 'birth_country']
for var in geo_vars:
    dt_main[var] = dt_main[var].str.lower()

# Separando birth_city de country #eliminar cosas en parentesis o luego de /
dt_main[['birth_city', 'birth_city2']] =  dt_main['birth_city']\
                                          .str.split(',', n=1, expand=True)

dt_main[['birth_city', 'birth_city3']] =  dt_main['birth_city']\
                                          .str.split('/', n=1, expand=True)
                                          
dt_main[['birth_city', 'birth_city4']] =  dt_main['birth_city']\
                                          .str.split('(', n=1, expand=True)
                                          
# Stripping birth_city #TODO strip all string vars 
dt_main['birth_city'] = dt_main['birth_city'].str.strip()

# Eliminar tildes
dt_main['birth_city'] = dt_main['birth_city'].str.normalize('NFKD')\
                        .str.encode('ascii', errors='ignore')\
                        .str.decode('utf-8')

# eliminar signos de puntuacion 
dt_main['birth_city'] = dt_main['birth_city'].str.replace('[^\w\s]','',
                                                          regex=True)

# sto dgo por santo domingo 
dt_main['birth_city'].replace(to_replace = ['sto dgo', 'repdom santo domingo', 
                                            'sabto domingo'],
                              value = 'santo domingo', inplace = True)
# d.n. y dn por distrito nacional, dn santo domingo, distrito nacional sd 
dt_main['birth_city'].replace(to_replace = ['dn'], value = 'distrito nacional',
                              inplace = True)
#broklyn, broocklyn, brooklyn new york, brooklyn ny por brooklyn 
#bronx new york, bronxnew york por bronx 
#buenos aires argentina, buenos aures por argentina 
#cabo y cabo haiti por cabo haitiano
# Sustituir bogota d.c. por bogota y bonao/monsenor nouel por bonao
# ny new york
# replacing rep dom 
dt_main['birth_city'].replace(to_replace = ['do', 'dominican republic', 
                                            'dominacana', 'dominicano', 
                                            'republica dominicana'], 
                              value = 'ciudad republica dominicana',
                              inplace = True)

# Eliminar len() <= 1 
dt_main['birth_city'] = dt_main['birth_city'].apply(lambda x: x if x==x and \
                                                    len(x)>2 else np.nan)
    
# Replacing dom for do in birth_country
dt_main['birth_country'].replace(to_replace=['dom'], value='do', inplace=True)


# PRUEBAS ADMIN ***************************************************************

# Revisando missings prueba de admision 
dt_main['check_poma'] = np.where(dt_main.poma_total.isna(),0,1)
dt_main['check_poma'].value_counts()
pd.crosstab(dt_main.year_in, dt_main.check_poma)

dt_main['check_paa'] = np.where(dt_main.paa_mat_fin.isna(),0,1)
pd.crosstab(dt_main.year_in, dt_main.check_paa)

dt_main['check_elash'] = np.where(dt_main.elash_total_fin.isna(),0,1)
pd.crosstab(dt_main.year_in, dt_main.check_elash)

dt_main['check_prueba_adm'] = dt_main[['check_poma', 'check_paa', 
                                       'check_elash']].sum(axis=1)
dt_main['check_prueba_adm'].value_counts()

dt_main['check_prueba_adm2'] = dt_main[['check_paa', 
                                        'check_elash']].sum(axis=1)
dt_main['check_prueba_adm2'].value_counts()

""" POMA no contribuye mucho, elash contribuye menos, por ahora solo usaremos 
PAA. Si uso encoding para procesar los nombres, puedo incluir direcciones y 
quizas agregar un nombre que represente los valores nulos (sin incluir city y 
country) """
pd.crosstab(dt_main.y, dt_main.check_paa)

# Revisando transferidos y programa 2+2 
temp2 = dt_main[dt_main.admin_type.isin(['TRANSFERIDO', 'PROGRAMA 2+2'])]
temp3 = temp2.isna().sum()

# revisando year_in casos con paa missing #TODO fix
#dt1[dt1.paa_mat_fin.isna()].year_in.value_counts().sort_index()

# revisando year_in casos con paa missing para estudiantes activos
#dt1[(dt1.paa_mat_fin.isna()) & (dt1.y != 'active')].year_in.value_counts().sort_index()

""" Hay un tema de registro """

# revisando year_in casos con paa missing para estudiantes activos crossed con si esta grad o no 
pd.crosstab(dt_main[(dt_main.paa_mat_fin.isna()) & (dt_main.y != 'active')].year_in, dt_main[(dt_main.paa_mat_fin.isna()) & (dt_main.y != 'active')].grad)


# TRIM DATA *******************************************************************

## Editando string vars 
list_str_cols = dt_main.columns[dt_main.dtypes == "object"].tolist()
list_str_cols.remove('address')

for var in list_str_cols: 
    # Converting to lower
    dt_main[var] = dt_main[var].str.lower()
    # Stripping vars 
    dt_main[var] = dt_main[var].str.strip()
    # Eliminar tildes strings 
    dt_main[var] = dt_main[var].str.normalize('NFKD')\
                            .str.encode('ascii', errors='ignore')\
                            .str.decode('utf-8')
    # eliminar signos de puntuacion 
    dt_main[var] = dt_main[var].str.replace('[^\w\s]', '', regex=True)


# Filling missings with simple imputations ************************************
    
# Filling birth_country con nationality #nans solved
dt_main['birth_country'] = np.where(dt_main['birth_country'].isna(), 
                                    dt_main['nationality'], 
                                    dt_main['birth_country'])


# Filling nationality con birth_country #nans solved
dt_main['nationality'] = np.where(dt_main['nationality'].isna(), 
                                    dt_main['birth_country'], 
                                    dt_main['nationality'])

# Revisando birth country de paises sin birth city 
dt_main[dt_main.birth_city.isna()].birth_country.value_counts()

# Filling birth city from birth country 
## Asociando birth_country con nombre ciudad + nombre pais
city_missing = {'do': 'ciudad republica dominicana',
                'us': 'ciudad estados unidos', 'ht': 'ciudad haiti',
                've': 'ciudad venezuela', 'cu': 'ciudad cuba',
                'es': 'ciudad espana', 'mx': 'ciudad mexico',
                'kr': 'ciudad corea sur', 'cl': 'ciudad chile',
                'gt': 'ciudad guatemala', 'ec': 'ciudad ecuador',
                'ru': 'ciudad rusia', 'nz': 'ciudad nueva zelanda',
                'it': 'ciudad italia'}

## Rellenando nan in birth_city with values in birth_country
dt_main['birth_city'] = np.where(dt_main['birth_city'].isna(), 
                                 dt_main['birth_country'],
                                 dt_main['birth_city'])

## Sustituyendo birth_country names in birth_city con missing name assigned
dt_main.replace({'birth_city': city_missing}, inplace=True)

#########################################################
#TODO complete

##Cleaning birth city 
datafile = pd.read_csv("C:/Users/DMatos/OneDrive - INTEC/proyecto_final/03_data/set_city_birth.csv", index_col=None)

dict_clean_birth_city = {}
for row in datafile.iterrows():
    dict_clean_birth_city[row[1][0]] = row[1][1]

dt_main.replace({'birth_city': dict_clean_birth_city}, inplace=True)


#*****************************************************************************#
#*********************** Converting strings to numbers ***********************#
#*****************************************************************************#

# Dummies: sex 
dt_main.replace({'sex': {'m': 0, 'f': 1}}, inplace=True)

#-----------------------------------------------------------------------------#

# Categories: 'admin_type', 'nationality', 'birth_country', 'area_in_name', 
# 'birth_city', 'program_in_desc'
le = {}
for var in ['admin_type', 'nationality', 'birth_country', 'area_in_name', 
            'birth_city', 'program_in_desc']:
    le[var] = LabelEncoder().fit(dt_main[var].dropna())
    dt_main[var] = le[var].transform(dt_main[var])

#-----------------------------------------------------------------------------#

# Categories trim: acad_condition, area_name

## acad_condition
### Identifying acad condition vars
acad_cond_vars = [x for x in dt_main.columns if 'acad_condition' in x]

### Identifying all possible classes in acad_condition 
l = []
for var in acad_cond_vars: 
    s = set(dt_main[var].dropna().to_list())
    l = l + list(s)
s = set(l)
    
### Manually assigning a number to each class
acad_cond_class = {'condicion observada': 1, 'en proceso': 2, 'normal': 3,
                   'primera suspension por bajo re': 4,
                   'progreso academico no satisfac': 5, 'prueba academica': 6,
                   'segunda suspension por bajo re': 7, 'separado': 8, 
                   'separado  definitivamente': 9, 'separado permanencia': 10, 
                   'separado por retiro o reprob': 11, 'suspendido': 12,
                   'suspendido exceso asgnatura re': 13}

### Replacing classes to numbers for each var
for var in acad_cond_vars:
    dt_main.replace({var: acad_cond_class}, inplace = True)
    
## area_name
### Identifying acad condition vars
area_name_vars= ['area_in_name'] \
                + [x for x in dt_main.columns if 'area_name' in x]

### Manually assigning a number to each class
area_name_class = {'basicas y ambientales': 1, 'ingenierias': 2, 'negocios': 3,
                   'salud': 4, 'sociales y humanidades': 5}     
           
### Replacing classes to numbers for each var
for var in area_name_vars:
    dt_main.replace({var: area_name_class}, inplace = True)
    
## program_name
### Identifying acad condition vars
prog_name_vars= [x for x in dt_main.columns if 'program_name' in x]   

### Manually assigning a number to each class
prog_name_class = {}
j = 1
for i in (le['program_in_desc'].classes_):
    prog_name_class[i] = j
    j += 1
           
### Replacing classes to numbers for each var using prev LabelEnconder
for var in prog_name_vars:
    dt_main.replace({var: prog_name_class}, inplace = True)

#-----------------------------------------------------------------------------#


#*****************************************************************************#
#************************* Predicting missing values *************************#
#*****************************************************************************#

# Predicting year_birth and completing birth vars including age #miss dropeados

# Predicting sex #missings dropeados

# Predicting PAA
gen_var_list = ['id', 'age_in', 'year_in', 'trim_in', 'program_in_desc', 
                'admin_type', 'credit_conv', 'cred_recon', 'year_birth', 
                'month_birth', 'day_birth', 'dayweek_birth', 'sex', 
                'nationality', 'birth_city', 'birth_country', 'area_in_name']

l = []
for var in ['paa_mat_fin', 'paa_verb_fin', 'paa_redact_fin']: 
    
    ## Original Data
    dt_pred_paa = dt_main[gen_var_list + l + [var]].copy()

    ## Process informacion with EXISTANT data ONLY
    dt_pred_paa_complete = dt_pred_paa.drop(index = dt_pred_paa\
                                .loc[pd.isna(dt_pred_paa[var]), :].index)
    X = dt_pred_paa_complete.drop(columns=[var])
    Y = dt_pred_paa_complete[var]
    reg = LinearRegression().fit(X, Y)

    ## Get missing data and predict model
    dt_pred_paa_miss = dt_pred_paa[dt_pred_paa[var].isna()]
    dt_pred_paa_index = dt_pred_paa[dt_pred_paa[var].isna()].index
    X = dt_pred_paa_miss.drop(columns=[var])
    resultado = reg.predict(X)

    ## Rellenar missing data con prediction
    dt_pred_paa.loc[dt_pred_paa_index, var] = resultado
    dt_main[var] = dt_pred_paa[var].astype(int)
    
    l.append(var)

""" 
# Predicting acad condition 

vars_filled = []
vars_X = []

for i in range(1,13):
    # Variable a rellenar
    var = 'trim'+str(i).str.zfill(2)+'_acad_condition'
    
    # Seleccionando variables predictivas del trimestre 1
    vars_list = [x for x in dt_main.columns if 'trim'+str(i).str.zfill(2) in x]
    for v in ['_nivel', '_grades_status_list', '_select_date_set', 
                '_teachid_list', '_termid', '_acad_condition']:
        vars_list = [x for x in vars_list if v not in x ]
    
    ## Original Data
    dt_pred = dt_main[vars_X + vars_list + vars_filled + [var]].copy()
    
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
    dt_main[var] = dt_pred[var].astype(int)
    
    vars_filled.append(var)
    vars_X = vars_X + vars_list
"""

#*****************************************************************************#
#******************* Separacion de dataset segun modelos *********************#
#*****************************************************************************#

# Pasando y from string to numeric 
dt_main.replace({'y': {'graduate': 0, 'voluntary desertion': 1, 
                       'involuntary desertion':2, 'active':3}}, inplace=True)

# Creating specific y by model/year
pd.crosstab(dt_main.year_desert, dt_main.y)

year_desert_dict = {1:['1st year'], 2:['2nd year'], 3:['3rd year'], 
                    4: ['4th year', '5th year']}
for i in range(1, 5):
    dt_main['y_m'+str(i)] = 0
    dt_main.loc[dt_main.year_desert.isin(year_desert_dict[i]),
                'y_m'+str(i)] = 1 
    print(dt_main['y_m'+str(i)].value_counts()) #TODO ojo, no esta reconociendo '5th year'
    
#dt_main['y_m4'] = 0
#dt_main.loc[dt_main.year_desert == '4th year','y_m4'] = 1 
#dt_main.loc[dt_main.year_desert == '5th year','y_m4'] = 1 

# Keeping active students for evaluating prediction
dt_main_active = dt_main[(dt_main.y == 3) | (dt_main.y.isna())]
dt_main_model = dt_main[dt_main.y.isin([0, 1, 2])]

# Making y dicotomic for modeling 
dt_main_model.replace({'y': {2:1}}, inplace=True)

# Saving dataset with active students
dt_main_active.to_csv(path+'02_prep/dt_std_active.csv', index=False)


# MODELO 1 ####################################################################
""" Modelo para predecir probabilidad de abandono en el primer año, a la 
entrada del estudiante """ 

#******************************************************************************

# Selecting vars to include in dataset for training model #1
vars_list1 = ['id', 'age_in', 'year_in', 'trim_in', 'program_in', 
              'program_in_desc', 'admin_type', 'credit_conv', 'cred_recon', 
              'year_birth', 'month_birth', 'day_birth', 'dayweek_birth', 'sex',
              'nationality', 'birth_city', 'birth_country', 'address', 'city', 
              'country', 'poma_total', 'poma_verb', 'poma_mat', 'poma_struct', 
              'poma_nat', 'poma_soc', 'poma_human', 'paa_verb_fin', 
              'paa_mat_fin', 'paa_redact_fin', 'elash_total_fin', 
              'elash_lang_fin', 'elash_list_fin', 'elash_read_fin', 
              'area_in_name', 'y']

vars_list1 = ['id', 'age_in', 'year_in', 'trim_in', 'program_in_desc', 'admin_type', 
 'credit_conv', 'cred_recon', 'year_birth', 'month_birth', 'day_birth', 
 'dayweek_birth', 'sex', 'nationality', 'birth_city', 'birth_country', 
 'paa_verb_fin', 'paa_mat_fin', 'paa_redact_fin',  'area_in_name', 'y', 'y_m1']

# Dataset for model #1
dt1 = dt_main_model[vars_list1]

# Valores negativos 
des = dt1.describe()

# Missings 
dt1.isna().sum()

#TODO Balance var to predict
dt1.y.value_counts()

# Saving dataset 
dt1.to_csv(path+'02_prep/dt_model1.csv', index=False)

# Outliers
dt1[['age_in', 'year_in', 'trim_in', 'program_in_desc', 'admin_type', 
 'credit_conv', 'cred_recon', 'year_birth', 'month_birth', 'day_birth', 
 'dayweek_birth', 'sex', 'nationality', 'birth_city', 'birth_country', 
 'paa_verb_fin', 'paa_mat_fin', 'paa_redact_fin',  'area_in_name', 'y']].boxplot()

# MODELO 2 ####################################################################
""" Modelo para predecir probabilidad de abandono en el segundo año incluyendo 
datos academicos del primer año"""

## Selecting vars that will be used
var_list2 = [x for x in dt_main.columns if 'trim01_' in x or 'trim02_' in x \
               or 'trim03_' in x or 'trim04_' in x]

## Removing some variables that will not be used
for var in ['_nivel', '_grades_status_list', '_select_date_set', 
            '_teachid_list', '_termid', '_grades_list', ]:
    var_list2 = [x for x in var_list2 if var not in x ]

## FILTER variables and students to keep in dataset
dt2 = dt_main_model[dt_main_model.trim_quant >= 4][['id', 'date_in_year', 
                                                    'date_in_month', 
                                        'date_in_day', 'y','y_m2'] + var_list2]

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
    
#TODO
# Codificar from vector as vector como categorias: trim course_list


dt_temp = dt_main_model[['trim01_course_list', 'trim02_course_list', 
                   'trim03_course_list', 'trim04_course_list']].fillna('')

dt_main_model['year01_course_list'] = dt_temp.apply(lambda x: " ".join(x), 
                                                    axis=1)


# Hacer copia
df_exploded = dt_main_model[["year01_course_list"]].dropna().copy()
# Lista
df_exploded['year01_course_list'] = df_exploded['year01_course_list']\
                                    .apply(lambda x: x.split(" "))
# Explotar listas
df_exploded = df_exploded.explode(column=["year01_course_list"])
# Crear lista a partir de elementos unicos
course_list = list(set(df_exploded["year01_course_list"]))

# one hot encodingg
trimestres = ['trim01_course_list', 'trim02_course_list', 'trim03_course_list', 
              'trim04_course_list']
for c in course_list:
    dt2["course_" + c] = [" ".join(df_row).split().count(c) for df_row in \
                          dt2[trimestres].values]

# Eliminando variables que siguen como strings 
dt2 =  dt2.select_dtypes(exclude=['object'])

# Eliminando course_
dt2.drop(columns='course_', inplace=True)

# Saving datos listos para el modelo 2
dt2.to_csv(path+'02_prep/dt_model2.csv', index=False)

# MODELO 3 ####################################################################
""" Modelo para predecir probabilidad de abandono en el 3er año incluyendo 
datos academicos del segundo año """

## Selecting vars that will be used
var_list3 = [x for x in dt_main.columns if 'trim05_' in x or 'trim06_' in x \
               or 'trim07_' in x or 'trim08_' in x]

## Removing some variables that will not be used
for var in ['_nivel', '_grades_status_list', '_select_date_set', 
            '_teachid_list', '_termid']:
    var_list3 = [x for x in var_list3 if var not in x ]

## FILTER variables and students to keep in dataset
dt3 = dt_main_model[dt_main_model.trim_quant >= 8][['id', 'y', 
                                                    'y_m3'] + var_list3]

## Checking missings 
temp = dt3.isna().sum()

## Filling nan acad_condition 
vars_filled = []
vars_X = []

for i in range(5, 9):
    # Variable a rellenar
    var = 'trim'+str(i).zfill(2)+'_acad_condition'
    
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
    
#TODO
# Codificar from vector as vector como categorias: trim course_list


dt_temp = dt_main_model[['trim05_course_list', 'trim06_course_list', 
                   'trim07_course_list', 'trim08_course_list']].fillna('')

dt_main_model['year02_course_list'] = dt_temp.apply(lambda x: " ".join(x), axis=1)


# Hacer copia
df_exploded = dt_main_model[["year02_course_list"]].dropna().copy()
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

# Saving datos listos para el modelo 2
dt3.to_csv(path+'02_prep/dt_model3.csv', index=False)




# MODELO 4 ####################################################################
""" Modelo para predecir probabilidad de abandono luego del 4to año incluyendo 
datos academicos del tercer año """

## Selecting vars that will be used
var_list4 = []
for i in range(9,13):
    l = [x for x in dt_main.columns if 'trim'+str(i).zfill(2) in x]
    var_list4 = var_list4 + l

## Removing some variables that will not be used
for var in ['_nivel', '_grades_status_list', '_select_date_set', 
            '_teachid_list', '_termid']:
    var_list4 = [x for x in var_list4 if var not in x ]

## FILTER variables and students to keep in dataset
dt4 = dt_main_model[dt_main_model.trim_quant >= 12][['id', 'y', 'y_m4'] + var_list4]

## Checking missings 
temp = dt4.isna().sum()


## Filling nan acad_condition 
vars_filled = []
vars_X = []

for i in range(9, 13):
    # Variable a rellenar
    var = 'trim'+str(i).zfill(2)+'_acad_condition'
    
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
    
#TODO
# Codificar from vector as vector como categorias: trim course_list


dt_temp = dt_main_model[['trim09_course_list', 'trim10_course_list', 
                   'trim11_course_list', 'trim12_course_list']].fillna('')

dt_main_model['year03_course_list'] = dt_temp.apply(lambda x: " ".join(x), 
                                                    axis=1)


# Hacer copia
df_exploded = dt_main_model[["year03_course_list"]].dropna().copy()
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

# Saving datos listos para el modelo 2
dt4.to_csv(path+'02_prep/dt_model4.csv', index=False)




# KEEPING VARS FOR MODELS #####################################################

dt_model = dt_main[vars_list1 + var_list2 + var_list3 + var_list4]







""" RECYCLING CODE
## Eliminando valores negativos en age_in
dt_main['age_in'] = np.where(dt_main['age_in'] <= 10, np.nan, dt_main['age_in'])
dt_main['age_in'] = np.where(dt_main['age_in'] >= 70, np.nan, dt_main['age_in'])

#-----------------------------------------------------------------------------#

# Word embeddings or vector from words codification: 
    #program_desc, birth_city, address*, trim program name, trim coures_list
    
# No incluire address por ahora

vectorize_layer = TextVectorization(
    output_mode='int')


vectorizer = {}
vectorizer['program_desc'] = CountVectorizer()
x = vectorizer['program_desc'].fit_transform(dt_main['program_desc'])
x_df = pd.DataFrame(data = x.toarray(), columns = vectorizer['program_desc']\
                    .get_feature_names_out())

vocab_size = len(vectorizer['program_desc'].get_feature_names_out())

embedding =  Sequential(Embedding(vocab_size, 1, name="embedding"))

embedding.compile('rmsprop', 'mse')
x_embed = embedding.predict(x_df)

temp = {}
vectorizer = {}
embedding = {}
for var in ['program_in_desc', 'birth_city']:
    vectorizer[var] = CountVectorizer().fit(dt_main[var])
    vocab_size = len(vectorizer[var].get_feature_names_out())
    embedding[var] =  Sequential(Embedding(vocab_size, 1, name="embedding"))
    embedding[var].compile('rmsprop', 'mse')
    temp[var] = embedding[var].predict(vectorizer[var]\
                                                      .transform(dt_main[var]).toarray())
    
        
## Original Data
dt_pred_paa_mat = dt_main[['id', 'age_in', 'year_in', 'trim_in', 
                           'program_in_desc', 'admin_type', 'credit_conv', 
                           'cred_recon', 'year_birth', 'month_birth', 
                           'day_birth', 'dayweek_birth', 'sex', 'nationality', 
                           'birth_city', 'birth_country', 'area_in_name', 
                           'paa_mat_fin']].copy()

## Process informacion with EXISTANT data ONLY
dt_pred_paa_mat_complete = dt_pred_paa_mat.drop(index = dt_pred_paa_mat\
                            .loc[pd.isna(dt_pred_paa_mat["paa_mat_fin"]), 
                                                         :].index)
X = dt_pred_paa_mat_complete.drop(columns=['paa_mat_fin'])
Y = dt_pred_paa_mat_complete['paa_mat_fin']
reg = LinearRegression().fit(X, Y)

## Get missing data and predict model
dt_pred_paa_mat_miss = dt_pred_paa_mat[dt_pred_paa_mat["paa_mat_fin"].isna()]
dt_pred_paa_mat_index = dt_pred_paa_mat[dt_pred_paa_mat["paa_mat_fin"].isna()]\
                        .index
X = dt_pred_paa_mat_miss.drop(columns=['paa_mat_fin'])
resultado = reg.predict(X)

## Rellenar missing data con prediction
dt_pred_paa_mat.loc[dt_pred_paa_mat_index, "paa_mat_fin"] = resultado
dt_pred_paa_mat["paa_mat_fin"] = dt_pred_paa_mat["paa_mat_fin"].astype(int)


"""
