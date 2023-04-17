# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 11:34:25 2023

@author: DMatos
"""

#*****************************************************************************#
#***************************Setting up environment****************************#
#*****************************************************************************#

#*LIBRARIES*#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

"""
#*WORKING DIRECTORY*#
import os
cwd = os.getcwd() #for getting current working directory
os.chdir('/tmp') #for changing working directory 
"""

#*DATA*# 
path = "OneDrive - INTEC/proyecto_final/03_data/"
dt_main = pd.read_csv(path + "02_prep/dt_main_pii.csv")


#*****************************************************************************#
#*************************** Visualizing Data ********************************#
#*****************************************************************************#

# Creando dummies para cada categoria de y
y_cat = set(dt_main.y)

y_cat_dict_names = {'active': 'active', 'graduate':'graduate', 
                    'involuntary desertion':'inv_desert', 
                    'voluntary desertion':'vol_desert'}
for cat in y_cat: 
    if cat == cat: #se agrego para obviar nan, mas adelante se contabilizan
        dt_main['y_'+y_cat_dict_names[cat]] = 0
        dt_main.loc[dt_main.y == cat, 'y_'+y_cat_dict_names[cat]] = 1
        
# Creando year order 
conditions = [(dt_main.trim_quant <= 4), 
              (dt_main.trim_quant > 4) & (dt_main.trim_quant <= 8),
              (dt_main.trim_quant > 8) & (dt_main.trim_quant <= 12), 
              (dt_main.trim_quant > 12) & (dt_main.trim_quant <= 16),
              (dt_main.trim_quant > 16)]
choices = ['1st year', '2nd year', '3rd year', '4th year', '5th year +']
dt_main['year_order'] = np.select(conditions, choices, default=np.nan)

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

pie_year_desert = dt_main['year_desert'].value_counts()
pie_year_desert.drop('nan', inplace=True)
pie_year_desert = pie_year_desert/pie_year_desert.sum()*100
pie_year_desert.plot.pie(y='year_desert', autopct='%1.1f%%').yaxis.set_visible(False)
        

#TODO
###############################################################################
# Esta parte puede incluir un análisis descriptivo de los programas 
# (tasa de completion, tasas de retiro, que materias se retiran mas…) 
# y de este ejercicio puede salir un dashboard permanente con este tipo de info 

## Preparing data for viz pr programa *****************************************
temp = dt_main[['id', 'program', 'program_desc', 'y_vol_desert', 'y_active', 
                'y_graduate', 'y_inv_desert']].groupby('program')\
                .agg({'id':'count', 'program_desc': set, 'y_vol_desert': 'sum',
                      'y_active':'sum', 'y_graduate':'sum', 
                      'y_inv_desert':'sum'})

temp['y_sum'] = temp[[x for x in dt_main.columns if 'y_' in x]].sum(axis=1)
temp['check'] = np.where(temp.id == temp.y_sum, 0, 1)
temp['dif'] = temp.id - temp.y_sum
temp.dif.sum()
temp['completion_rate'] = temp.y_graduate/temp.id
temp['dropout_rate'] = temp.y_vol_desert/temp.id*100
#temp[['y_vol_desert', 'y_graduate', 'y_inv_desert']].sum(axis=1)*100

## Sorting temp dataset by id_quantity
temp.sort_values('id', ascending=False, inplace=True)

## Figure 
fig, ax = plt.subplots()
l = []
for var in ['y_graduate', 'y_active', 'y_vol_desert', 'y_inv_desert']:
    ax.bar(temp.index, temp[var], label=var, bottom = temp[l].sum(axis=1))
    l.append(var)
#ax.set_ylabel('Scores')
ax.set_title('Students enrolled by program')
ax.legend()
plt.xticks(rotation=90)
ax.tick_params(axis='x', labelsize=5)
plt.show()

## Separando muy frecuentes de poco frecuentes 

temp_1 = temp[temp.id >= 100]
temp_2 = temp[temp.id < 100]

## Figure mas frecuentes 
fig, ax = plt.subplots()
l = []
for var in ['y_graduate', 'y_active', 'y_vol_desert', 'y_inv_desert']:
    ax.bar(temp_1.index, temp_1[var], label=var, bottom = temp_1[l].sum(axis=1))
    l.append(var)
#ax.set_ylabel('Scores')
ax.set_title('Students enrolled by program')
ax.legend()
plt.xticks(rotation=90)
ax.tick_params(axis='x', labelsize=7)
plt.show()

## Figure menos frecuentes 
fig, ax = plt.subplots()
l = []
for var in ['y_graduate', 'y_active', 'y_vol_desert', 'y_inv_desert']:
    ax.bar(temp_2.index, temp_2[var], label=var, bottom = temp_2[l].sum(axis=1))
    l.append(var)
#ax.set_ylabel('Scores')
ax.set_title('Students enrolled by program')
ax.legend()
plt.xticks(rotation=90)
ax.tick_params(axis='x', labelsize=7)
plt.show()

## Preparing data for viz por area ********************************************
temp1 = dt_main[['id', 'last_area', 'last_area_name', 'y_vol_desert', 
                'y_active', 'y_graduate', 'y_inv_desert']]\
                .groupby('last_area')\
                .agg({'id':'count', 'last_area_name': lambda x:list(set(x))[0],
                      'y_vol_desert':'sum', 'y_active':'sum', 
                      'y_graduate':'sum', 'y_inv_desert':'sum'})
                
## Figure 
fig1, ax1 = plt.subplots()
l = []
for var in ['y_graduate', 'y_active', 'y_vol_desert', 'y_inv_desert']:
    ax1.bar(temp1.last_area_name, temp1[var], label=var, 
            bottom = temp1[l].sum(axis=1))
    l.append(var)
#ax.set_ylabel('Scores')
ax1.set_title('Students enrolled by academic area')
ax1.legend()
plt.xticks(rotation=90)
plt.show()

## Preparing data for viz por year in *****************************************
temp2 = dt_main[['id', 'year_in', 'y_vol_desert', 
                'y_active', 'y_graduate', 'y_inv_desert']]\
                .groupby('year_in')\
                .agg({'id':'count', 'year_in': 'max',
                      'y_vol_desert':'sum', 'y_active':'sum', 
                      'y_graduate':'sum', 'y_inv_desert':'sum'})
temp2['y_sum'] = temp2[[x for x in dt_main.columns if 'y_' in x]].sum(axis=1)
temp2['dif'] = temp2.id - temp2.y_sum
temp2.dif.sum()
                
## Figure 
fig2, ax2 = plt.subplots()
l = []
for var in ['y_graduate', 'y_active', 'y_vol_desert', 'y_inv_desert']:
    ax2.bar(temp2.index, temp2[var], label=var, bottom = temp2[l].sum(axis=1))
    l.append(var)


ax2.set_title('Students enrolled by year')
ax2.legend()
plt.show()

## Algunas tablas *************************************************************

### Year order value counts 
dt_main['year_order'].value_counts(sort=True)

### Year order value counts cumulative 
dt_main['year_order'].value_counts().sort_index(ascending=False).cumsum()

#Distribucion y vs. year_order
pd.crosstab(dt_main.y, dt_main.year_order)
tab = pd.crosstab(dt_main.year_order, dt_main.y)
tab['dropout_rate'] = tab['voluntary desertion']/tab[['active', 'graduate', 
                                                      'involuntary desertion', 
                                                      'voluntary desertion']]\
                      .sum(axis=1)*100
                      
""" Como es de esperar, el numero de estudiantes activos disminuye a medida que 
disminuye el año que cursa, y el numero de graduados aumenta a medida que 
aumenta el año que cursa. Dos comportamientos curiosos son que la desercion 
involuntaria hace pico en el 2do año y se mantiene relativamente alto en los 
siguientes años; y la desercion voluntaria tiene alta ocurrencia en el primer 
año cursado, ~64% de los estudiantes en su primer año abandonan, y el 50% de 
los abandonos voluntarios ocurren en el primer año."""

dt_main[dt_main.y == 'voluntary desertion'].year_order.value_counts()/4719*100                      


###############################################################################
# Análisis descriptivo características estudiantes que desertan voluntariamente

# Sexo 
dt_main[dt_main.y == 'voluntary desertion'].sex.value_counts()
dt_main[dt_main.y == 'involuntary desertion'].sex.value_counts()
""" Los hombres desertan mas voluntaria e involuntariamente """

pie_year_sex = dt_main['sex'].value_counts()
pie_year_sex = pie_year_sex/pie_year_sex.sum()*100
pie_year_sex.plot.pie(autopct='%1.1f%%').yaxis.set_visible(False)

pie_year_sex = dt_main[dt_main.y == 'voluntary desertion'].sex.value_counts()
pie_year_sex = pie_year_sex/pie_year_sex.sum()*100
pie_year_sex.plot.pie(autopct='%1.1f%%').yaxis.set_visible(False)

pie_year_sex = dt_main[dt_main.y == 'involuntary desertion'].sex.value_counts()
pie_year_sex = pie_year_sex/pie_year_sex.sum()*100
pie_year_sex.plot.pie(autopct='%1.1f%%').yaxis.set_visible(False)

# Edad 
dt_main[dt_main.y == 'voluntary desertion'].age_in.value_counts()

# Programa 
dt_main[dt_main.y == 'voluntary desertion'].program.value_counts()

# Program change 
dt_main[dt_main.y == 'voluntary desertion'].program_change_quant.value_counts()

# Nationality 
dt_main[dt_main.y == 'voluntary desertion'].nationality.value_counts()

###############################################################################
# Ojo con desbalances en la data (muy probablemente habran mas observaciones 
# para estudiantes que completan sus programas que data para estudiantes que 
# desertan). Hay que ver como manejar esto y hasta que punto deberíamos estar 
# incluyendo la data. 

## Sexo
dt_main.sex.value_counts()
pd.crosstab(dt_main.y, dt_main.sex)

## Programas 
var = dt_main.program.value_counts()
tab = pd.crosstab(dt_main.y, dt_main.program)

## Años 
dt_main.year_in.value_counts()
pd.crosstab(dt_main.y, dt_main.year_in)

## Y
dt_main.y.value_counts()
""" Si hay desbalance, vamos a ver como se comporta cuando separamos por año"""

dt_main.trim_quant.value_counts().sort_index(ascending=False).cumsum()

dt_main.y.value_counts() #modelo 1 
dt_main[dt_main.trim_quant >= 4].y.value_counts() #modelo 2 
dt_main[dt_main.trim_quant >= 8].y.value_counts() #modelo 3
dt_main[dt_main.trim_quant >= 12].y.value_counts() #modelo 4

###############################################################################
# Explorar que tanto reingreso hay en INTEC y cuáles son las características

dt_main.prev_trims_out_sum.value_counts()

pd.crosstab(dt_main.prev_trims_out_sum, dt_main.grad)

pd.crosstab(dt_main.prev_trims_out_sum, dt_main.sex)

###############################################################################
dt_main['check_first_term'] = np.where(dt_main.year_in == dt_main.trim01_year, 
                                       0, 1)
dt_main['check_first_term'].value_counts()

temp4 = dt_main[dt_main.check_first_term == 1][['year_in','trim_in',
                                                'trim01_year','trim01_trim',
                                                'grad','trim_quant',
                                                'last_program']]
""" Revise si habian estudiantes que habian iniciado antes y no teniamos esos
trimestres registrados, pero no, todos los estudiantes iniciaron en el 2007 or 
later."""
