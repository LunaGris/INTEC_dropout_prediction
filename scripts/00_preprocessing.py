# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 14:37:50 2022
Primer preprocesamiento data Proyecto Final Maestría en Ciencia de Datos
Uso de Machine Learning apra reducir dropout rates en INTEC
@author: DMatos
"""


#*****************************************************************************#
#***************************Setting up environment****************************#
#*****************************************************************************#

#*LIBRARIES*#
import pandas as pd
import numpy as np

"""
#*WORKING DIRECTORY*#
import os
cwd = os.getcwd() #for getting current working directory
os.chdir('/tmp') #for changing working directory 
"""

#*DATA*# 
path = "OneDrive - INTEC/proyecto_final/03_data/"
dt = pd.read_excel(path + "01_raw/Data para Proyecto Investigación.xlsx",
                   sheet_name=None)
dt_program_area = pd.read_excel(path + "01_raw/programas_areas.xlsx")
                                

#*****************************************************************************#
#***************************Preprocessing Data********************************#
#*****************************************************************************#

""" Los datos consisten en un excel con 15 hojas con datos en distintos niveles
 de desagregación """

## some previous cleaning
#dt["Evolución Acad. Alum. 2017-2022"].rename(columns={'ID ESTUDAINTE':"ID ESTU\
#                                                      DIANTE"}, inplace=True)

## Descomprimiendo excel en dataframes separados
dt_registro_est = dt["Programas Alumnos"].copy()
dt_programas = dt["Programas"].copy()
dt_gpa_trim = dt["GPA Por Trimestre"].copy()
dt_evol_acad = pd.concat([dt["Evolución Acad. Alum. 2007-2016"],\
                       dt["Evolución Acad. Alum. 2017-2022"]])
dt_calif = dt["Calificaciones"].copy()
dt_docentes = dt["Docentes"].copy()
dt_docentes_adic = dt["Docentes Adicionales"].copy()
dt_eval = dt["Evaluaciones"].copy()
dt_eval_est = dt["Eval Comp Estudiantes"].copy()
dt_autoeval_doc = dt["Eval Comp Docentes"].copy()
dt_esc_eval = dt["Escala Evaluacion por Trimestre"].copy()
dt_prueba_adm = dt["Pruebas Admision"].copy()
dt_becas = dt["Becas o Patrocinios"].copy()
dt_biograf_est = dt["Datos Biograficos"].copy()
    
## Saving raw form of sheets separately for better facility
"""
dt_names = {'programas_alumnos':dt_registro_est, 'programas':dt_programas, 
            'GPA_por_trimestres': dt_gpa_trim, 
            'evolucion_academica_alumnos': dt_evol_acad, 
            'calificaciones': dt_calif, 'docentes': dt_docentes, 
            'docentes_adicionales': dt_docentes_adic, 
            'evaluaciones': dt_eval, 
            'evaluaciones_completas_estudiantes': dt_eval_est,
            'evaluaciones_completas_docentes': dt_autoeval_doc, 
            'escala_evaluacion_por_trimestre': dt_esc_eval, 
            'pruebas_admisiones': dt_prueba_adm, 
            'becas_patrocinios': dt_becas, 
            'datos_biograficos_estudiantes': dt_biograf_est}

for d in dt_names:      
    dt_names[d].to_csv("OneDrive - INTEC/proyecto_final/03_data/01_raw/raw_"+d\
             +".csv")
"""

## Preparing data     

# Programas Áreas *************************************************************
dt_program_area.rename(columns = {'CODIGO PROGRAMA':'program', 
                                  'PROGRAMA':'program_name', 'NIVEL':'nivel', 
                                  'CODIGO AREA':'area', 
                                  'AREA ACADEMICA':'area_name'}, inplace=True)

# Stripping vars
dt_program_area['program'] = dt_program_area['program'].str.strip()

## Programas ******************************************************************
dt_programas.rename(columns = {'Version': 'version', 'Programa': 'program', 
                               'Descripcion': 'description', 
                               'Asignatura': 'course', 'Trimestre': 'trim', 
                               'Descripcion_Asignatura': 'course_description', 
                               'Creditos Asignatura': 'credits', 
                               'Total Creditos Pensum': 'credits_total', 
                               'PRERREQUISITOS': 'prerequisites', 
                               'Requicito de Creditos': 'credits_prereq', 
                               'COREQUISITOS': 'corequisites'}, inplace = True)

# Stripping vars
dt_programas['program'] = dt_programas['program'].str.strip()

# Checking evol quant trim by program
## grouping by program and version
dt_prog_trim = dt_programas[['version', 'program','description', 'trim', 
                             'credits']].groupby(['program','version'])\
               .agg({'version':'min','description':lambda x:list(x)[0],
                     'trim':'max', 'credits':sum})              
## agregando nivel y area del programa 
dt_prog_trim = dt_prog_trim.merge(dt_program_area[['program','nivel',
                                                   'area_name']], how='left', 
                                  on='program')

## agrupando resultado para ver cant min y max de trimestres para cada programa
dt_prog_trim2 = dt_prog_trim[dt_prog_trim.nivel == 'GRADO'][['program', 'trim',
                                                             'version']]\
                .groupby('program').describe()
## identificando programas para los que ha variado la cantidad de trimestres                
dt_prog_trim2['min_max']=0
dt_prog_trim2.loc[dt_prog_trim2[('trim','min')] 
                  != dt_prog_trim2[('trim','max')], 'min_max'] = 1 

# Programas alumnos / registro estudiantes ************************************
""" 
Nos quedaremos con estudiantes que estén en su primer grado. 

Notas del proceso de exploracion de este dataset:
    - Los cambios de carrera se registran en dos observaciones distintas 
    - Al quedarnos con un dataset solo con ids que tienen al menos un G2, 
    resultan mas G1 que G2 porque hay personas con cambios de carrera en su G1
"""

# Renaming vars
dt_registro_est.rename(columns = {'ID ESTUDIANTE': 'id', 'ACTIVO': 'active', 
                                  'FECHA GRADUACION': 'grad_date', 
                                  'NIVEL': 'nivel', 'AÑO INGRESO': 'year_in',
                                  'TRIMESTRE INGRESO': 'trim_in',
                                  'HONOR': 'honor', 'EDAD INGRESO': 'age_in',
                                  'EDAD EGRESO': 'age_out',
                                  'FECHA INGRESO': 'date_in', 
                                  'PROGRAMA': 'program',
                                  'DESCRPCION PROGRAMA': 'program_desc',
                                  'PROGRAMA INGRESO': 'program_in',
                                  'DESCRIPCION PROGRAMA INGRESO': 
                                      'program_in_desc',
                                  'TIPO SOLICITUD ADMISION': 'admin_type',
                                  'CREDITOS CONVALIDADOS': 'credit_conv',
                                  'CREDITOS RECONOCIDOS': 'cred_recon'},
                       inplace=True)
    
# Stripping vars
dt_registro_est['program_in'] = dt_registro_est['program_in'].str.strip()
dt_registro_est['program'] = dt_registro_est['program'].str.strip()

# Checking nivel distribution of dataset 
dt_registro_est.nivel.value_counts()

# FILTRO 1: Keeping only students with G1
dt_registro_est = dt_registro_est[dt_registro_est.nivel == 'G1']

# Checking if ids in this dataframe are unique
#assert len(set(dt_registro_est.id)) == len(dt_registro_est.id) #ids not unique

# Checking missing values
dt_registro_est.isnull().sum()

# Checking distribución año de ingreso 
dt_registro_est.year_in.value_counts()
dt_registro_est.year_in.plot.hist(grid=True, bins=20, rwidth=0.9,
                                 color='#607c8e') 

# FILTRO 2: Dropping cases with income year before 2000 
# (there are inconsistencies in this early data)
dt_registro_est = dt_registro_est[dt_registro_est.year_in >= 2000]

# FILTRO 3: Dropping cases with income year before 2007 
# (dt_gpa_trim and dt_evol_acad don't have data before that year)
dt_registro_est = dt_registro_est[dt_registro_est.year_in >= 2007]

# Checking que filtrar por NO en CAMBIO PROGRAMA? incluya todos los distintos 
# ids de la base
assert len(set(dt_registro_est.id)) ==\
   len(set(dt_registro_est[dt_registro_est['CAMBIO PROGRAMA?'] == 'NO']['id']))

# FILTRO 4: Keeping observations sin cambio de carrera 
# (hay un registro distinto para cada cambio de carrera, aunque haya habido un 
# cambio anterior el ultimo registro esta en NO para cambio de carrera)
dt_registro_est = dt_registro_est[dt_registro_est['CAMBIO PROGRAMA?'] == 'NO']

"""
#terminar de identificar por que len(set(dt_registro_est.id)) != al numero de 
##casos sin cambiar programa 
#temp = dt_registro_est[['id','PROGRAMA']].groupby(['id']).count()
#ids_repeated = list(temp[temp.PROGRAMA >= 2].index.values)

#keeping observations sin cambio de carrera
temp = dt_registro_est[dt_registro_est.cambio_programa == 0] 
#contando casids repetidos
temp2 = temp[['id','PROGRAMA']].groupby(['id']).count() 
ids_repeated = list(temp2[temp2.PROGRAMA >= 2].index.values) 
#hay 48 casos sin marcar con cambios de carrera que tienen multiples 
##observaciones, revisar estos casos en dt_evol_acad, podrian ser reingresos
temp3 = temp[temp.id.isin(ids_repeated)] 
#por ahora me quedare con ultima observacion de estos 51 casos
temp3.drop_duplicates(inplace=True)

temp4 = dt_evol_acad[dt_evol_acad.id.isin(ids_repeated)]
temp5 = dt_gpa_trim[dt_gpa_trim.id.isin(ids_repeated)]
#cuantas veces ha cambiado de programa/carrera para llegar al current program
#CONCLUSION
#parecen haber casos de simples duplicados por registros, parece haber un error 
en el sistema para otros que generaba otra observacion para FECHA INGRESO 
2017-02-06, y otros con errores en registro (no reportan cambios de carrera 
aunque si hubo)
"""

# FILTRO 5: Dropeando duplicados de id y quedandonos con la primera observacion
#algunos ids tienen mas de una observacion sin reportar cambio de carrera
#se obviara esta info de este subset de datos y se obtendra de dt_gpa_trim
dt_registro_est.drop_duplicates(subset=['id'], inplace=True) #19227

# Checking if ids in this dataframe are unique
assert len(set(dt_registro_est.id)) == len(dt_registro_est.id) #ids not unique

# Creating vars
# ronda 2, con filtros aplicados
dt_registro_est['grad'] = 0
dt_registro_est.loc[dt_registro_est.grad_date == dt_registro_est.grad_date, 
                    'grad'] = 1 #si se graduo 
dt_registro_est['program_change'] = np.select([dt_registro_est['program'].eq\
                                             (dt_registro_est['program_in'])],\
                                              [0], 1)
    #si ha cambiado o no de programa/carrera en el current program

# FILTRO 6: Dropping cases enrolled in term 2022-4
# clearly the dataset does not contain all students matriculated in that term
# Seems like it mostly includes freshman students 
dt_registro_est['termid_in'] = dt_registro_est[["year_in", "trim_in"]]\
                               .astype(str).apply("_".join, axis=1)
dt_registro_est = dt_registro_est[dt_registro_est.termid_in != "2022_4"]
    
# Datos biograficos ***********************************************************  

# Renaming vars 
dt_biograf_est.rename(columns = {'ID ESTUDIANTE': 'id', 
                                 'FECHA NACIMIENTO': 'dob', 'SEXO': 'sex', 
                                 'NACIONALIDAD': 'nationality', 
                                 'CIUDAD NACIMIENTO': 'birth_city', 
                                 'PAIS NACIMIENTO': 'birth_country', 
                                 'DIRECCION ACTUAL': 'address',
                                 'CUIDAD ACTUAL': 'city', 
                                 'PAIS ACTUAL': 'country'}, inplace = True)

# Checking if ids in this dataframe are unique
assert len(set(dt_biograf_est.id)) == len(dt_biograf_est.id)  #ids unique        

# Changing empty strings to NaN  
dt_biograf_est.replace(r'^\s*$', np.nan, regex=True, inplace=True)  
dt_biograf_est.isnull().sum()

# Evaluaciones docentes *******************************************************


# Evolución académica *********************************************************

"""
# Resetting dt_eval_acad
dt_evol_acad = pd.concat([dt["Evolución Acad. Alum. 2007-2016"],\
                       dt["Evolución Acad. Alum. 2017-2022"]])
"""

# Renaming vars
dt_evol_acad.rename(columns = {'ID ESTUDIANTE': 'id', 'CLAVE': 'clave',
                               'SECCION': 'sec', 'AÑO': 'year', 
                               'TRIMESTRE': 'trim', 'ID DOCENTE': 'teachid', 
                               'FECHA SELECCION': 'select_date_time', 
                               'RETIRO': 'retiro', 
                               'FECHA RETIRO': 'retiro_date_time', 
                               'CAIFICACION': 'grade', 'CREDITOS': 'credits'}, 
                    inplace=True)

# Exploring
temp = dt_evol_acad.head(10000)

"""la var retiro_date la obviare, tiene muchos missings cuando si hubo retiro 
y tambien hay fechas marcadas cuando no hubo retiros que habria que dropear. 
Si hubiese estado mas consistente, habria sido interesante incluir en el modelo
a qué altura del trimestre retiro (dias de diferencia entre dia de selección y 
día de retiro pudo haber sido una manera) """
dt_evol_acad['retiro_date_dum'] = 0
dt_evol_acad.loc[dt_evol_acad.retiro_date_time==dt_evol_acad.retiro_date_time, 
                    'retiro_date_dum'] = 1 
pd.crosstab(dt_evol_acad.retiro, dt_evol_acad.retiro_date_dum)

"""todos los retiros tienen su correspondiente R o RI asignada en grade"""
tb = pd.crosstab(dt_evol_acad.retiro, dt_evol_acad.grade)

# Cleaning grade var 
""" Llame y pregunte por estas letras para reemplazarlas correctamente, me 
recomendaron comunicarme con area academica. Por ahora continuare con las 
principales letras que ya conozco, las demas quedan pendientes de confirmar.""" 

""" Resulta que en la data que me compartio Victor la hoja que converti a 
dt_calif es esto lo que incluye, que significan las letras, que GPA points dan 
y que calificacion representan. Aqui dejo las equivalencias para simplificar 
la variable de grade: 
    - Convalidaciones (CO): AI, AU, CI, CM, CO, CU, EC, EI, ET, EX, FV, IT, MU, 
      NC, NT, PF, PS, RI, SM, TC, TM, UB, UC, UN, US, UT, VA, VI
    - Reprobaciones: D, F, Q
    - Otros (OTH): I (incompleto), NA (no asistio), NS
    - Aprobaciones: A, B, B+, C, C+, P, S (satisfactorio) + CO
    - Retiros: R, RE, RO
"""

## Guardando grades originales en una var separada
dt_evol_acad.rename(columns = {'grade': 'grade_orig'}, inplace = True)
dt_evol_acad['grade'] = dt_evol_acad['grade_orig'].copy()

## Replacing withdraws RE y RO to R
dt_evol_acad['grade'].replace(to_replace = ['RE', 'RO'], value = 'R', 
                              inplace = True)

## Replacing validations 
dt_evol_acad['grade'].replace(to_replace = ['AI', 'AU', 'CI', 'CM', 'CO', 'CU', 
                                            'EC', 'EI', 'ET', 'EX', 'FV', 'IT', 
                                            'MU', 'NC', 'NT', 'PF', 'PS', 'RI', 
                                            'SM', 'TC', 'TM', 'UB', 'UC', 'UN', 
                                            'US', 'UT', 'VA', 'VI'],
                              value = 'CO', inplace = True)

## Replacing I and NA to OTH 
dt_evol_acad['grade'].replace(to_replace = ['I', 'NA', 'NS'], value = 'OTH', 
                              inplace = True)

# Creando dummies para cada letter grade
grades = set(dt_evol_acad.grade)

for g in grades: 
    if g == g: #se agrego para obviar nan, mas adelante se contabilizan
        dt_evol_acad['grade_'+g] = 0
        dt_evol_acad.loc[dt_evol_acad.grade == g, 'grade_'+str(g)] = 1
        
# Changing to lower all characters in columns names and + sign to letters
dt_evol_acad.columns = dt_evol_acad.columns.str.lower()
dt_evol_acad.columns = dt_evol_acad.columns.str.replace("+", "_plus")

# Identificando cursos aprobados, reprobados y retirados
cond_grade_stat = [(dt_evol_acad['grade'].isin(['A','B','B+','C','C+','CO','S',
                                                'P'])), 
                   (dt_evol_acad['grade'].isin(['D','F','Q'])), 
                   (dt_evol_acad['grade'].isin(['R'])),
                   (dt_evol_acad['grade'].isin(['OTH']))]
choi_grade_stat= ['passed', 'failed', 'withdrawn', 'other']
dt_evol_acad['grade_status'] = np.select(cond_grade_stat, choi_grade_stat, 
                                         default=np.nan)
# Creando dummies para cada grade_status
for status in choi_grade_stat:
    dt_evol_acad[status] = 0
    dt_evol_acad.loc[dt_evol_acad.grade_status == status, status] = 1  

# Vars para sumar cant creditos aprobados/reprobados/retirados
for status in choi_grade_stat:
    dt_evol_acad['credits_'+status] = dt_evol_acad[status]\
                                      * dt_evol_acad['credits']

# Explorando registros sin grades asignadas
temp2 = dt_evol_acad[dt_evol_acad.grade.isnull()]
temp2.credits.value_counts()
temp2.isnull().sum()
temp2.year.value_counts()
"""Hay 89993 registros que no tienen grade, por ahora solo identificare estas 
asignaturas para ver que tanto impacto estan teniendo en el dt_main"""

# Identificando registros sin grades asignadas (nan grades dummie)
dt_evol_acad['grade_nan'] = 0
dt_evol_acad.loc[dt_evol_acad.grade.isnull(), 'grade_nan'] = 1
dt_evol_acad['credits_nan'] = dt_evol_acad['grade_nan']*dt_evol_acad['credits']
## Checking if this makes sense with previous grades dummies vars created
## aprobados/reprobados/retirados dummies
assert dt_evol_acad[dt_evol_acad.grade_nan==1][choi_grade_stat].sum().sum()==0
## Letter grades dummies
grades_vars = [x for x in dt_evol_acad.columns if "grade_" in x]
grades_vars.remove('grade_status')
grades_vars.remove('grade_nan')
assert dt_evol_acad[dt_evol_acad.grade_nan==1][grades_vars].sum().sum()==0

# Separating selection date and time 
dt_evol_acad['select_date'] = dt_evol_acad['select_date_time'].dt.date
dt_evol_acad['select_time'] = dt_evol_acad['select_date_time'].dt.time

# Creating var which identifies current year and trim 
dt_evol_acad['termid'] = dt_evol_acad[["year", "trim"]].astype(str)\
                         .apply("_".join, axis=1)
dt_evol_acad['id_termid'] = dt_evol_acad['id'].astype(str) + '_' \
                            + dt_evol_acad['termid']
    
# groupby (indicadores por trimestre que cursa, 
#hay que crear un identificador para esto,hasta ahora tenemos trimestre del año
#pero no si es su primer trimestre o cual)
dt_evol_acad_flat = dt_evol_acad[['id','clave', 'year', 'trim', 'teachid', 
                                  'select_date','grade','credits', 
                                  'grade_status', 'passed', 'failed', 
                                  'withdrawn', 'other', 'credits_passed', 
                                  'credits_failed', 'credits_withdrawn',
                                  'credits_other', 'grade_nan', 'credits_nan',
                                  'grade_d', 'grade_oth', 'grade_b', 
                                  'grade_co', 'grade_b_plus', 'grade_f', 
                                  'grade_c', 'grade_c_plus', 'grade_a', 
                                  'grade_r', 'id_termid']]\
                    .groupby(['id_termid'])\
                    .agg({'id':'min','clave':[len, list], 'year':'min', 
                          'trim':'max', 'teachid':list, 'select_date':set, 
                          'grade':list,'credits':'sum', 'grade_status':list, 
                          'passed':'sum', 'failed':'sum', 'withdrawn':'sum', 
                          'other':'sum', 'credits_passed':'sum',
                          'credits_failed':'sum', 'credits_withdrawn':'sum',
                          'credits_other':'sum', 'grade_nan':'sum', 
                          'credits_nan':'sum', 'grade_d':'sum', 
                          'grade_oth':'sum', 'grade_b':'sum', 'grade_co':'sum',
                          'grade_b_plus':'sum', 'grade_f':'sum', 
                          'grade_c':'sum', 'grade_c_plus':'sum', 
                          'grade_a':'sum', 'grade_r':'sum'})

"""
temp6 = dt_evol_acad[['id', 'select_date_time','termid']]\
    .groupby(['id', 'termid']).agg({'select_date_time':['min','max',set]})
temp6['dum'] = 0
temp6.loc[temp6[('select_date_time', 'min')] == temp6[('select_date_time', 
                                                       'max')], 'dum'] = 1
temp6['min'] = min(temp6[('select_date_time','set')])
"""
# Renaming columns
## changing flatten tabla levels order and renaming
dt_evol_acad_flat = dt_evol_acad_flat.pipe(lambda x: x.set_axis(map('_'.join, 
                                                                   x), axis=1))

dt_evol_acad_flat.rename(columns={'clave_len':'course_quant', 
                                  'clave_list':'course_list', 
                                  'credits_sum': 'credits_total',
                                  'passed_sum':'passed_quant', 
                                  'failed_sum':'failed_quant', 
                                  'withdrawn_sum':'withdrawn_quant',
                                  'other_sum':'other_quant',
                                  'grade_nan_sum':'grade_nan_quant'}, 
                         inplace=True)

dt_evol_acad_flat.columns = dt_evol_acad_flat.columns.str.replace("_min", "") 
dt_evol_acad_flat.columns = dt_evol_acad_flat.columns.str.replace("_max", "") 
#dt_evol_acad_flat.columns = dt_evol_acad_flat.columns.str.replace("_list", "")
#dt_evol_acad_flat.columns = dt_evol_acad_flat.columns.str.replace("_set", "") 
dt_evol_acad_flat.columns = dt_evol_acad_flat.columns.str.replace("_sum", "")
dt_evol_acad_flat.columns = dt_evol_acad_flat.columns.str.replace("grade_", 
                                                                  "grades_")                          

#cant de asignaturas seleccionadas
#cantidad de creditos seleccionados
#cantidad de A (o %) y cada letra #crear other para valores menos comunes 
#cant de asignaturas aprobadas
#cant de asignaturas retiradas
#cant de asignaturas reprobadas 
#cant de creditos aprobados
#cant de creditos retirados
#cant de creditos reprobados
#promedio evaluacion profesoral trimestre anterior
#promedio perfil docentes 
#cantidad asignaturas por areas academicas #no hace sentido esto aqui ❌
#que trimestre del año es 
#TODO: no se como/si hay que incluir identificador de linea de tiempo, e.g.año
#TODO: no se si agregar identificador de cambio de carrera en cada trimestre 
#y/o identificador de cual cambio es, si su primer programa, segundo, etc. 
##esto ultimo se podria hacer en dt_gpa_trim, no aqui

#combinacion de asignaturas 
#(esto quizas hace mas sentido para desercion por asignatura)

"""informacion mas detallada/granular puede servir para predecir desercion de 
por asignatura"""

# GPA por Trimestre ***********************************************************

"""
# Reset dt_gpa_trim
dt_gpa_trim = dt["GPA Por Trimestre"].copy()
"""

# Renaming vars
dt_gpa_trim.rename(columns = {'ID ESTUDIANTE': 'id', 'AÑO': 'year', 
                              'TRIMESTRE': 'trim', 'PROGRAMA': 'program', 
                              'NIVEL': 'nivel', 'GPA TRIMESTRAL': 'gpa_trim', 
                              'GPA ACUMULADO': 'gpa_cum', 
                              'CONDICION ACADEMICA': 'acad_condition'}, 
                   inplace=True)

#año, trimestre, programa, GPA trim, GPA acum, condicion adademica

# agregar area academica a este dataset
dt_gpa_trim = dt_gpa_trim.merge(dt_program_area[['program','area','area_name',
                                           'program_name']], 
                          how='left', on='program')

# creating id-termid for gpa_trim
dt_gpa_trim['termid'] = dt_gpa_trim['year'].astype(str)\
                        + '_' + dt_gpa_trim['trim'].astype(str)
dt_gpa_trim['id_termid'] = dt_gpa_trim['id'].astype(str) + '_'\
                           + dt_gpa_trim['termid']

# mergear info trimestral obtenida de evol_acad con gpa_trim
dt_trim = dt_gpa_trim.merge(dt_evol_acad_flat, how='left', on='id_termid')

# some cleaning after merge

## assering if merge was done correctly
assert dt_trim['id_x'].equals(dt_trim['id_y'])

## revisando casos en los que id_x != id_y 
dt_trim['new'] = np.where((dt_trim['id_x'] == dt_trim['id_y']), 0, 1) 
dt_trim.new.value_counts() #hay 46 (43 filtrando por G1) casos sin data de evol 
#acad, aparentemente se inscribieron pero no llegaron a seleccionar
#TODO no se si contarlos como desercion voluntaria o si solo dropearlos
temp = dt_trim[dt_trim.new == 1]
temp.area_name.value_counts()
temp.termid.value_counts()

## eliminando y renombrando duplicated variables
dt_trim.drop(['id_y', 'year_y', 'trim_y'], axis=1, inplace=True)
dt_trim.rename(columns={'id_x':'id', 'year_x':'year', 'trim_x':'trim'},
               inplace=True)

# FILTRO 1: Filtrando para estudiantes de G1 
dt_trim = dt_trim[dt_trim.nivel == 'G1']

# creating trimesters order (which is the 1st, 2nd...)
#luego de tener solo los trimestres del primer grado para todos los estudiantes
dt_trim['trim_order'] = dt_trim.groupby(['id'])['termid'].rank(method='dense')
dt_trim.trim_order.value_counts()
dt_trim['trim_order'] = dt_trim['trim_order'].astype(int)
dt_trim['trim_order_str'] = 'trim'+dt_trim['trim_order'].astype(str)\
                            .str.zfill(2)
                            
# FILTRO 2: Dropping cases enrolled in term 2022-4
# clearly the dataset does not contain all students matriculated in that term
# Seems like it mostly includes freshman students 
## Revisando
temp = dt_trim[dt_trim.termid == "2022_4"]
temp.shape #347 estudiantes matriculados en este term
temp.trim_order.value_counts() #para 307 de ellos es su primer trimestre
## Filtrando
dt_trim = dt_trim[dt_trim.termid != "2022_4"]

# Pruebas de admision *********************************************************

# Renaming vars 
dt_prueba_adm.rename(columns = {'ID ESTUDIANTE': 'id', 
                                'PAA RAZONAMIENTO VERBAL 1': 'paa_verb', 
                                'PAA RAZONAMIENTO MATEMATICO 1': 'paa_mat', 
                                'PAA REDACCION INDIRECTA 1 ': 'paa_redact', 
                                'PAA RAZONAMIENTO VERBAL 2': 'paa_verb_2', 
                                'PAA RAZONAMIENTO MATEMATICO 2': 'paa_mat_2', 
                                'PAA REDACCION INDIRECTA 2': 'paa_redact_2', 
                                'POMA TOTAL': 'poma_total', 
                                'POMA CONCEPTOS VERBALES': 'poma_verb', 
                                'POMA CONCEPTOS MATEMATICOS': 'poma_mat', 
                                'POMA ESTRUCTURAS ESPACIALES': 'poma_struct', 
                                'POMA CONCEPTOS NATURALES': 'poma_nat', 
                                'POMA CONCEPTOS SOCIALES': 'poma_soc', 
                                'POMA COMPORTAMIENTOS HUMANOS': 'poma_human', 
                                'ELASH TOTAL': 'elash_total', 
                                'ELASH LANGUAGE': 'elash_lang',  
                                'ELASH LISTENING': 'elash_list', 
                                'ELASH READING': 'elash_read', 
                                'ELASH READING V2': 'elash_read_2', 
                                'ELASH LANGUAGE V2': 'elash_lang_2', 
                                'ELASH TOTAL V2': 'elash_total_2', 
                                'ELASH LISTENING V2': 'elash_list_2', 
                                'PAEP': 'paep', 
                                'PAEP RAZONAMIENTO VERBAL': 'paep_verb', 
                                'PAEP RAZONAMIENTO MATEMATICO': 'paep_mat', 
                                'PAEP HABILIDADES COGNITIVAS': 'paep_cogn', 
                                'PAEP REDACCION': 'paep_redact', 
                                'PAEP INGLES': 'paep_en', 
                                'PAI - PRUEBA ADMISION INTEC': 'pai', 
                                'PAI LENGUA ESPAÑOLA': 'pai_sp', 
                                'PAI MATEMATICA': 'pai_mat', 
                                'PAI - PRUEBA ADMISION INTEC 2': 'pai_2', 
                                'PAI LENGUA ESPAÑOLA 2': 'pai_sp_2', 
                                'PAI MATEMATICA 2': 'pai_mat_2', 
                                'PRUEBA EPAA': 'epaa', 
                                'EPAA MATEMATICA': 'epaa_mat', 
                                'EPAA VERBAL': 'epaa_verb', 
                                'PRUEBA EPAA 2 ': 'epaa_2', 
                                'EPAA MATEMATICA 2': 'epaa_mat_2', 
                                'EPAA VERBAL 2': 'epaa_verb_2'}, 
                     inplace = True)

# Checking distinct ids
#assert len(set(dt_prueba_adm.id)) == len(dt_prueba_adm.id)

# revisar hay 2 casos duplicados
temp = dt_prueba_adm[dt_prueba_adm.id.isin([37724,37241])]
#si hay unos duplicados extraños, pero estos ids no estan en dt_main
temp2 = dt_registro_est[dt_registro_est.id.isin([37724,37241])]

# Revisar columns equivalentes y cuales pruebas son las que aplican para grado
tb = dt_prueba_adm.describe()
"""no encontre informacion sobre pruebas PAI y EPAA, y tienen pocas 
observaciones, las obviare. PAA, POMA e ELASH son pruebas para grado. PAA e 
ELASH tienen columnas con versiones 2, estas seran analizadas para determinar
con que quedarnos."""

# Keeping pruebas para grado 
dt_prueba_adm = dt_prueba_adm[['id', 'paa_verb', 'paa_mat', 'paa_redact', 
                               'paa_verb_2', 'paa_mat_2', 'paa_redact_2', 
                               'poma_total', 'poma_verb', 'poma_mat', 
                               'poma_struct', 'poma_nat', 'poma_soc', 
                               'poma_human', 'elash_total', 'elash_lang', 
                               'elash_list', 'elash_read', 'elash_read_2', 
                               'elash_lang_2', 'elash_total_2', 
                               'elash_list_2']]

# Exploring dataset
tb = dt_prueba_adm.describe()

## Identificando ids con resultados para cruzar datos y ver si pueden 
## complementarse los resultados de las pruebas, creamos algunas dummies
for var in ['paa_verb','paa_verb_2','poma_total','elash_total',
            'elash_total_2']:
    dt_prueba_adm['dum_'+var] = 1
    dt_prueba_adm.loc[dt_prueba_adm[var].isnull(), 'dum_'+var] = 0
    
dt_prueba_adm['dum_elash'] = dt_prueba_adm['dum_elash_total'] \
                           + dt_prueba_adm['dum_elash_total_2']

## Cross tables 
## PAA vs. PAA_2
pd.crosstab(dt_prueba_adm.dum_paa_verb, dt_prueba_adm.dum_paa_verb_2)
# most cases with paa_2 also have grades for paa (just 1 doesn't)
## ELASH vs. ELASH_2
pd.crosstab(dt_prueba_adm.dum_elash_total, dt_prueba_adm.dum_elash_total_2)
# most cases with elash_2 dont have elash but 93 cases have both
## PAA vs. POMA
pd.crosstab(dt_prueba_adm.dum_paa_verb, dt_prueba_adm.dum_poma_total)
# most cases with poma also have grades for paa (81 cases don't)
## PAA vs. ELASH
pd.crosstab(dt_prueba_adm.dum_paa_verb, dt_prueba_adm.dum_elash)
# most cases with elash have paa, but many (3617) don't

## Visualizing outliers results POMA
poma = ['poma_total', 'poma_verb', 'poma_mat', 'poma_struct','poma_nat', 
        'poma_soc', 'poma_human']
dt_prueba_adm[poma].boxplot()

# Cleaning outliers o valores sin sentidos 

## Replacing paa_verb value for id 51227 from 1595 to 5595
idx = dt_prueba_adm[dt_prueba_adm.id == 51227].index[0]
dt_prueba_adm.at[idx, 'paa_verb'] = 595 #seems to be the right value to me

## Replacing values execeeding 800 in paa_verb, paa_redact
dt_prueba_adm.loc[dt_prueba_adm['paa_verb'] > 800, 'paa_verb'] = np.nan
dt_prueba_adm.loc[dt_prueba_adm['paa_redact'] > 800, 'paa_redact'] = np.nan

## Replacing 0 por nan en PAA_redact y elash_total_2
dt_prueba_adm.loc[dt_prueba_adm['paa_redact'] == 0, 'paa_redact'] = np.nan
dt_prueba_adm.loc[dt_prueba_adm['elash_total_2'] == 0, 
                  'elash_total_2'] = np.nan

## Changing values bigger than 4000 and smaller than 0 to nan
for var in poma:
    dt_prueba_adm.loc[dt_prueba_adm[var] > 4000, var] = np.nan
    dt_prueba_adm.loc[dt_prueba_adm[var] < 0, var] = np.nan

# Promedios de versiones 1 y 2 de las pruebas PAA e ELASH
paa = ['paa_verb', 'paa_mat', 'paa_redact']
dt_prueba_adm['paa_avg'] = dt_prueba_adm[paa].mean(axis=1)
paa_2 = ['paa_verb_2', 'paa_mat_2', 'paa_redact_2']
dt_prueba_adm['paa_avg_2'] = dt_prueba_adm[paa_2].mean(axis=1)

# Me quedo con version con mayor puntuacion para PAA e ELASH
vars_fin = ['paa_verb', 'paa_mat', 'paa_redact', 'elash_total', 'elash_lang',
'elash_list', 'elash_read'] 
for var in vars_fin:
    dt_prueba_adm[var+'_fin'] = pd.Series(int)
    dt_prueba_adm[var+'_fin'] = np.where(dt_prueba_adm['paa_avg_2'] \
                                         > dt_prueba_adm['paa_avg'], 
                                         dt_prueba_adm[var+'_2'], 
                                         dt_prueba_adm[var])

# List with vars _fin
paa_elash_fin = [x for x in dt_prueba_adm.columns if '_fin' in x]

## Becas o patrocinios ********************************************************

# Renaming vars 
dt_becas.rename(columns = {'ID ESTUDIANTE': 'id', 
                           'TIPO PATROCINIO': 'sponsor_type', 
                           'TIPO COVERTURA': 'cover_type', 
                           'ESTATUS': 'status', 
                           'FECHA APROBACION': 'approval_date', 
                           'AÑO INICIO': 'year_start_sponsor', 
                           'TRIMESTRE INICIO': 'trim_start_sponsor', 
                           'MONTO TRIMESTRAL': 'trim_amount', 
                           'PORCENTAJE TRIMESTRAL': 'trim_percent'}, 
                inplace = True)

# Checking distinct ids
#assert len(set(dt_becas.id)) == len(dt_becas.id)

#TODO revisar duplicados (2234 casos)
dt_becas['dup_id'] = dt_becas.duplicated('id',keep=False)
temp = dt_becas[dt_becas.dup_id == True]
temp1 = dt_becas[dt_becas.id.isin(dt_registro_est.id)]
temp1['dup_id'] = temp1.duplicated('id',keep=False)
temp1 = temp1.merge(dt_registro_est[['id','year_in','trim_in',
                                     'program_in_desc','admin_type','grad']], 
                    how='left', on='id')

#TODO Identificar a que programa corresponde cada patrocinio para cada estudiante
## Revisar si year_start_sponsor siempre es igual a year de approval date
dt_becas['appoval_year'] = dt_becas['approval_date'].dt.year
dt_becas['comp_year'] = np.where(dt_becas.year_start_sponsor \
                                 == dt_becas.appoval_year, 0, 1)



#*****************************************************************************#
        
## Exploring data
""" Quiero un dataset cuadrado/rectangular con una observacion por estudiante y 
que vaya incluyendo las variables/características para explorar y modelar """

# ¿Para qué años hay data en cada subset?

dt_registro_est.year_in.plot.hist() #a partir de 1986 en data original
dt_gpa_trim.year.plot.hist() #a partir del 2007
dt_evol_acad.year.plot.hist() #a partir del 2007

# Creating flat dataset

## biograf data + register data
dt_main = dt_registro_est[['id', 'age_in', 'age_out', 'date_in', 'grad_date',
 'honor', 'nivel', 'program', 'program_desc', 'active', 'year_in', 'trim_in',
 'program_in', 'program_in_desc', 'admin_type', 'credit_conv', 'cred_recon',
 'grad', 'program_change']].copy()

### Merging biography data
dt_main = dt_main.merge(dt_biograf_est, how='left', on='id')

## merging admin test results to main dataset
dt_main = dt_main.merge(dt_prueba_adm[['id']+poma+paa_elash_fin], how='left', 
                        on='id', indicator='admin_test_merge')
### checking merge status
print(dt_main.admin_test_merge.value_counts())
dt_main.drop('admin_test_merge', axis=1, inplace=True)

##TODO merging sponsorship data
#dt_main = dt_main.merge(dt_becas, how='left', on='id')

#TODO Revisar missings en data, los casos que no tienen grades o biograph info

#TODO Revisar outliers

## PROCESING TRIM DATA 

#TODO hasta ahora tengo todo sobre calificaciones y aprobaciones en cantidades,
#quizas lo ideal seria alimentarlo al modelo como %

## flattening trim data #pivot (trimestres en las columnas) (from long to wide)
dt_trim_flat = dt_trim.drop('new',axis=1).pivot(index='id', 
                                                columns='trim_order_str')
## changing flatten tabla levels order and renaming
dt_trim_flat = dt_trim_flat.reorder_levels([1,0],axis=1)\
               .pipe(lambda x: x.set_axis(map('_'.join, x), axis=1))
                   
## sorting columns in flatten dataframe and reseting index
dt_trim_flat = dt_trim_flat.reindex(sorted(dt_trim_flat), axis=1).reset_index()                                             
                                                    
assert len(set(dt_trim.id)) == dt_trim_flat.shape[0]
"""en este dataset hay 23585 distintos ids de estudiantes es probable que 
contenga data para estudiantes que iniciaron antes del 2007 para mantener la 
consistencia, mantendremos el analisis para estudiantes que iniciaron en el 
2007 o despues"""

## Merging trim flat data 
dt_main = dt_main.merge(dt_trim_flat, how='left', on='id')

# Explporando dt para crear var to predict ************************************

## Var active in dt_main
## parece consistente con si el estudiantes esta o no activo en intec
pd.crosstab(dt_main.active, dt_main.grad)

# Revisar 5536 casos no grad no activos vs. cond acad de su ultimo trimestre
## Generando var con cantidad de trimestres cursados por cada estudiante
dt_main['trim_quant'] = dt_main[[x for x in dt_main.columns if '_trim_order'\
                                 in x]].max(axis=1)
dt_main['trim_quant'] = dt_main['trim_quant'].astype('Int16')

## Creando var que contenga ultima condicion academica de cada estudiante
## manteniendo vars con _acad_condition en nombre y aplicando funcion que 
## mantiene ultima condicion academica
dt_main['last_acad_condition'] = dt_main[[x for x in dt_main.columns if \
                                          '_acad_condition' in x]]\
    .apply(lambda x: list(x.dropna())[-1] if (len(list(x.dropna())) != 0)\
           else np.nan, axis=1)

## Creating var with lates year enrolled
dt_main['last_year_enroll'] = dt_main[[x for x in dt_main.columns if \
                                          '_year' in x]]\
    .apply(lambda x: list(x.dropna())[-1] if (len(list(x.dropna())) != 0)\
           else np.nan, axis=1)
### Checking last_year_enroll
dt_main['last_year_enroll'].value_counts()
        
## Creating var with lates trim enrolled
dt_main['last_trim_enroll'] = dt_main[[x for x in dt_main.columns if '_trim' \
                                       in x and 'm_' not in x and 'gpa' not in\
                                       x]]\
    .apply(lambda x: list(x.dropna())[-1] if (len(list(x.dropna())) != 0)\
           else np.nan, axis=1)
### Checking last_trim_enroll
dt_main['last_trim_enroll'].value_counts()

## Creating var with latest termid
dt_main['last_termid'] = dt_main[[x for x in dt_main.columns if '_termid'\
                                       in x and 'id_' not in x]]\
    .apply(lambda x: list(x.dropna())[-1] if (len(list(x.dropna())) != 0)\
           else np.nan, axis=1)

## CASOS ESTUDIANTES NO ACTIVOS NO GRADUADOS ----------------------------------
## Revisando 5536 casos no graduados no activos vs. ultima condicion academica
dt_main[(dt_main.grad == 0) & (dt_main.active == 'NO')].last_acad_condition\
    .value_counts()
### No suman 5536
dt_main[(dt_main.grad == 0) & (dt_main.active == 'NO')].last_acad_condition\
    .value_counts().sum()
### Contando nas
dt_main[(dt_main.grad == 0) & (dt_main.active == 'NO')].last_acad_condition\
    .isna().sum() #OJO sigue sin sumar 5536
    
## CASOS ESTUDIANTES GRADUADOS NO ACTIVOS -------------------------------------
## Revisando ultima condicion academica 8492 casos graduados no activos
dt_main[(dt_main.grad == 1) & (dt_main.active == 'NO')].last_acad_condition\
    .value_counts() #no hay nan, pero si 18 casos con cond acad no normal
    
## CASOS ESTUDIANTES ACTIVOS NO GRADUADOS -------------------------------------
## Revisando ultima condicion academica 5055 casos activos no graduados
dt_main[(dt_main.grad == 0) & (dt_main.active == 'SI')].last_acad_condition\
    .value_counts() #no suman 5055
### Contando nas    
dt_main[(dt_main.grad == 0) & (dt_main.active == 'SI')].last_acad_condition\
    .isna().sum() #con esto suman 5055
### Cruzando estos acad_cond finales con el year_in 
tb = pd.crosstab(dt_main[(dt_main.grad == 0) & (dt_main.active == 'SI')]\
                 .last_acad_condition, dt_main[(dt_main.grad == 0) \
                                               & (dt_main.active == 'SI')]\
                     .year_in)
### Cruzando estos acad_cond finales con last year enrolled
tb = pd.crosstab(dt_main[(dt_main.grad == 0) & (dt_main.active == 'SI')]\
                 .last_acad_condition, dt_main[(dt_main.grad == 0) \
                                               & (dt_main.active == 'SI')]\
                     .last_termid)
    
""" EL si/no en la variable active parece ser coherente con si el estudiante se
graduo o no, siempre es no cuando el estudiante presenta una fecha de graduaci-
on. La ultima condicion academica de los estudiantes no parece siempre ser co-
herente con si el estudiante esta o no activo"""

## Cruzando datos last_termid con active 
tb = pd.crosstab(dt_main.last_termid, dt_main.active)
### Para estudiantes no graduados
tb = pd.crosstab(dt_main[dt_main.grad == 0].last_termid, 
                 dt_main[dt_main.grad == 0].active)
### Para estudiantes graduados
tb = pd.crosstab(dt_main[dt_main.grad == 1].last_termid, 
                 dt_main[dt_main.grad == 1].active)

""" Los estudiantes que tengan fecha de graduacion seran categorizados como 
graduados. Los estudiantes que esten actualmente inscritos (trim 2022-3 y 
2022-4) seran los considerados como activos. Los estudiantes que no se hayan 
graduado ni esten actualmente activos seran considerados como abandono. Lo que 
queda es decidir como serparar los abandonos voluntarios de los no voluntarios.
"""

## Generating var containing quant of terms not enrolled until actuality 
dt_main['trim_unenroll_quant'] = (2022 - dt_main['last_year_enroll']) * 4 \
                                 + (4 - dt_main['last_trim_enroll']) - 1 
                                 #-1 para solo considerar hasta 2022_3
### Checking var generated
temp1 = dt_main['trim_unenroll_quant'].value_counts()
### Crossing last_acad_condition vs. trim_unenroll_quant
tb = pd.crosstab(dt_main['last_acad_condition'], 
                 dt_main['trim_unenroll_quant'])

## Generating var containing quant trim out between terms 
### Max amount of terms enrolled by a student in the sample
trim_quant_max = dt_main.trim_quant.max()
for i in range(trim_quant_max-1):
    ### Creating empty vars to contain quantities for each term
    dt_main['trim'+str(i+2).zfill(2)+'_prev_trims_out'] = pd.Series(dtype=int)
    ### Year current term for specific student (starting at second term)
    year = dt_main['trim'+str(i+2).zfill(2)+'_year']
    ### Year previous term for specific student
    prev_year = dt_main['trim'+str(i+1).zfill(2)+'_year']
    ### Trim current term for specific student (starting at second term)
    trim = dt_main['trim'+str(i+2).zfill(2)+'_trim']
    ### Trim previous term for specific student (starting at second term)
    prev_trim = dt_main['trim'+str(i+1).zfill(2)+'_trim']
    ### Conditions for filling vars with np.select
    conditions = [(year - prev_year == 0), (year - prev_year > 0)]
    ### Choices for filling vars with np.select 
    choices = [trim - prev_trim - 1,
               (year - prev_year - 1)*4 + (4 - prev_trim) + trim - 1]
    ### Filling vars with np.select     
    dt_main['trim'+str(i+2).zfill(2)+'_prev_trims_out'] = np.select(conditions, 
                                                                   choices, 
                                                                default=np.nan)

### Checking results 
temp = dt_main[[x for x in dt_main.columns if '_prev_trims_out' in x]]

temp1 = dt_main[['year_in','trim_in']+[x for x in dt_main.columns if 'termid'\
                                       in x and '_id' not in x]]

for var in [x for x in dt_main.columns if '_prev_trims_out' in x]:
    print(dt_main[var].value_counts())

### Getting values into a list for accounting frecuency
prev_trims_out_vals = dt_main[[x for x in dt_main.columns if '_prev_trims_out'\
                               in x]].values.tolist()

prev_trims_out_val_list = [item for sublist in prev_trims_out_vals for item in\
                           sublist] 

# Frecuencia total de quantities terms out con posterior reingreso
# OJO un estudiante puede presentar mas de un reingreso
# con igual o distinta cnatidad 
pd.Series(prev_trims_out_val_list).value_counts()

# Calculando suma de trimestres out por estudiante
## Sumando trimestres out pot estudiante
dt_main['prev_trims_out_sum'] = dt_main[[x for x in dt_main.columns \
                                         if '_prev_trims_out' in x]]\
                                .sum(axis=1)
## Calculando cuantos tienen al menos 1 reingreso (al menos un trimestre sin
## sin matricularse)
dt_main[dt_main.prev_trims_out_sum != 0].sum()

## Calculando numero de estudiantes que dejan de matricularse al menos un 
## trimestre, con posterior reingreso o no
unenroll_students_total = dt_main[(dt_main.prev_trims_out_sum > 0) | \
                                  (dt_main.trim_unenroll_quant > 0)].shape[0]
    
# Calculando cambios de carrera por trimestre
for i in range(trim_quant_max-1):
    ### Creating empty vars to contain quantities for each term
    dt_main['trim'+str(i+2).zfill(2)+'_program_change'] = pd.Series(dtype=int)
    ### Program current term (starting at second term)
    program = dt_main['trim'+str(i+2).zfill(2)+'_program']
    ### Program previous term
    prev_program = dt_main['trim'+str(i+1).zfill(2)+'_program']
    ### Conditions for filling vars with np.select
    conditions = [(program == prev_program), 
                  (program != prev_program) & (program == program)]
    ### Choices for filling vars with np.select 
    choices = [0, 1]
    ### Creating and filling vars with np.select   
    dt_main['trim'+str(i+2).zfill(2)+'_program_change'] = np.select(conditions, 
                                                                   choices, 
                                                                default=np.nan)

# Calculando total cambios de carrera hechos por estudiantes
dt_main['program_change_quant'] = dt_main[[x for x in dt_main.columns \
                                           if '_program_change' in x]]\
                                  .sum(axis=1)

# Agregando area academica a program_in y program provenientes de registro
## agregando area del programa 
dt_main = dt_main.merge(dt_program_area[['program', 'area_name']], how='left', 
                                  on='program')

dt_main = dt_main.merge(dt_program_area[['program', 'area_name']], how='left', 
                        left_on='program_in', right_on='program', 
                        suffixes=("","_in"))

dt_main.rename(columns={'area_name_in':'area_in_name'}, inplace=True)

# Recueprando last program y area academica de datos trimestrales
dt_main['last_program'] = dt_main[[x for x in dt_main.columns if \
                                          '_program' in x and 'm_' not in x]]\
    .apply(lambda x: list(x.dropna())[-1] if (len(list(x.dropna())) != 0)\
           else np.nan, axis=1)
        
dt_main['last_program_name'] = dt_main[[x for x in dt_main.columns if \
                                          '_program_name' in x]]\
    .apply(lambda x: list(x.dropna())[-1] if (len(list(x.dropna())) != 0)\
           else np.nan, axis=1)

dt_main['last_area'] = dt_main[[x for x in dt_main.columns if \
                                          '_area' in x and 'a_' not in x]]\
    .apply(lambda x: list(x.dropna())[-1] if (len(list(x.dropna())) != 0)\
           else np.nan, axis=1)

dt_main['last_area_name'] = dt_main[[x for x in dt_main.columns if \
                                          '_area_name' in x]]\
    .apply(lambda x: list(x.dropna())[-1] if (len(list(x.dropna())) != 0)\
           else np.nan, axis=1)

# Creating var to predict *****************************************************
## graduado/activo/abandono voluntario/abandono involuntario
"""supongo que los activos no entran en el modelo 🤔 ¿?"""
dt_main['y'] = pd.Series(dtype = str)

## Identificando estudiantes activos
dt_main.loc[dt_main['last_termid'].isin(['2022_3','2022_4']), 'y'] = "active"

## Identificando estudiantes graduados
dt_main.loc[dt_main['grad'] == 1, 'y'] = "graduate"

# Parentesis para seguir explorando *******************************************

### Crossing last_acad_cond vs. trim_unenroll_quant for not grad not active st.
tb = pd.crosstab(dt_main[dt_main.y.isna()]['last_acad_condition'], 
                 dt_main[dt_main.y.isna()]['trim_unenroll_quant'])

"""Estaba considerando etiquetar a los estudiantes no activos y no graduados 
que tuvieran un trimestre o mas sin inscribirse y que no sea abandono 
involuntario como abandono voluntario, pero si pasa con relativa frecuencia 
que los estudiantes dejan de inscribirse en 1 trimestre y retoman el siguiente,
haria mas sentido tomar 2 o 4 trimestres como treshold"""

# Estudiantes no graduados last_termid vs. trim_unenroll_quant
tb = pd.crosstab(dt_main[dt_main.grad == 0].last_termid, 
                 dt_main[dt_main.grad == 0].trim_unenroll_quant)

# Estimando frecuencia relativa en que estudiantes reingresan
## Funchion que me calcula la probabilidad de reingreso given number trims out
def probability_reenroll(trim_out_quant):
    tp = [x for x in prev_trims_out_vals if trim_out_quant in x]
    return len(tp)/unenroll_students_total*100#dt_main.shape[0]*100

## DataFrame para capturar probabilidades correspondientes a no. trims out
reenroll_probabilities = pd.DataFrame(columns=['trim_out_quant', 
                                               'prob_reenroll'])
## Filling DataFrame con probabilidades
for i in range(int(dt_main.trim_unenroll_quant.max())):
    reenroll_probabilities.at[i, 'trim_out_quant'] = i+1
    reenroll_probabilities.at[i, 'prob_reenroll'] = probability_reenroll(i+1)
## Plotting probabilities 
reenroll_probabilities.plot(y = 'prob_reenroll', x = 'trim_out_quant', 
                            kind='scatter', grid=True)

""" Efectivamente es relativamente alta la probabilidad de que un estudiante 
que se deje de inscribir un trimestre vuelva a inscribirse (~13%). A partir del 
trimestre 2 esta probabilidad baja a aproximadamente 4% y a partir del 4to 
trimestre la probabilidad esta por debajo del 1%. Se considerara como abandono 
cuando el estudiante tenga 2 o mas trimestres sin matricularse. Hace sentido 
incluir a los estudiantes que no se matriculan 2 trimestres porque se estima 
que solo el 4% de ellos volveran a matricularse, igual para 3 trimestres que la
probabilidad baja a 1.5% y a partir de 4 trimestres baja a menos de 1%."""

""" Originalmente calcule la probabilidad de que un estudiante volviera a 
matricularse luego de tener x trimestres no inscrito como la cantidad de 
estudiantes que duran x trimestres consecutivos sin inscribirse y luego 
reingresan entre el total de estudiantes en la muestra. Cambie esta estimacion 
por la cantidad de estudiantes que duran x trimestres consecutivos sin 
inscribirse y reingresas entre la cantidad total de estudiantes que dejan de 
matricularse al menos un trimestre (con reingreso o sin reingreso, graduados
o no graduados. En este escenario aumentan un poco las probabilidades pero 
continuan siendo similares. La probabilidad de que un estudiante que tenga 1
trimestre sin matricularse reingrese es de ~18%; que tenga 2 trimestres sin 
matricularse y reingrese es ~5%; que tenga 3 trimestres sin matricularse y 
reingrese es de ~2%; y que tenga 4 trimestres sin matricularse y reingrese 
sigue un poco por debajo de 1%. Continuamos con la misma estrategia, 
consideramos abandono a los estudiantes que no se matriculen 2 trimestres
coonsecutivos."""

""" OJO la estimacion de estas probabilidades se puede mejorar restando los 
estudiantes que reingresan luego de x-1 trimestres consecutivos no matriculados
pero para los fines esta estimacion es suficiente, no variarian mucho los 
porcentajes porque no son muchos los estudiantes que retornan, estimo que max 
30 estudiantes para el caso con mayor tasa de retorno que es cuando x = 1."""

# Revisando last_acad_condition para decidir que considerar como abandono 
# involuntario y que como abandono voluntario cuando no esta activo ni graduado
dt_main[dt_main.y.isnull()].last_acad_condition.value_counts()
assert dt_main[dt_main.y.isnull()].last_acad_condition.value_counts().sum()\
       + dt_main[dt_main.y.isnull()].last_acad_condition.isnull().sum()\
       == dt_main.y.isnull().sum()
       
## Revisando los casos en que last_acad_condition is not null
tb = pd.crosstab(dt_main[dt_main.y.isnull()].last_acad_condition, 
                 dt_main[dt_main.y.isnull()].trim_unenroll_quant)
""" 
Reglas para casos que tienen last_acad_condition:
    - Si es separado o progreso academico no satisfactorio, es involuntario
    - Si es suspendido de algun tipo y tiene mas de 4 trimestres sin 
    matricularse, es abandono voluntario 
    - Si es suspendido de algun tipo y tiene menos de 4 trimestres sin 
    matricularse, lo registraremos como suspencion por ahora
    - En todos los demas casos (normal, condicion observada, en proceso, 
    prueba academica), es abandono voluntario si tiene 2 trimestres o mas sin
    matricularse.
"""

## Revisando casos en que last_acad_condition is null 
temp = dt_main[(dt_main.y.isnull()) & (dt_main.last_acad_condition.isnull())]
temp1 = temp.trim_unenroll_quant.value_counts()
""" Todos estos casos (que no estan graduados, no estan activos, y no tienen
last_acad_condition, no tuvieron mas de un trimestre activos en INTEC. Me 
parece que hace sentido aplicarles la regla de considerarlos como abandono
voluntario si tienen 2 o mas trimestres sin matricularse."""

# Creating var to predict *****************************************************

## Identificando estudiantes abandono involuntario
acad_cond_aband_invol = [x for x in set(dt_main.last_acad_condition.dropna())\
                         if 'SEPARADO' in x or 'NO SATISF' in x]
dt_main.loc[(dt_main.y.isnull()) & (dt_main['last_acad_condition']\
        .isin(acad_cond_aband_invol)), 'y'] = "involuntary desertion"
### Revisando unenroll_quant para involunatry desertion 
dt_main[dt_main.y =="involuntary desertion"].trim_unenroll_quant.value_counts()

## Identificando estudiantes abandono voluntario 

### Suspendidos
acad_cond_suspen = [x for x in set(dt_main.last_acad_condition.dropna()) \
                   if 'SUSPEN' in x]
#### Suspendido y mas de 4 trimestres sin matricularse
dt_main.loc[(dt_main['y'].isnull()) \
            & (dt_main['last_acad_condition'].isin(acad_cond_suspen)) \
            & (dt_main['trim_unenroll_quant'] >= 4), 
            'y'] = "voluntary desertion"
#### Suspendido y menos de 4 trimestres sin matricularse
dt_main.loc[(dt_main['y'].isnull()) \
            & (dt_main['last_acad_condition'].isin(acad_cond_suspen)) \
            & (dt_main['trim_unenroll_quant'] < 4), 
            'y'] = "suspended"
assert "suspended" not in set(dt_main.y.dropna()) #este dataset no tiene estos

### Los demas estados
acad_cond_other = [x for x in set(dt_main.last_acad_condition.dropna()) \
                   if 'SUSPEN' not in x and 'SEPARADO' not in x \
                       and 'NO SATISF' not in x]
dt_main.loc[(dt_main['y'].isnull()) \
            & (dt_main['last_acad_condition'].isin(acad_cond_other)) \
            & (dt_main['trim_unenroll_quant'] >= 2), 
            'y'] = "voluntary desertion"
    
### Casos en que last_acad_condition is null 
dt_main.loc[(dt_main['y'].isnull()) \
            & (dt_main['last_acad_condition'].isna()) \
            & (dt_main['trim_unenroll_quant'] >= 2), 
            'y'] = "voluntary desertion"               
               
## Revisando resultados
### Revisando que todos tengan un valor asignado o sean nan
assert dt_main.y.value_counts().sum() + dt_main.y.isna().sum() \
    == dt_main.shape[0]                                             

### Revisando que todos los nan sean estudiantes que tiene 1 trim sin matricula
dt_main[dt_main.y.isna()].trim_unenroll_quant.value_counts() 
assert dt_main[dt_main.y.isna()].trim_unenroll_quant.value_counts().keys() == 1


#*****************************************************************************#
#******************************** Output *************************************#
#*****************************************************************************#

# Ordenando columnas segun nombre 
#dt_main = dt_main.reindex(sorted(dt_main), axis=1).reset_index() #mejor no

# Saving final dataset 
dt_main.to_csv(path+"02_prep/dt_main_pii.csv")



"""
####RECYCLING CODE

dt_names = {'dt_registro_est':'programas_alumnos', 'dt_programas':'programas', 
            'dt_gpa_trim': 'GPA_por_trimestres', 
            'dt_evol_acad': 'evolucion_academica_alumnos', 
            'dt_calif': 'calificaciones', 'dt_docentes': 'docentes', 
            'dt_docentes_adic': 'docentes_adicionales', 
            'dt_eval': 'evaluaciones', 
            'dt_eval_est': 'evaluaciones_completas_estudiantes', 
            'dt_autoeval_doc': 'autoevaluaciones_completas_docentes', 
            'dt_esc_eval': 'escala_evaluacion_por_trimestre', 
            'dt_prueba_adm': 'pruebas_admisiones', 
            'dt_becas': 'becas_patrocinios', 
            'dt_biograf_est': 'datos_biograficos_estudiantes'}

#agrupando vars de registro_est para revisar las combinaciones de G1 con los demas niveles
temp5 = dt['Programas Alumnos'][dt['Programas Alumnos']['AÑO INGRESO']>=2008]}
        [['ID ESTUDIANTE', 'NIVEL','PROGRAMA','CAMBIO PROGRAMA?']]}
        .groupby('ID ESTUDIANTE')['NIVEL','PROGRAMA','CAMBIO PROGRAMA?']}
        .agg({'NIVEL':set,'PROGRAMA':[list,}
        lambda x: len(list(x))-len(set(x))], 'CAMBIO PROGRAMA?':list})
                                      
# ronda 1, sin filtros
#dt_registro_est['program_type'] = dt_registro_est['nivel'].str[0] #grado o postgrado
#dt_registro_est['program_num'] = dt_registro_est['nivel'].str[1] #numero del programa, 1er 2do grado/postgrado
#dt_registro_est['cambio_programa'] = 0
#dt_registro_est.loc[dt_registro_est['CAMBIO PROGRAMA?'] == 'SI',
#                   'cambio_programa'] = 1#si ha cambiado o no de programa/carrera en el current program                                      


### Filling vars with amount of terms unenrolled if any
### Going through each observation
for i in range(dt_main.shape[0]):
    #Identifying student id
    idx = dt_main.iloc[i]['id']
    ### Going through each term group 
    for j in range(dt_main.iloc[i]['trim_quant']-1): 
        ### Year current term for specific student (starting at second term)
        year = dt_main[dt_main.id == idx]['trim'+str(j+2).zfill(2)+'_year']
        ### Year previous term for specific student
        prev_year = dt_main[dt_main.id ==idx]['trim'+str(j+1).zfill(2)+'_year']
        trim = dt_main[dt_main.id == idx]['trim'+str(j+2).zfill(2)+'_trim']
        prev_trim = dt_main[dt_main.id ==idx]['trim'+str(j+1).zfill(2)+'_trim']
        conditions = [(year - prev_year == 0), (year - prev_year > 0)]
        choices = [trim - prev_trim - 1,
                   (year - prev_year - 1)*4 + (4 - prev_trim) + trim - 1]
            
        dt_main.at[i,'trim'+str(j+1).zfill(2)+'_prev_trims_out'] = \
            np.select(conditions, choices, default=np.nan)
            
# Estimando frecuencia relativa en que estudiantes reingresan
## Luego de 1 trimestre 
temp2 = [x for x in prev_trims_out_vals if 1 in x]
len(temp2)/dt_main.shape[0]*100 #2531 -> 13.26%
## Luego de 2 trimestres 
temp2 = [x for x in prev_trims_out_vals if 2 in x]
len(temp2)/dt_main.shape[0]*100 #738 -> 3.87%
## Luego de 4 trimestres 
temp2 = [x for x in prev_trims_out_vals if 4 in x]
len(temp2)/dt_main.shape[0]*100 #138 -> 0.72%
"""