#variable cualitativa nominal nombre de las carreras

from hmac import digest
from turtle import pd

import numpy as np


frec_cualita = digest["carrera"].value_counts().reset_index() # type: ignore

#renombrar columnas
frec_cualita.columns = ["carrera","fi"]

#frecuencia relativa
frec_cualita["hi"] = frec_cualita["fi"] / len(digest) # type: ignore

#frecuencia relativa porcentual
frec_cualita["hip"] = frec_cualita["hi"] * 100

#frecuencia acumulada
frec_cualita["Fi"] = frec_cualita["fi"].cumsum()

#frecuencia relativa acumulada
frec_cualita["Hi"] = frec_cualita["hi"].cumsum()

print("TABLA DE FRECUENCIAS: CARRERAS")

print(frec_cualita)

# 1. Conteo de frecuencias para la variable discreta 'materias_aprobadas'

tabla_discreta = digest["materias_aprobadas"].value_counts().sort_index().reset_index() # type: ignore

# 2. Renombramos las columnas para que coincidan con tu Guía Metodológica

tabla_discreta.columns = ["Materias_X", "fi"]

# cálculo de Frecuencia relativa

tabla_discreta["hi"] = tabla_discreta["fi"] / len(digest) # type: ignore

# 3. Cálculo de Frecuencias Acumuladas (Fi)
# El método .cumsum() realiza la suma sucesiva que explicaste en el PDF

tabla_discreta["Fi"] = tabla_discreta["fi"].cumsum()

tabla_discreta["Hi"] = tabla_discreta["hi"].cumsum()

tabla_discreta["hip"] = tabla_discreta["hi"] * 100

print("TABLA DE FRECUENCIAS: MATERIAS APROBADAS")

print(tabla_discreta)

#TABLA DE FRECUENCIAS PARA LA VARIABLE CUANTITATIVA DISCRETA EDAD

n = len(digest) # type: ignore

rango = digest["edad"].max() - digest["edad"].min() # type: ignore

# Aplicación de la Regla de Sturges (Rigor académico)
#ceil redondea hacia arriba

k = int(np.ceil(1 + 3.322 * np.log10(n)))

amplitud = rango / k

print(f"n: {n}, Rango: {rango}, Intervalos (k): {k}, Amplitud: {amplitud}")

#divide el rango en k partes tipo array

cortes = np.arange( digest["edad"].min(),digest["edad"].max() + amplitud,amplitud) # type: ignore

#Definicion de intervalos

#include_lowest=True incluye el primer intervalo

#right=False indica que el intervalo es [a,b)

digest["intervalos"] = pd.cut( # type: ignore
    digest["edad"], # type: ignore
    bins=cortes,
    include_lowest=True,
    right=False
)

#a partir de los intervalos se cuentan las frecuencias

tabla_agrupada = (
    digest["intervalos"] # type: ignore
    .value_counts()
    .sort_index()
    .reset_index()
)

tabla_agrupada.columns = ["intervalos","fi"]

#nos permite calcular la media de los intervalos

#lambda se usa para aplicar una funcion a cada elemento de la columna

tabla_agrupada["marca_clase"] = (
    tabla_agrupada["intervalos"]
    .apply(lambda x: x.mid)
)

#frecuencia relativa

tabla_agrupada["hi"] = (
    tabla_agrupada["fi"] / len(digest) # type: ignore
)

#frecuencia relativa porcentual

tabla_agrupada["hip"] = (
    tabla_agrupada["hi"] * 100
)

#frecuencia acumulada

tabla_agrupada["Fi"] = (
    tabla_agrupada["fi"].cumsum()
)

#frecuencia relativa acumulada

tabla_agrupada["Hi"] = (
    tabla_agrupada["hi"].cumsum()
)

print(tabla_agrupada)