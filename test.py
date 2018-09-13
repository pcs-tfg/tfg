"""
Módulo que se encarga e hacer las solicitudes de la optimizción
"""
import pandas as pd
from pandas import DataFrame
from sklearn.naive_bayes import GaussianNB

import constantes as c
from Imagenes import plot_confusion_matrix


from optimiza_comunicacion import OptimizadorNB

clasificador: OptimizadorNB = OptimizadorNB()
clasificador.nombrefichero = 'Clasificadores-Corto.xlsx'


def get_prediccion():
    """
    simula una llamada a la predicción de los datos
    :return:
    """

    dat = pd.read_excel('APredecir.xlsx')

    clasificador.predice(dat)



def procesa_prediccion():
    # datoshombres = clasificador.dataset_resultados.loc[clasificador.dataset_resultados['sexo'] == 'Hombre']
    # datosmujeres = clasificador.dataset_resultados.loc[clasificador.dataset_resultados['sexo'] == 'Mujer']

    seguir = True
    while seguir:

        erroneos = clasificador.dataset_entrenamiento.loc[(clasificador.dataset_entrenamiento['sexo'] == 'Hombre')]
        mas65 = erroneos.loc[(erroneos['ingresos'] == 'Alto') & (erroneos['educacion'] == 'Master') & (erroneos['edad'] > 65) and erroneos[c.COMUNICACION] != c.COMUNICACION_WHATS]
        medianos = erroneos.loc[(erroneos['situacion_laboral'] == 'Empleado/a') & ((erroneos['ingresos'] == 'Medio') | (erroneos['ingresos'] == 'Alto')) & (erroneos['miembros'] <= 4) & (erroneos['edad'] <= 65) & (erroneos['edad'] > 25)]
        erroneos = clasificador.dataset_entrenamiento.loc[(clasificador.dataset_entrenamiento['sexo'] == 'Hombre') &
                                                          (clasificador.dataset_entrenamiento['edad'] <= 25)]
        erroneos= erroneos.append(mas65, sort=False, ignore_index=True)
        erroneos = erroneos.append(medianos, sort=False, ignore_index=True)
        seguir = False
        for index, row in erroneos.iterrows():
            if row[c.COMUNICACION] != c.COMUNICACION_WHATS:
                clasificador.set_envio_erroneo(row['uuid'])
                seguir = True



    print(clasificador.dataset_entrenamiento.loc[(clasificador.dataset_entrenamiento['sexo'] == 'Hombre') &
                                                 (clasificador.dataset_entrenamiento['edad'] < 25)])

    dataframe_prueba = clasificador.dataset_entrenamiento.loc[(clasificador.dataset_entrenamiento['sexo'] == 'Hombre') &
                                                 (clasificador.dataset_entrenamiento['edad'] < 25)]
    dataframe_prueba.pop('uuid')
    dataframe_prueba.pop(c.COMUNICACION)
    clasificador.predice(dataframe_prueba)
    print(clasificador.dataset_resultados.info())
    print(clasificador.dataset_entrenamiento.info())
    for index, row in clasificador.dataset_resultados.iterrows():
        if row['edad'] < 25 and row[c.COMUNICACION] == c.COMUNICACION_EMAIL:
            clasificador.set_envio_correcto(row['uuid'])
        else:
            if row['edad'] < 25:
                clasificador.set_envio_erroneo(row['uuid'])
    print(clasificador.dataset_entrenamiento.info())
    print('La precisión actual es de: {0}'.format(clasificador.precision))

    # print (datoshombres)
    # print(datosmujeres)




def main():
    get_prediccion()
    print (clasificador.dataset_entrenamiento.info())
    # Tenemos la predicción ya hecha, ahora "sólo" hay que procesarla
    print (clasificador.matriz_confusion)

    # plot_confusion_matrix(cm=clasificador.matriz_confusion, normalize=False,
    #                                   target_names=clasificador.mapasCodificacion.get(c.COMUNICACION).values(), title='Matriz de Confusión')
    # plot_confusion_matrix(cm=clasificador.matriz_confusion, normalize=True,
    #                                   target_names=clasificador.mapasCodificacion.get(c.COMUNICACION).values(), title='Matriz de Confusión')

    precision_anterior = clasificador.precision
    procesa_prediccion()
    precision_actual = clasificador.precision
    print('La precisión ha variado de: {0} a {1}'.format(precision_anterior,precision_actual))

    plot_confusion_matrix(cm=clasificador.matriz_confusion, normalize=False,
                          target_names=clasificador.mapasCodificacion.get(c.COMUNICACION).values(), title='Matriz de Confusión')
    plot_confusion_matrix(cm=clasificador.matriz_confusion, normalize=True,
                          target_names=clasificador.mapasCodificacion.get(c.COMUNICACION).values(), title='Matriz de Confusión')
    print (clasificador.matriz_confusion)
    # clasificador.plot_dibuja_clases()

main()

