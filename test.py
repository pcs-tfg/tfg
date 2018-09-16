"""
Módulo que se encarga e hacer las solicitudes de la optimizción
"""
from os import error

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

def get_prediccion2():
    """
    simula una llamada a la predicción de los datos
    :return:
    """

    dat = pd.read_excel('APredecir2.xlsx')

    clasificador.predice(dat)


def get_prediccion3():
    """
    simula una llamada a la predicción de los datos
    :return:
    """

    dat = pd.read_excel('APredecir3.xlsx')

    clasificador.predice(dat)
def get_prediccion4():
    """
    simula una llamada a la predicción de los datos
    :return:
    """

    dat = pd.read_excel('APredecir4.xlsx')

    clasificador.predice(dat)
def procesa_prediccion():

    seguir = True
    intento = 0
    modificados = 0
    precision_previa = clasificador.precision
    precision_nueva = 0
    plot_confusion_matrix(cm=clasificador.matriz_confusion, normalize=True,
                          target_names=clasificador.mapasCodificacion.get(c.COMUNICACION).values(),
                          title='Matriz de Confusión {0}'.format(intento))

    while seguir:
        intento += 1
        erroneos = clasificador.dataset_entrenamiento.loc[(clasificador.dataset_entrenamiento['sexo'] == 'Hombre')]
        mas65 = erroneos.loc[(erroneos['ingresos'] == 'Alto') & (erroneos['educacion'] == 'Master')
                             & (erroneos['edad'] > 65) & (erroneos[c.COMUNICACION] != c.COMUNICACION_WHATS)]
        medianos = erroneos.loc[(erroneos['situacion_laboral'] == 'Empleado/a')
                                & ((erroneos['ingresos'] == 'Medio') | (erroneos['ingresos'] == 'Alto'))
                                & (erroneos['miembros'] <= 4) & (erroneos['edad'] <= 65) & (erroneos['edad'] > 25)
                                & (erroneos[c.COMUNICACION] != c.COMUNICACION_WHATS)]
        erroneos = clasificador.dataset_entrenamiento.loc[(clasificador.dataset_entrenamiento['sexo'] == 'Hombre')
                                                          & (clasificador.dataset_entrenamiento['edad'] <= 25)
                                                          & (erroneos[c.COMUNICACION] != c.COMUNICACION_WHATS)]
        erroneos = erroneos.append(mas65, sort=False, ignore_index=True)
        erroneos = erroneos.append(medianos, sort=False, ignore_index=True)

        seguir = False
        print(erroneos.count())

        for index, row in erroneos.iterrows():
            if row[c.COMUNICACION] != c.COMUNICACION_WHATS:
                clasificador.set_envio_erroneo(row['uuid'])
                seguir = True
                modificados += 1
        if modificados != 0:
            precision_nueva = clasificador.precision
            print(
                f'Con {modificados} modificaciones la precisión del '
                f'algoritmo varía de {precision_previa} a {precision_nueva}')
            plot_confusion_matrix(cm=clasificador.matriz_confusion, normalize=True,
                                  target_names=clasificador.mapasCodificacion.get(c.COMUNICACION).values(),
                                  title='Matriz de Confusión {0}'.format(intento))
            precision_previa = precision_nueva
            modificados = 0


def procesa_prediccion_mujeres():

    prueba = clasificador.dataset_resultados.loc[(clasificador.dataset_resultados['sexo'] == 'Mujer')]
    print(prueba.count())
    positivo = 0
    negativo = 0
    for index, row in prueba.iterrows():
        if 25 < row['edad'] <= 45:
            if row[c.COMUNICACION] == c.COMUNICACION_EMAIL:
                clasificador.set_envio_correcto(row['uuid'])
                positivo += 1
            else:
                clasificador.set_envio_erroneo((row['uuid']))
        if 45 < row['edad'] <= 65:
            if row[c.COMUNICACION] == c.COMUNICACION_POSTAL:
                clasificador.set_envio_correcto(row['uuid'])
                positivo += 1
            else:
                clasificador.set_envio_erroneo((row['uuid']))
        if row['edad'] <= 25:
            if row[c.COMUNICACION] == c.COMUNICACION_WHATS:
                clasificador.set_envio_correcto(row['uuid'])
                positivo += 1
            else:
                clasificador.set_envio_erroneo((row['uuid']))
        if row['edad'] > 65:
            if row['ingresos'] == 'Muy Alto':
                if row[c.COMUNICACION] == c.COMUNICACION_POSTAL:
                    clasificador.set_envio_correcto(row['uuid'])
                    positivo += 1
                else:
                    clasificador.set_envio_erroneo((row['uuid']))
                    negativo += 1
            elif row['ingresos'] == 'Alto':
                if row['educacion'] == 'Master':
                    if row[c.COMUNICACION] == c.COMUNICACION_WHATS:
                        clasificador.set_envio_correcto(row['uuid'])
                        positivo += 1
                    else:
                        clasificador.set_envio_erroneo((row['uuid']))
                        negativo += 1
                else:
                    if row[c.COMUNICACION] == c.COMUNICACION_SMS:
                        clasificador.set_envio_correcto(row['uuid'])
                        positivo += 1
                    else:
                        clasificador.set_envio_erroneo((row['uuid']))
                        negativo += 1
            else:
                if row[c.COMUNICACION] == c.COMUNICACION_SMS:
                    clasificador.set_envio_correcto(row['uuid'])
                    positivo += 1
                else:
                    clasificador.set_envio_erroneo((row['uuid']))
                    negativo += 1


def procesa_prediccion_hombres():

    prueba = clasificador.dataset_resultados.loc[(clasificador.dataset_resultados['sexo'] == 'Hombre')]
    # prueba = prueba.loc[(prueba['edad'] > 25) & (prueba['edad'] <= 45)]
    print(prueba.count())
    positivo = 0
    negativo = 0
    for index,row in prueba.iterrows():
        if row['edad'] > 65:
            if row['ingresos'] == 'Muy Alto':
                if row[c.COMUNICACION] == c.COMUNICACION_EMAIL:
                    clasificador.set_envio_correcto(row['uuid'])
                    positivo += 1
                else:
                    clasificador.set_envio_erroneo((row['uuid']))
                    negativo += 1
            elif row['ingresos'] == 'Alto':
                if row['educacion'] == 'Master':
                    if row[c.COMUNICACION] == c.COMUNICACION_WHATS:
                        clasificador.set_envio_correcto(row['uuid'])
                        positivo += 1
                    else:
                        clasificador.set_envio_erroneo((row['uuid']))
                        negativo += 1
                else:
                    if row[c.COMUNICACION] == c.COMUNICACION_POSTAL:
                        clasificador.set_envio_correcto(row['uuid'])
                        positivo += 1
                    else:
                        clasificador.set_envio_erroneo((row['uuid']))
                        negativo += 1
            else:
                if row[c.COMUNICACION] == c.COMUNICACION_SMS:
                    clasificador.set_envio_correcto(row['uuid'])
                    positivo += 1
                else:
                    clasificador.set_envio_erroneo((row['uuid']))
                    negativo += 1
        if row['edad'] <= 25:
            if row[c.COMUNICACION] == c.COMUNICACION_WHATS:
                clasificador.set_envio_correcto(row['uuid'])
                positivo += 1
            else:
                clasificador.set_envio_erroneo((row['uuid']))
                negativo += 1
        if 25 < row['edad'] <= 65:
            if row ['situacion_laboral'] == 'Empleado/a':
                if row['ingresos'] == 'Bajo' or row['ingresos'] == 'Medio':
                    if row['miembros'] < 4:
                        if row[c.COMUNICACION] == c.COMUNICACION_WHATS:
                            clasificador.set_envio_correcto(row['uuid'])
                            positivo += 1
                        else:
                            clasificador.set_envio_erroneo((row['uuid']))
                            negativo += 1
                    else:
                        if row[c.COMUNICACION] == c.COMUNICACION_POSTAL:
                            clasificador.set_envio_correcto(row['uuid'])
                            positivo += 1
                        else:
                            clasificador.set_envio_erroneo((row['uuid']))
                            negativo += 1
                else:
                    if row[c.COMUNICACION] == c.COMUNICACION_POSTAL:
                        clasificador.set_envio_correcto(row['uuid'])
                        positivo += 1
                    else:
                        clasificador.set_envio_erroneo((row['uuid']))
                        negativo += 1
            else:
                if row['ingresos'] == 'Bajo' :
                    if row[c.COMUNICACION] == c.COMUNICACION_POSTAL:
                        clasificador.set_envio_correcto(row['uuid'])
                        positivo += 1
                    else:
                        clasificador.set_envio_erroneo((row['uuid']))
                        negativo += 1
                else:
                    if row[c.COMUNICACION] == c.COMUNICACION_SMS:
                        clasificador.set_envio_correcto(row['uuid'])
                        positivo += 1
                    else:
                        clasificador.set_envio_erroneo((row['uuid']))
                        negativo += 1



def main():
    # get_prediccion()
    # get_prediccion2()
    # get_prediccion3()
    get_prediccion4()
    print (clasificador.dataset_entrenamiento.info())
    # Tenemos la predicción ya hecha, ahora "sólo" hay que procesarla
    print (clasificador.matriz_confusion)

    # plot_confusion_matrix(cm=clasificador.matriz_confusion, normalize=False,
    #                                    target_names=clasificador.mapasCodificacion.get(c.COMUNICACION).values(), title='Matriz de Confusión')
    plot_confusion_matrix(cm=clasificador.matriz_confusion, normalize=True,
                                      target_names=clasificador.mapasCodificacion.get(c.COMUNICACION).values(), title='Matriz de Confusión')

    precision_anterior = clasificador.precision
    procesa_prediccion_hombres()
    procesa_prediccion_mujeres()
    # procesa_prediccion()

    precision_actual = clasificador.precision
    print('La precisión ha variado de: {0} a {1}'.format(precision_anterior,precision_actual))




    # plot_confusion_matrix(cm=clasificador.matriz_confusion, normalize=False,
    #                       target_names=clasificador.mapasCodificacion.get(c.COMUNICACION).values(), title='Matriz de Confusión')
    plot_confusion_matrix(cm=clasificador.matriz_confusion, normalize=True,
                          target_names=clasificador.mapasCodificacion.get(c.COMUNICACION).values(), title='Matriz de Confusión')
    print (clasificador.matriz_confusion)
    # clasificador.plot_dibuja_clases()

main()

