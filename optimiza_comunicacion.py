"""
Clase encargada de optimizar el proceso de comunicación
Utilizará la implementación del algoritmo de  Naive Bayes de sklearn.
Se trata de un método supervisado que se basa en la aplicación del teorema de Bayes con
la asunción de la independencia entre las características (ngenuo)
"""
import logging
import uuid
import pandas as pd
import constantes as c
import numpy as np
import matplotlib.pyplot as plt

from pandas import DataFrame
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from decission import clasifica_decission_tree
from collections import defaultdict
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from copy import copy

# class Optimizador(metaclass=ABCMeta):
#     @classmethod
#     def predice(self,valores):
#     @classat
#     def


class OptimizadorNB:
    _nombreFichero: str
    _dataset_original: pd.DataFrame = None
    _dataset_entrenamiento: pd.DataFrame = None
    _datasetDecision: pd.DataFrame = None
    _dataset_resultados: pd.DataFrame = None
    _mapasCodificacion: defaultdict = defaultdict(LabelEncoder)
    _naive_bayes: GaussianNB = None
    _encoders: defaultdict = defaultdict(LabelEncoder)
    _uuids: pd.DataFrame = None
    _matriz_confusion = None
    _precision = 0.0
    _caracteristicas_entrenamiento = None
    _clasificacion_entrenamiento = None
    _hay_que_entrenar = None

    def __init__(self):
        nombreFichero = 'Clasificadores-Corto.xlsx'
        self._hay_que_entrenar = True

    @property
    def caractaristicas_entrenamiento(self):
        return self._caracteristicas_entrenamiento

    @property
    def clasificacion_entrenamiento(self):
        return self._clasificacion_entrenamiento

    @property
    def nombrefichero(self):
        return self._nombreFichero

    @nombrefichero.setter
    def nombrefichero(self, value):
        self._nombreFichero = value

    @property
    def dataset_decision(self):
        """
        El dataset clasificado mediante el árbol de decisión
        :return:
        """
        return self._datasetDecision

    @property
    def dataset_entrenamiento(self):
        """
        El dataset de entrenamiento con los datos válidos y verificados en la comunicación
        :return:
        """
        return self.__inversa_caracteristicas(self._dataset_entrenamiento)

    @property
    def dataset_resultados(self):
        """
        El dataset con todos los resultados pendientes de confirmación
        :return:
        """
        return self.__inversa_caracteristicas(self._dataset_resultados)

    @property
    def matriz_confusion(self):
        """
        La matriz de confusion calculada
        :return:
        """
        if self._hay_que_entrenar:
            self.__entrena_naive_bayes(self._dataset_entrenamiento)
        return self._matriz_confusion

    @property
    def precision(self):
        if self._hay_que_entrenar:
            self.__entrena_naive_bayes(self._dataset_entrenamiento)
        return self._precision

    @property
    def mapasCodificacion(self):
        return self._mapasCodificacion

    def __carga_dataset(self, nombre=None):
        """
        Realiza la carga del dataFrame Original para inciar con algo y posteriormente aplicacr un decission tree
        :param nombre:  nombre del fichero a cargar
        :return:
        """
        if nombre is not None:
            self._nombreFichero = nombre
        self._dataset_original = pd.read_excel(self._nombreFichero)

    def __prepara_dataset(self, dataset: DataFrame):
        """
        Prepara el conjunto de datos para poder utilizar Naive Bayes
        :param dataset: Dataframe a codificar
        :return: dataframe codificado + transformaciones realizadas
        """
        # hacemos una copia del DataSet
        dataset_codificado = dataset.copy(deep=True)
        mapas = defaultdict(LabelEncoder)
        for i in dataset_codificado.columns:
            # transformamos sólo los datos que no son numéricos
            if not (dataset_codificado[i].dtype == np.float64 or dataset_codificado[i].dtype == np.int64):
                self.__transforma_categoricas(dataset_codificado, i, i.lower(), mapas)
            else:
                dataset_codificado.rename(columns={i: i.lower()}, inplace=True)
        return [dataset_codificado, mapas]

    def __prepara_caracteristicas(self, dataset: DataFrame):

        for i in dataset.columns:
            if not (dataset[i].dtype == np.float64 or dataset[i].dtype == np.int64):
                # gle: LabelEncoder
                gle = self._encoders.get(i.lower())
                # print(gle.classes_)
                etiquetas = gle.transform(dataset[i])
                dataset[i] = etiquetas
                # dataset.pop(i)
            dataset.rename(columns={i: i.lower()}, inplace=True)

        return None

    def __inversa_caracteristicas(self, dataset: DataFrame):
        resultado: pd.DataFrame = None
        if not dataset is None:
            resultado = dataset.copy(deep=True)
            for i in resultado.columns:
                if not self._encoders.get(i.lower()) is None:
                    # gle: LabelEncoder
                    gle = copy(self._encoders.get(i.lower()))
                    # print('Etiquetando  {0}'.format(i))
                    etiquetas = gle.inverse_transform(resultado[i])
                    resultado[i] = etiquetas
                    # resultado.pop(i)
            # dataset.rename(columns={i: i.upper()}, inplace=True)
            return resultado
        else:
            return None

    def __transforma_categoricas(self, dataset: DataFrame, columna_origen: str,
                                 columna_destino: str, mapas: dict) -> DataFrame:
        """
        Transforma los valores categóricos de una columna en valores numéricos
        :param dataset: el conjunto de todos los datos
        :param columna_origen: nombre de la columna de la que partimos
        :param columna_destino: nombre de la columna en la que lo vamos a dejar
        :param mapas: mapas de codificación para el uso de los valores
        :return:
        """
        gle = LabelEncoder()
        etiquetas = gle.fit_transform(dataset[columna_origen])
        mapeos = {index: label for index, label in enumerate(gle.classes_)}
        dataset[columna_destino] = etiquetas
        dataset.pop(columna_origen)
        mapas[columna_destino] = mapeos
        self._encoders[columna_destino] = gle
        return dataset

    def __entrena_naive_bayes(self, dataset: pd.DataFrame):
        """
        Realiza el entrenamiento de naive bayes además de la validación
        Actualiza los valores de la matriz de confusión y de precisión, que permiten analizar la evolución
        en cuanto a la calidad del algoritmo entrenado
        :param dataset: conjunto de datos para hacer el entrenamiento
        """
        self._hay_que_entrenar = False
        self._dataset_entrenamiento = dataset.copy(deep=True)
        if 'uuid' not in self._dataset_entrenamiento.columns:
            self._dataset_entrenamiento.insert(0, "uuid", [str(uuid.uuid4()) for _ in range(len(dataset.index))])
        array = dataset

        if 'uuid' in dataset.columns:
            caracteristicas = array.iloc[:, 1:len(dataset.columns) - 1].values
        else:
            caracteristicas = array.iloc[:, 0:len(dataset.columns) - 1].values
        clasificacion = array.iloc[:, - 1].values
        tamanio_validacion = 0.23
        semilla_random = 7
        self._caracteristicas_entrenamiento, caracteristicas_validacion,  \
            self._clasificacion_entrenamiento, clasificacion_validacion = \
            model_selection.train_test_split(caracteristicas,
                                             clasificacion, test_size=tamanio_validacion,
                                             random_state=semilla_random)
        self._naive_bayes = GaussianNB()

        self._naive_bayes.fit(self._caracteristicas_entrenamiento, self._clasificacion_entrenamiento)
        predicciones = self._naive_bayes.predict(caracteristicas_validacion)
        self._matriz_confusion = confusion_matrix(clasificacion_validacion, predicciones)
        self._precision = accuracy_score(clasificacion_validacion, predicciones)

    def predice(self, caracteristicas: DataFrame) -> DataFrame:
        """
        Predice el tipo de comunicación para el conjunto de características pasado
        :rtype: DataFrame
        :param caracteristicas: conjunto de valors para predecir
        :return: El conjunto de predicciones
        """
        logging.info('Realizando la predicción con el fichero {0}'.format(self.nombrefichero))

        # si no hay datos entrenados, los tendremos que entrenar
        if self._naive_bayes is None:
            logging.info('No está entrenado el dataSet, procedemos a entrenarlo con los datos %s', self.nombrefichero)
            self.__carga_dataset()
            # Clasificamos con un arbol de decisión
            self._datasetDecision = clasifica_decission_tree(self._dataset_original)
            dataset_codificado, self._mapasCodificacion = self.__prepara_dataset(self._datasetDecision)
            self.__entrena_naive_bayes(dataset_codificado)
        if self._hay_que_entrenar:
            self.__entrena_naive_bayes(self._dataset_entrenamiento)
        # codificamos las características
        self.__prepara_caracteristicas(caracteristicas)

        resultado: DataFrame
        caracteristicas = caracteristicas[['edad','miembros','sexo','estado_civil','ingresos','educacion',
                                          'situacion_laboral']]
        array = caracteristicas.iloc[:, 0:len(caracteristicas.columns)].values
        resultado = self._naive_bayes.predict(array)
        caracteristicas.insert(0, "uuid", [str(uuid.uuid4()) for _ in range(len(caracteristicas.index))])
        caracteristicas['tipocomunicacion'] = resultado
        # Lo metemos en la lista de resultados enviados, para esperar la confirmación
        if self._dataset_resultados is None:
            self._dataset_resultados = caracteristicas.copy(deep=True)
        else:
            self._dataset_resultados =  self._dataset_resultados.append(caracteristicas, sort=False, ignore_index=True)
        # Realizamos la codificación inversa
        return self.__inversa_caracteristicas(caracteristicas)

    def set_envio_correcto(self, id_envio: object) -> str:
        """
        Incorpora un id correcto en la base de conocimiento
        :rtype: str
        :param id_envio: uuiid del envío
        :return: ok o ko en caso si se ha podido almacenar el envío
        """
        resultado = 'ko'
        if self._dataset_resultados is not None:
            fila = self._dataset_resultados.loc[self._dataset_resultados['uuid'] == id_envio]
            if len(fila) > 0:
                self._dataset_entrenamiento = self._dataset_entrenamiento.append(fila, ignore_index=True)
                self._dataset_resultados = self._dataset_resultados[self._dataset_resultados.uuid != id_envio]
                resultado = 'ok'
                # procedemos a hacer un partial fit
                array = fila
                caracteristicas = array.iloc[:, 1:len(fila.columns) - 1].values
                clasificacion = array.iloc[:, len(fila.columns) - 1].values
                self._naive_bayes.partial_fit(caracteristicas, clasificacion)
                self._hay_que_entrenar = True
        return resultado

    def _cambia_comunicacion(self, fila):

        dict_comunicacion = self._mapasCodificacion.get(c.COMUNICACION)
        dict_invertido = {}
        tipo_comunicacion = ''
        for clave, valor in dict_comunicacion.items():
            dict_invertido[valor] = clave
            if clave == fila[c.COMUNICACION].iloc[0]:
                tipo_comunicacion = valor

        if tipo_comunicacion == c.COMUNICACION_WHATS:
            fila[c.COMUNICACION] = dict_invertido.get(c.COMUNICACION_EMAIL)
        elif tipo_comunicacion == c.COMUNICACION_EMAIL:
            fila[c.COMUNICACION] = dict_invertido.get(c.COMUNICACION_SMS)
        elif tipo_comunicacion == c.COMUNICACION_SMS:
            fila[c.COMUNICACION] = dict_invertido.get(c.COMUNICACION_POSTAL)
        elif tipo_comunicacion == c.COMUNICACION_POSTAL:
            fila[c.COMUNICACION] = dict_invertido.get(c.COMUNICACION_WHATS)

    def set_envio_erroneo(self, id_envio: str) -> str:
        """
        descarta el envío por mal clasificado
        :type id_envio: str
        :param id_envio: uuid del envío
        :rtype: str
        :return: ok o ko en caso de que se haya podido descartar el envio
        """
        resultado = 'ko'
        if self._dataset_resultados is not None:
            fila = self._dataset_resultados.loc[self._dataset_resultados['uuid'] == id_envio]
            if len(fila) > 0:
                self._dataset_resultados = self._dataset_resultados[self._dataset_resultados.uuid != id_envio]
                resultado = 'ok'
        if self._dataset_entrenamiento is not None:
            fila = self._dataset_entrenamiento.loc[self._dataset_entrenamiento['uuid'] == id_envio]
            if len(fila) > 0:
                self._cambia_comunicacion(fila)
                self._dataset_entrenamiento.loc[self._dataset_entrenamiento['uuid'] == id_envio] = fila
                # self._dataset_entrenamiento = self._dataset_entrenamiento.loc[self._dataset_entrenamiento.uuid != id_envio]
                #self.__entrena_naive_bayes(self._dataset_entrenamiento)
                self._hay_que_entrenar = True

                resultado = 'ok'
        return resultado
