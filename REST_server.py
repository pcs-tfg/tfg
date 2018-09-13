#!flask/bin/python


from flask import Flask
from flask.json import jsonify
from flask import request
import pandas as pd
from pandas import DataFrame
from optimiza_comunicacion import OptimizadorNB


app = Flask(__name__)

clasificador = OptimizadorNB()
clasificador.nombrefichero = 'Clasificadores-Corto.xlsx'


@app.route('/predice', methods=['POST'])
def get_predicion():

    print('Vamos a predecir los datos')
    if not request.is_json:
        return 'No hay una llamada JSON'

    contenido = request.json['caracteristicas']

    dat = pd.DataFrame.from_dict(contenido)

    resultado = clasificador.predice(dat)
    if resultado is None:
        return jsonify('El sistema todavía no está entrenado, proceda a obtener el dataset de decisión '
                       'para comunicar el resultado de las comunicaciones')
    else:
        resultado = resultado.to_json(orient="records")
        return resultado


@app.route('/envio/correcto/<id_envio>', methods=['PUT'])
def envio_correcto(id_envio):
    """
    Indica que el envío ha sido correcto, de modo que se puede incluir en el conjunto de entrenamiento
    :param id_envio: el identificador del envío que está en resultados
    :return:
    """
    # clasificador.set_envio_correcto(id_envio)
    return jsonify('respuesta', clasificador.set_envio_correcto(id_envio))


@app.route('/envio/erroneo/<id_envio>', methods=['PUT'])
def envio_erroneo(id_envio):
    """
    Indica que el envío ha sido erroeno, de modo que lo descartamos
    :param id_envio: el identificador del envío que está en resultados
    :return:
    """
    # clasificador.set_envio_correcto(id_envio)
    return jsonify('respuesta', clasificador.set_envio_erroneo(id_envio))


@app.route('/resultados', methods=['GET'])
def get_resultado():
    """
    Devuelve los resultados acumulados de la predición
    :return: JSON con los resultados acumulados de la predicción
    """
    if clasificador.dataset_resultados is None:
        return jsonify('Todavía no existen resultados de predicción')
    else:
        return clasificador.dataset_resultados.to_json(orient='records')


@app.route('/entrenamiento', methods=['GET'])
def get_entrenamiento():
    """
    Devuelve el conjunto de entrenamiento del sistema
    :return: JSON con el dataset que se usa para el entrenamiento
    """
    if clasificador.dataset_entrenamiento is None:
        return jsonify('Todavía no existen resultados de predicción')
    else:
        return clasificador.dataset_entrenamiento.to_json(orient='records')


app.run(host='0.0.0.0', port=5000, debug=False)
