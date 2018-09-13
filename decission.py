"""
Modulo encargado de hacer el árbol de decisión
"""
import pandas as pd
import constantes as c


def clasifica_decission_tree(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la clasificación del dataset que recibe en función a las decisiones a prioristicas que hemos decidido tomar
    -----------
    :param dataset:     Los valores a clasificar
    :return:            Un nuevo dataset dataset con la columna Tipo de comunicación
    """
    tipo_comunicacion = c.COMUNICACION_POSTAL
    dataset = dataset.copy(deep=True)
    for index, row in dataset.iterrows():
        # print (row)

        if row['Sexo'] == 'Hombre':
            # hacemos una primerar clasidifación si edad > 60 correo postal
            if row['Edad'] > 65:
                if row['Ingresos'] == 'Muy Alto':
                        tipo_comunicacion = c.COMUNICACION_EMAIL
                elif row['Ingresos'] == 'Alto':
                    # preguntamos por la educación
                    if row['Educacion'] == 'Master':
                        tipo_comunicacion = c.COMUNICACION_WHATS
                    else:
                        tipo_comunicacion = c.COMUNICACION_POSTAL
                else:
                    tipo_comunicacion = c.COMUNICACION_WHATS
            # elif row['Edad'] <= 65 and row['Edad'] > 25:
            elif 25 > row['Edad'] >= 65:
                if row['Situacion_laboral'] == 'Desempleado/a':
                    if row['Ingresos'] == 'Bajo':
                        tipo_comunicacion= c.COMUNICACION_POSTAL
                    else:
                        tipo_comunicacion = c.COMUNICACION_SMS
                else:
                    if row['Ingresos'] == 'Bajo' or row['Ingresos'] == 'Medio':
                        if row['Miembros'] < 4:
                            tipo_comunicacion = c.COMUNICACION_WHATS
                        else:
                            tipo_comunicacion = c.COMUNICACION_POSTAL
                    else:
                        tipo_comunicacion = c.COMUNICACION_POSTAL

            elif row['Edad'] <= 25:
                tipo_comunicacion = c.COMUNICACION_WHATS
        else:
            if row['Edad'] > 65:
                if row['Ingresos'] == 'Muy Alto':
                        tipo_comunicacion = c.COMUNICACION_POSTAL
                elif row['Ingresos'] == 'Alto':
                    # preguntamos por la educación
                    if row['Educacion'] == 'Master':
                        tipo_comunicacion = c.COMUNICACION_WHATS
                    else:
                        tipo_comunicacion = c.COMUNICACION_SMS
                else:
                    tipo_comunicacion = c.COMUNICACION_SMS
            elif 45 < row['Edad'] <= 65:
                tipo_comunicacion = c.COMUNICACION_POSTAL
            elif 25 < row['Edad'] <= 45:
                tipo_comunicacion = c.COMUNICACION_EMAIL
            elif row['Edad'] <= 25:
                tipo_comunicacion = c.COMUNICACION_WHATS

        dataset.at[index, 'TipoComunicacion'] = tipo_comunicacion
    return dataset
