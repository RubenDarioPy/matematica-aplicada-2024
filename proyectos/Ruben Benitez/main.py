import time
import nltk
import re
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import skfuzzy.control as ctrl

from nltk import find
from nltk.sentiment import SentimentIntensityAnalyzer

def read_csv_to_dataframe(csv_file_path):
    """
    Lee un archivo CSV y lo devuelve como un DataFrame de pandas.
    
    Args:
        csv_file_path (str): Ruta del archivo CSV.
        
    Returns:
        pandas.DataFrame: DataFrame que contiene los datos del archivo CSV.
    """
    try:
        df = pd.read_csv(csv_file_path)
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
def preprocess_dataframe(df):
    """
    Preprocesa datos de texto en un DataFrame eliminando caracteres especiales y palabras de una sola letra.
    
    Args:
        df (pandas.DataFrame): DataFrame de entrada que contiene datos de texto.
        
    Returns:
        pandas.DataFrame: DataFrame preprocesado.
    """
    # Crear una copia para evitar modificar el DataFrame original
    processed_df = df.copy()
    
    # Función para limpiar el texto individual
    def clean_text(text):
        # Eliminar caracteres especiales
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Eliminar palabras de una sola letra
        text = ' '.join([word for word in text.split() if len(word) > 1])
        # Eliminar espacios en blanco adicionales
        text = ' '.join(text.split())
        # Diccionario de palabras abreviadas comunes y sus equivalentes formales
        shortened_words = {
            'u': 'you',
            'r': 'are',
            'y': 'why',
            'ur': 'your',
            'n': 'and',
            'w': 'with',
            'abt': 'about',
            'bc': 'because',
            'b4': 'before',
            'cuz': 'because',
            'k': 'okay',
            'thx': 'thanks',
            'pls': 'please',
            'plz': 'please',
            'tho': 'though',
            'tbh': 'to be honest',
            'idk': 'i do not know',
            'imo': 'in my opinion',
            'rn': 'right now',
            'gonna': 'going to',
            'wanna': 'want to',
            'gotta': 'got to',
            'lemme': 'let me',
            'gimme': 'give me'
        }
        
        # Reemplazar palabras abreviadas con sus equivalentes formales
        for shortened, formal in shortened_words.items():
            text = re.sub(r'\b' + shortened + r'\b', formal, text, flags=re.IGNORECASE)
        return text
    
    # Aplicar el preprocesamiento a la columna 'sentence'
    processed_df['sentence'] = processed_df['sentence'].apply(clean_text)
    
    return processed_df


def analyze_sentiment(df):
    senti = SentimentIntensityAnalyzer()
    # Se agrega la columna de los puntajes positivos
    df['positive_score'] = df['sentence'].apply(lambda x: senti.polarity_scores(x)['pos'])
    # Se agrega la columna de los puntajes positivos
    df['negative_score'] = df['sentence'].apply(lambda x: senti.polarity_scores(x)['neg'])
    return df

def fuzzyfy_sentiment(df):
    global_min_p = df['positive_score'].min()  # Mínimo global de los puntajes positivos
    global_max_p = df['positive_score'].max()  # Máximo global de los puntajes positivos
    global_min_n = df['negative_score'].min()  # Mínimo global de los puntajes negativos
    global_max_n = df['negative_score'].max()  # Máximo global de los puntajes negativos
    global_mid_p = (global_min_p + global_max_p) / 2 # Mid value de los puntajes positivos
    global_mid_n = (global_min_n + global_max_n) / 2 # Mid value de los puntajes negativos

    # Crea las variables difusas
    # Del minimo global al maximo global de los positivos
    positive = ctrl.Antecedent(np.arange(global_min_p, global_max_p + 0.1, 0.1), 'positive')
    # Del minimo global al maximo global de los negativos
    negative = ctrl.Antecedent(np.arange(global_min_n, global_max_n + 0.1, 0.1), 'negative')
    # Rango del output: 0 - 10
    output = ctrl.Consequent(np.arange(0, 10.1, 0.1), 'output')

    # Funciones de membresía para positive
    # Low: {min, min, mid}
    positive['low'] = fuzz.trimf(positive.universe, [global_min_p, global_min_p, global_mid_p])
    # Medium: {min, mid, max}
    positive['medium'] = fuzz.trimf(positive.universe, [global_min_p, global_mid_p, global_max_p])
    # High: {mid, max, max}
    positive['high'] = fuzz.trimf(positive.universe, [global_mid_p, global_max_p, global_max_p])

    # Funciones de membresía para negative
    # Low: {min, min, mid}
    negative['low'] = fuzz.trimf(negative.universe, [global_min_n, global_min_n, global_mid_n])
    # Medium: {min, mid, max}
    negative['medium'] = fuzz.trimf(negative.universe, [global_min_n, global_mid_n, global_max_n])
    # High: {mid, max, max}
    negative['high'] = fuzz.trimf(negative.universe, [global_mid_n, global_max_n, global_max_n])

    # Funciones de membresía para output
    # Negative(op_neg): {0,0,5}
    output['negative'] = fuzz.trimf(output.universe, [0, 0, 5])
    # Neutral(op_neu): {0,5,10}
    output['neutral'] = fuzz.trimf(output.universe, [0, 5, 10])
    # Positive(op_pos): {5,10,10}
    output['positive'] = fuzz.trimf(output.universe, [5, 10, 10])

    # Reglas de Mamdani
    rule1 = ctrl.Rule(positive['low'] & negative['low'], output['neutral']) # pos_low & neg_low -> neutral
    rule2 = ctrl.Rule(positive['medium'] & negative['low'], output['positive']) # pos_medium & neg_low -> positive
    rule3 = ctrl.Rule(positive['high'] & negative['low'], output['positive']) # pos_high & neg_low -> positive
    rule4 = ctrl.Rule(positive['low'] & negative['medium'], output['negative']) # pos_low & neg_medium -> negative
    rule5 = ctrl.Rule(positive['medium'] & negative['medium'], output['neutral']) # pos_medium & neg_medium -> neutral
    rule6 = ctrl.Rule(positive['high'] & negative['medium'], output['positive']) # pos_high & neg_medium -> positive
    rule7 = ctrl.Rule(positive['low'] & negative['high'], output['negative']) # pos_low & neg_high -> negative
    rule8 = ctrl.Rule(positive['medium'] & negative['high'], output['negative']) # pos_medium & neg_high -> negative
    rule9 = ctrl.Rule(positive['high'] & negative['high'], output['neutral']) # pos_high & neg_high -> neutral

    # Se utiliza ControlSystem de Scikit Fuzz para crear el sistema de inferencia y se definen las reglas que creamos
    fuzzy_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    # Se inicializa el simulador
    fuzzy_simulation = ctrl.ControlSystemSimulation(fuzzy_system)

    return fuzzy_simulation

def classify_output(coa):
    if coa < 3.3:
        return 'Negative'
    elif 3.3 <= coa < 6.7:
        return 'Neutral'
    elif 6.7 <= coa:
        return 'Positive'

def compute_sentiment_for_row(row, sentiment):
    """
    Calcula el sentimiento para una fila dada utilizando el modelo de lógica difusa.
    """
    sentiment.input['positive'] = row['positive_score']
    sentiment.input['negative'] = row['negative_score']
    sentiment.compute()  # Realiza la inferencia difusa y la defuzificación
    return sentiment.output['output']

def measure_execution_time(func, *args):
    """
    Mide el tiempo de ejecución de una función.
    """
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

def process_output(df, sentiment):
    """
    Procesa el análisis de sentimiento en cada fila de un DataFrame utilizando lógica difusa 
    y mide el tiempo de ejecución.
    
    Esta función itera sobre cada fila en el DataFrame, aplicando un modelo de lógica difusa 
    para generar una clasificación de sentimiento. También mide el tiempo de ejecución para 
    procesar cada fila. Los resultados se agregan como nuevas columnas en el DataFrame y se 
    guardan en un archivo CSV.
    Args:
        df (pandas.DataFrame): DataFrame de entrada con datos de texto y puntajes de sentimiento.
    Returns:
        tuple:
            - pandas.DataFrame: DataFrame actualizado
            - list of dict: Lista detallada de resultados para cada fila
    """

    # Inicializa las listas para almacenar los resultados
    sentiment_output = [None] * len(df)
    results = []

    # Itera a través de cada fila para procesar el sentimiento y medir el tiempo de ejecución
    for index, row in df.iterrows():
        sentiment_result, execution_time = measure_execution_time(compute_sentiment_for_row, row, sentiment)

    # Almacena el resultado de sentimiento para la fila actual
        sentiment_output[index] = sentiment_result

    # Agrega resultados detallados para cada fila
        results.append({
            'sentence': row['sentence'],
            'sentiment': row['sentiment'],
            'positive_score': row['positive_score'],
            'negative_score': row['negative_score'],
            'calculated_sentiment': sentiment_result,
            'execution_time': execution_time
        })

    # Agrega los resultados del sentimiento al dataframe
    df['resultado_inferencia'] = sentiment_output
    df['sentimiento'] = df['resultado_inferencia'].apply(classify_output)

    # Guarda los resultados en un archivo CSV
    save_results_to_csv(df, "Data/resultados.csv")
    return df, results

def save_results_to_csv(df, output_path):
    """
    Guarda el DataFrame resultante en un archivo CSV.
    
    Args:
        df: DataFrame con los resultados del análisis
        output_path: Ruta donde se guardará el archivo CSV
    """
    try:
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\nResultados guardados exitosamente en: {output_path}")
    except Exception as e:
        print(f"\nError al guardar los resultados: {str(e)}")


def print_results(df, results):
    """
    Imprime un resumen con el total de resultados positivos, negativos y neutrales,
    tiempo para calcular cada uno y el tiempo promedio total.
    """
    # Conteo de sentimientos
    sentiment_counts = df['sentimiento'].value_counts()
    
    # Cálculo de tiempos por categoría
    positive_times = []
    negative_times = []
    neutral_times = []
    
    for index, row in df.iterrows():
        execution_time = results[index]['execution_time']
        sentiment = row['sentimiento']
        
        if sentiment == 'Positive':
            positive_times.append(execution_time)
        elif sentiment == 'Negative':
            negative_times.append(execution_time)
        elif sentiment == 'Neutral':
            neutral_times.append(execution_time)
    
    # Cálculo de promedios
    avg_positive_time = sum(positive_times) / len(positive_times) if positive_times else 0
    avg_negative_time = sum(negative_times) / len(negative_times) if negative_times else 0
    avg_neutral_time = sum(neutral_times) / len(neutral_times) if neutral_times else 0
    total_avg_time = sum([avg_positive_time, avg_negative_time, avg_neutral_time]) / 3
    
    # Impresión del resumen
    print("\n=== RESUMEN DE RESULTADOS ===")
    print(f"\nTotal de resultados:")
    print(f"Positivos: {sentiment_counts.get('Positive', 0)}")
    print(f"Negativos: {sentiment_counts.get('Negative', 0)}")
    print(f"Neutrales: {sentiment_counts.get('Neutral', 0)}")
    
    print(f"\nTiempos promedio de cálculo:")
    print(f"Positivos: {avg_positive_time:.6f} segundos")
    print(f"Negativos: {avg_negative_time:.6f} segundos")
    print(f"Neutrales: {avg_neutral_time:.6f} segundos")
    
    print(f"\nTiempo promedio total: {total_avg_time:.6f} segundos")


if __name__ == "__main__":
    csv_file_path = "Data/train_data_lite.csv"
    df = read_csv_to_dataframe(csv_file_path)
    try:
        find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

    processed_df = preprocess_dataframe(df)
    df_sentiment = analyze_sentiment(processed_df)
    sentiment = fuzzyfy_sentiment(df_sentiment)
    df_sentiment, result = process_output(df_sentiment, sentiment)
    print_results(df_sentiment, result)