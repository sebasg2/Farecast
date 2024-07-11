import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

# Función para calcular la pérdida de cuantiles (necesaria para cargar algunos modelos)
def quantile_loss(q, y_true, y_pred):
    err = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * err, (q - 1) * err), axis=-1)

# Función para calcular la pérdida combinada (cuantil + Huber)
def combined_loss(q, y_true, y_pred):
    err = y_true - y_pred
    quantile_loss = tf.reduce_mean(tf.maximum(q * err, (q - 1) * err), axis=-1)
    huber_loss = tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)
    return quantile_loss + huber_loss

# Cargar los datos de prueba
test_data = pd.read_csv('test.csv')
X_test = test_data.drop('y_price', axis=1)
y_test = test_data['y_price']

# Aplicar la transformación logarítmica a y_test
y_test_log = np.log1p(y_test)

# Lista de modelos a evaluar
modelos = [
    ('Trained_model_1.pkl', MinMaxScaler(), 0.95),
    ('final_model.pkl', StandardScaler(), 0.95),
    ('Trained_model_3.pkl', MinMaxScaler(), 0.95),
    ('Trained_model_4.pkl', MinMaxScaler(), 0.95),
    ('Trained_model_5.pkl', MinMaxScaler(), 0.9)
]

resultados = {}

for modelo_nombre, scaler, pca_componentes in modelos:
    print(f"\nEvaluando modelo: {modelo_nombre}")
    
    # Cargar el modelo
    modelo = joblib.load(modelo_nombre)
    
    # Aplicar el escalado a X_test
    X_test_scaled = scaler.fit_transform(X_test)
    
    # Aplicar PCA a X_test_scaled
    pca = PCA(n_components=pca_componentes)
    X_test_pca = pca.fit_transform(X_test_scaled)
    
    # Realizar predicciones
    if 'final_model' in modelo_nombre:  # El modelo KNN
        y_pred_log = modelo.predict(X_test_pca)
    else:  # Modelos de redes neuronales
        y_pred_log = modelo.predict(X_test_pca).flatten()
    
    # Invertir la transformación logarítmica en las predicciones
    y_pred = np.expm1(y_pred_log)
    
    # Calcular el Error Absoluto Medio
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Error Absoluto Medio: {mae}')
    
    # Calcular el porcentaje de error
    error_porcentaje = np.abs(y_test - y_pred) / y_test * 100
    error_promedio = np.mean(error_porcentaje)
    print(f'Error porcentual promedio: {error_promedio:.2f}%')
    
    # Guardar resultados
    resultados[modelo_nombre] = {
        'MAE': mae,
        'Error_Porcentual_Promedio': error_promedio,
        'Predicciones': y_pred
    }
    
    # Visualizar los resultados
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(y_test)), y_test, color='green', label='Datos reales', alpha=0.5)
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicciones', alpha=0.5)
    plt.title(f'Predicciones vs Valores Reales - {modelo_nombre}')
    plt.xlabel('Índice de muestra')
    plt.ylabel('Precio')
    plt.legend()
    plt.savefig(f'resultados_{modelo_nombre}.png')
    plt.close()

# Comparar resultados de todos los modelos
comparacion = pd.DataFrame({
    modelo: {
        'MAE': res['MAE'],
        'Error_Porcentual_Promedio': res['Error_Porcentual_Promedio']
    } for modelo, res in resultados.items()
}).T

print("\nComparación de modelos:")
print(comparacion)

# Guardar la comparación en un archivo CSV
comparacion.to_csv('comparacion_modelos.csv')

# Guardar todas las predicciones en un archivo CSV
predicciones = pd.DataFrame({
    f'Predicciones_{modelo}': res['Predicciones'] for modelo, res in resultados.items()
})
predicciones['Valores_Reales'] = y_test
predicciones.to_csv('todas_las_predicciones.csv', index=False)

print("\nSe han guardado los resultados en 'comparacion_modelos.csv' y 'todas_las_predicciones.csv'")
print("También se han guardado gráficos de cada modelo como archivos PNG.")