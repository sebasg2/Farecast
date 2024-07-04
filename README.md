## Modelo para Predecir la Categoría de Precio de Vuelos

Este proyecto utiliza un modelo de aprendizaje automático para predecir la categoría de precio de vuelos. Las categorías de precio se dividen en tres grupos: Bajo, Medio y Alto, basados en los precios de los vuelos.

### Datos y Preprocesamiento

Los datos provienen de vuelos aleatorios disponibles actualmente, obtenidos mediante la API de Amadeus. Incluyen vuelos desde y hacia cualquier país, considerando solo el mes y el día del mes para crear un modelo generalizable a través del tiempo.

Se utilizaron varias características, incluyendo la distancia del vuelo en kilómetros, la duración del vuelo, el número de paradas y la aerolínea. A traves de  estas características se codificaron nuevas variables con pandas.dummies.

### Modelado

El modelo se construyó utilizando un clasificador XGBoost y PCA (Análisis de Componentes Principales) para reducir la dimensionalidad y mejorar la precisión del modelo. PCA ayuda a capturar las características más importantes de manera eficiente.

### Categorización de Precios

Para etiquetar los datos, se definió una función que asigna cada precio a una categoría específica: Bajo, Medio o Alto, basado en umbrales predefinidos.

```python
def categorize_price(price):
    if price <= 834.024:
        return 0  # Bajo
    elif price <= 1536.825:
        return 1  # Medio
    else:
        return 2  # Alto

df_cleaned['Price_Category_Encoded'] = df_cleaned['Price'].apply(categorize_price)
