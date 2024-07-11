## Descripción
Esta herramienta de predicción de precios de vuelos utiliza un modelo de vecinos Más Cercanos (KNN) para estimar el costo de un boleto aéreo basado en varios factores. Diseñada para ayudar a viajeros y profesionales de la industria aérea, esta aplicación proporciona estimaciones rápidas y precisas de precios de vuelos.

## Características
- Predicción de precios basada en múltiples factores como aerolínea, origen, destino, fecha y número de escalas.
- Interfaz de usuario amigable creada con Streamlit.
- Modelo KNN para predicciones precisas.
- Preprocesamiento de datos automático incluyendo codificación one-hot y escalado de características.


## Modelo
El predictor utiliza un modelo KNN (K-Nearest Neighbors) entrenado con datos históricos de precios de vuelos. El modelo considera factores como:
- Hora de salida
- Aerolínea
- Número de escalas
- País de origen y destino
- Fecha del vuelo
