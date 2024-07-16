# Presentando....Farecast!

## Descripción
Presentando Farecast! Esta herramienta de predicción de precios de vuelos utiliza un modelo de vecinos Más Cercanos (KNN) para estimar el costo de un boleto aéreo basado en varios factores. Diseñada para ayudar a viajeros y profesionales de la industria aérea, esta aplicación proporciona estimaciones rápidas y precisas de precios de vuelos.

## Características
- Predicción de precios basada en múltiples factores como aerolínea, origen, destino, fecha y número de escalas.
- Interfaz de usuario amigable creada con Streamlit.
- Modelo KNN para predicciones precisas.
- Preprocesamiento de datos automático incluyendo codificación one-hot y escalado de características.

### Dos funciones principales:
1. Predecir el precio de un vuelo específico basado en ciertas características (aerolínea, origen, destino, fecha y número de escalas).
2. Conseguir el período más barato para viajar desde un país de origen a un país destino, identificando el día más barato del mes más barato para viajar.

## Modelo
El predictor utiliza un modelo KNN (K-Nearest Neighbors) entrenado con datos históricos de precios de vuelos obtenidos del API de Amadeus. El modelo considera los siguientes factores:
- Hora de salida
- Aerolínea
- Número de escalas
- País de origen y destino
- Fecha del vuelo

---

# Introducing... Farecast!

## Description
Introducing Farecast! This flight price prediction tool uses a K-Nearest Neighbors (KNN) model to estimate the cost of an airline ticket based on various factors. Designed to help travelers and airline industry professionals, this application provides quick and accurate flight price estimates.

## Features
- Price prediction based on multiple factors such as airline, origin, destination, date, and number of stops.
- User-friendly interface created with Streamlit.
- KNN model for accurate predictions.
- Automatic data preprocessing including one-hot encoding and feature scaling.

### Two main functions:
1. Predict the price of a specific flight based on certain characteristics (airline, origin, destination, date, and number of stops).
2. Find the cheapest period to travel from an origin country to a destination country, identifying the cheapest day of the cheapest month to travel.

## Model
The predictor uses a KNN (K-Nearest Neighbors) model trained with historical flight price data obtained from the Amadeus API. The model considers the following factors:
- Departure time
- Airline
- Number of stops
- Origin and destination country
- Flight date

