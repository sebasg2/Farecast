import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import tensorflow.keras.backend as K
import joblib

# Cargar los datos procesados
df_final = pd.read_csv('processed.csv')

# Preparar las características y la variable objetivo
X = df_final.drop(['Price_Category', 'Price'], axis=1).select_dtypes(include=[float, int])
X.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
y_price = df_final['Price']

# Aplicar la transformación logarítmica a la variable objetivo
y_price_log = np.log1p(y_price)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_price_log, test_size=0.2, random_state=42)

# Escalar las características
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reducir la dimensionalidad usando PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Guardar el scaler y el PCA para uso futuro
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')

# Definir la función de error cuadrático medio logarítmico
def mean_squared_logarithmic_error(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), None)
    y_true = K.clip(y_true, K.epsilon(), None)
    first_log = K.log(y_pred + 1.)
    second_log = K.log(y_true + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)

# Definir y entrenar modelos
def train_model(model, X_train, y_train, model_name, early_stop, loss_func='mean_squared_logarithmic_error'):
    model.compile(optimizer='adam', loss=loss_func)
    model.fit(X_train, y_train, epochs=200, validation_split=0.2, callbacks=[early_stop], verbose=1)
    joblib.dump(model, f'{model_name}.pkl')

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Modelo 1: Red Neuronal con Error Cuadrático Medio Logarítmico
model_1 = Sequential([
    Dense(512, activation='swish', input_shape=(X_train_pca.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='swish'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='swish'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='swish'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1)
])
train_model(model_1, X_train_pca, y_train, 'Trained_model_1', early_stop, mean_squared_logarithmic_error)

#modelo 2: 
# KNN Regression con GridSearchCV
param_grid = {
    'n_neighbors': [5, 7, 9, 11, 13],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid_search = GridSearchCV(estimator=KNeighborsRegressor(), param_grid=param_grid, scoring='neg_mean_absolute_error', cv=5)
grid_search.fit(X_train_pca, y_train)
best_knn = grid_search.best_estimator_
joblib.dump(best_knn, 'final_model.pkl')

# Modelo 3: Red Neuronal con Pérdida de Huber
model_3 = Sequential([
    Dense(512, activation='swish', input_shape=(X_train_pca.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='swish'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='swish'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='swish'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1)
])
train_model(model_3, X_train_pca, y_train, 'Trained_model_3', early_stop, tf.keras.losses.Huber())

# Modelo 4: Red Neuronal con Pérdida de Cuantil
def quantile_loss(q, y_true, y_pred):
    err = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * err, (q - 1) * err), axis=-1)

quantile = 0.56
model_4 = Sequential([
    Dense(512, activation='swish', input_shape=(X_train_pca.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='swish'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='swish'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='swish'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1)
])
train_model(model_4, X_train_pca, y_train, 'Trained_model_4', early_stop, lambda y_true, y_pred: quantile_loss(quantile, y_true, y_pred))

# Modelo 5: Red Neuronal con Pérdida Combinada (Cuantil + Huber)
def combined_loss(q, y_true, y_pred):
    err = y_true - y_pred
    quantile_loss = tf.reduce_mean(tf.maximum(q * err, (q - 1) * err), axis=-1)
    huber_loss = tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)
    return quantile_loss + huber_loss

model_5 = Sequential([
    Dense(512, activation='swish', input_shape=(X_train_pca.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='swish'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='swish'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='swish'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1)
])
train_model(model_5, X_train_pca, y_train, 'Trained_model_5', early_stop, lambda y_true, y_pred: combined_loss(quantile, y_true, y_pred))

# Guardar los datos de entrenamiento y prueba para uso futuro
train_data = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
train_data['y_price'] = np.expm1(y_train)  # Convertir las etiquetas a la escala original
train_data.to_csv('train.csv', index=False)

test_data = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
test_data['y_price'] = np.expm1(y_test)  # Convertir las etiquetas a la escala original
test_data.to_csv('test.csv', index=False)
