
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('raw.csv')


df_sorted = df.sort_values(by='Origin Country')


df_sorted.reset_index(drop=True, inplace=True)


airports_dict = {
    "Afghanistan": "KBL",
    "Albania": "TIA",
    "Algeria": "ALG",
    "Andorra": "LEU",
    "Angola": "LAD",
    "Antigua and Barbuda": "ANU",
    "Argentina": "EZE",
    "Armenia": "EVN",
    "Australia": "SYD",
    "Austria": "VIE",
    "Azerbaijan": "GYD",
    "Bahamas": "NAS",
    "Bahrain": "BAH",
    "Bangladesh": "DAC",
    "Barbados": "BGI",
    "Belarus": "MSQ",
    "Belgium": "BRU",
    "Belize": "BZE",
    "Benin": "COO",
    "Bhutan": "PBH",
    "Bolivia": "VVI",
    "Bosnia and Herzegovina": "SJJ",
    "Botswana": "GBE",
    "Brazil": "GRU",
    "Brunei Darussalam": "BWN",
    "Bulgaria": "SOF",
    "Burkina Faso": "OUA",
    "Burundi": "BJM",
    "Cabo Verde": "SID",
    "Cambodia": "PNH",
    "Cameroon": "DLA",
    "Canada": "YYZ",
    "Central African Republic": "BGF",
    "Chad": "NDJ",
    "Chile": "SCL",
    "China": "PEK",
    "Colombia": "BOG",
    "Comoros": "HAH",
    "Congo": "FIH",
    "Costa Rica": "SJO",
    "Ivory Coast": "ABJ",
    "Croatia": "ZAG",
    "Cuba": "HAV",
    "Cyprus": "LCA",
    "Czech Republic": "PRG",
    "South Korea": "FNJ",
    "Congo": "FIH",
    "Denmark": "CPH",
    "Djibouti": "JIB",
    "Dominica": "DOM",
    "Dominican Republic": "SDQ",
    "Ecuador": "UIO",
    "Egypt": "CAI",
    "El Salvador": "SAL",
    "Equatorial Guinea": "SSG",
    "Eritrea": "ASM",
    "Estonia": "TLL",
    "Eswatini": "MTS",
    "Ethiopia": "ADD",
    "Fiji": "NAN",
    "Finland": "HEL",
    "France": "CDG",
    "Gabon": "LBV",
    "Gambia": "BJL",
    "Georgia": "TBS",
    "Germany": "FRA",
    "Ghana": "ACC",
    "Greece": "ATH",
    "Grenada": "GND",
    "Guatemala": "GUA",
    "Guinea": "CKY",
    "Guinea Bissau": "OXB",
    "Guyana": "GEO",
    "Haiti": "PAP",
    "Honduras": "SAP",
    "Hungary": "BUD",
    "Iceland": "KEF",
    "India": "DEL",
    "Indonesia": "CGK",
    "Iran": "THR",
    "Iraq": "BGW",
    "Ireland": "DUB",
    "Israel": "TLV",
    "Italy": "FCO",
    "Jamaica": "SIA",
    "Japan": "NRT",
    "Jordan": "AMM",
    "Kazakhstan": "ALA",
    "Kenya": "NBO",
    "Kiribati": "TRW",
    "Kuwait": "KWI",
    "Kyrgyzstan": "FRU",
    "Lao": "VTE",
    "Latvia": "RIX",
    "Lebanon": "BEY",
    "Lesotho": "MSU",
    "Liberia": "ROB",
    "Libya": "TIP",
    "Liechtenstein": "LI",
    "Lithuania": "VNO",
    "Luxembourg": "LUX",
    "Madagascar": "TNR",
    "Malawi": "LLW",
    "Malaysia": "KUL",
    "Maldives": "MLE",
    "Mali": "BKO",
    "Malta": "MLA",
    "Marshall Islands": "MAJ",
    "Mauritania": "NKC",
    "Mauritius": "MRU",
    "Mexico": "MEX",
    "Micronesia": "KSA",
    "Monaco": "MUC",
    "Mongolia": "ULN",
    "Montenegro": "TIV",
    "Morocco": "CMN",
    "Mozambique": "LUM",
    "Myanmar": "RGN",
    "Namibia": "WDH",
    "Nauru": "INU",
    "Nepal": "LUA",
    "Netherlands": "AMS",
    "New Zealand": "AKL",
    "Nicaragua": "MGA",
    "Niger": "NIM",
    "Nigeria": "LOS",
    "North Macedonia": "SKP",
    "Norway": "OSL",
    "Oman": "MCT",
    "Pakistan": "KHI",
    "Palau": "ROR",
    "Panama": "PTY",
    "Papua New Guinea": "POM",
    "Paraguay": "ASU",
    "Peru": "LIM",
    "Philippines": "MNL",
    "Poland": "WAW",
    "Portugal": "LIS",
    "Qatar": "DOH",
    "Republic of Korea": "IIA",
    "Republic of Moldova": "KIV",
    "Romania": "OTP",
    "Russian Federation": "SVO",
    "Rwanda": "KGL",
    "Saint Kitts and Nevis": "SKB",
    "Saint Lucia": "UVF",
    "Saint Vincent and the Grenadines": "SVD",
    "Samoa": "APW",
    "San Marino": "RMI",
    "Sao Tome and Principe": "TMS",
    "Saudi Arabia": "DMM",
    "Senegal": "DSS",
    "Serbia": "BEG",
    "Seychelles": "SEZ",
    "Sierra Leone": "FNA",
    "Singapore": "SIN",
    "Slovakia": "BTS",
    "Slovenia": "LJU",
    "Solomon Islands": "HIR",
    "Somalia": "MGQ",
    "South Africa": "JNB",
    "South Sudan": "JUB",
    "Spain": "MAD",
    "Sri Lanka": "CMB",
    "Sudan": "KRT",
    "Suriname": "PBM",
    "Sweden": "ARN",
    "Switzerland": "ZRH",
    "Syrian Arab Republic": "DAM",
    "Tajikistan": "DYU",
    "Thailand": "BKK",
    "Timor-Leste": "DIL",
    "Togo": "LFW",
    "Tonga": "TBU",
    "Trinidad and Tobago": "POS",
    "Tunisia": "TUN",
    "Turkey": "IST",
    "Turkmenistan": "ASB",
    "Tuvalu": "FUN",
    "Uganda": "EBB",
    "Ukraine": "KBP",
    "United Arab Emirates": "DXB",
    "United Kingdom": "LHR",
    "United Republic of Tanzania": "DAR",
    "United States of America": "DEN",
    "Uruguay": "MVD",
    "Uzbekistan": "TAS",
    "Vanuatu": "VLI",
    "Venezuela": "CCS",
    "Viet Nam": "SGN",
    "Yemen": "SAH",
    "Zambia": "LUN",
    "Zimbabwe": "HRE"
}


import airportsdata

airports = airportsdata.load('IATA')

def get_airport_coordinates(country):
    iata_code = airports_dict.get(country)
    print(iata_code)
    if iata_code:
        airport = airports.get(iata_code)
        if airport:
            return airport['lat'], airport['lon']
    return None, None




from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la tierra en kilometros
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def calculate_distance(row):
    origin_lat, origin_lon = get_airport_coordinates(row['Origin Country'])
    destination_lat, destination_lon = get_airport_coordinates(row['Destination Country'])
    if origin_lat is not None and destination_lat is not None:
        return haversine(origin_lat, origin_lon, destination_lat, destination_lon)
    return None


df_sorted['Distance (km)'] = df_sorted.apply(calculate_distance, axis=1)




df_sorted

def usd_to_eur(price_usd):
    exchange_rate = 0.93  
    return price_usd * exchange_rate


df_sorted['Price'] = df_sorted['Price'].apply(usd_to_eur)


df_sorted



from Countrydetails import countries
country = countries.all_countries()
continent_countries_dict=country.countries_in_continents()

def map_to_continent(country_name):
    try:
        # Iterate through the dictionary to find the continent
        for continent, countries_list in continent_countries_dict.items():
            if country_name in countries_list:
                return continent
        return 'Unknown'  # Return 'Unknown' if country not found in the dictionary
    except Exception as e:
        print(f"Error mapping {country_name} to continent: {e}")
        return 'Unknown'
df_sorted['Origin Continent'] = df_sorted['Origin Country'].apply(map_to_continent)
df_sorted['Destination Continent'] = df_sorted['Destination Country'].apply(map_to_continent)



unknown_origin = df_sorted[df_sorted['Origin Continent'] == 'Unknown']['Origin Country']
unknown_destination = df_sorted[df_sorted['Destination Continent'] == 'Unknown']['Destination Country']

unknown_countries = pd.concat([unknown_origin, unknown_destination]).reset_index(drop=True)

unknown_countries.unique()

country_to_continent = {
    'Brunei Darussalam': 'Asia',
    'Cabo Verde': 'Africa',
    'Eswatini': 'Africa',
    'Fiji': 'Oceania',
    'Lao': 'Asia',
    'Libya': 'Africa',
    'Micronesia': 'Oceania',
    'Serbia': 'Europe',
    'Timor-Leste': 'Asia',
    'United Republic of Tanzania': 'Africa',
    'United States of America': 'North America',
    'Viet Nam': 'Asia',
    'Guinea Bissau': 'Africa'
}

df_sorted.loc[df_sorted['Origin Country'].isin(country_to_continent.keys()), 'Origin Continent'] = \
    df_sorted['Origin Country'].map(country_to_continent)


df_sorted.loc[df_sorted['Destination Country'].isin(country_to_continent.keys()), 'Destination Continent'] = \
    df_sorted['Destination Country'].map(country_to_continent)



# Eliminar los rows que tengan missing values en duration ya que son pocos y no representan una gran parte de los datos


df_sorted.info()


df_cleaned = df_sorted.dropna(subset=['Duration'])


df_cleaned.info()


# La columna de Departure time se convierte a formato datetime


df_cleaned['Departure Time'] = pd.to_datetime(df_cleaned['Departure Time'])


# Se ve la correlacion de los features con el precio


numeric_cols = ['Duration', 'Price', 'Number of Stops', 'Distance (km)']

numeric_data = df_cleaned[numeric_cols]


correlation_matrix = numeric_data.corr()

print("Correlation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Flight Data')
plt.show()

# Se observa que existe una correlación aproximada de 0.3 entre varios de los atributos y el precio en los datos de vuelos. Aunque estas correlaciones no son malas, es crucial  crear la mayor cantidad de features a traves de estas ya existentes


# Los primeros features que crease son la hora de salida y el hecho de que si el vuelo es directo o no


df_cleaned['Hour_of_Departure'] = pd.to_datetime(df_cleaned['Departure Time']).dt.hour
df_cleaned['Is_Direct_Flight'] = df_cleaned['Number of Stops'].apply(lambda x: 1 if x == 0 else 0)




numeric_data = df_cleaned.select_dtypes(include=['int', 'float'])


correlation_matrix = numeric_data.corr()

print("Correlation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Flight Data')
plt.show()


# Aunque las correlaciones no exhiben valores altos, es crucial incrementar la cantidad de características disponibles. La estrategia que se va a seguir para este modelo consiste en adquirir una amplia gama de características y luego emplear PCA para identificar aquellas que, aunque no sean inicialmente evidentes, contribuyan significativamente a la explicación de la variabilidad en los datos.

# Se saca el mes y el dia




df_cleaned['Month'] = df_cleaned['Departure Time'].dt.month
df_cleaned['Day'] = df_cleaned['Departure Time'].dt.day





# Se crean categorias basada en la duracion del vuelo y la distancia del vuelo.


# Function to categorize flight durations
def categorize_flight_duration(duration):
    if duration <= 12:  
        return 'Short'
    elif duration <= 24: 
        return 'Medium'
    elif duration>24:
        return 'Long'


def categorize_flight_distance(distance):
    if distance <= 4000:
        return 'Short Haul'
    elif distance <= 7500:
        return 'Medium Haul'
    elif distance >7500:
        return 'Long Haul'

df_cleaned['Flight_Duration_Category'] = df_cleaned['Duration'].apply(categorize_flight_duration)
df_cleaned['Flight_Distance_Category'] = df_cleaned['Distance (km)'].apply(categorize_flight_distance)


df_cleaned['Flight_Distance_Category'].value_counts()
df_cleaned['Flight_Duration_Category'].value_counts()


# Ahora se consiguen mucho mas categorias con pandas get dummies.



columns_to_dummy = ['Airline Code', 'Origin Continent', 'Destination Continent','Flight_Distance_Category','Flight_Duration_Category']


dummy_df = pd.get_dummies(df_cleaned[columns_to_dummy], dtype=float)


df_with_dummies = pd.concat([df_cleaned.drop(columns_to_dummy, axis=1), dummy_df], axis=1)



X = df_with_dummies.drop([ 'Price'], axis=1)
y_price = df_with_dummies['Price']


# Se consigue la cantidad de dias que falta para que salga el vuelo


from datetime import datetime

def days_until_flight(departure_time):
    now = datetime.now()
    days_difference = (departure_time - now).days
    return days_difference


X['Days_Until_Flight'] = X['Departure Time'].apply(days_until_flight)

# %%
X['Extra_Long_Flight'] = (X['Duration'] > 40).astype(float)

# Hay 206 columnas 


X.columns


# Se crea un histograma del precio para ver que tipo de distribucion sigue


plt.figure(figsize=(10, 6))
plt.hist(y_price, bins=30, edgecolor='k')
plt.title('Histogram of y_price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# Con este histograma se puede ver que si se transforma logaritmicamente el precio seguiria una distribucion mas normal que cuando no se hace la transformacion


import numpy as np

y_price_log = np.log1p(y_price)  

plt.figure(figsize=(10, 6))
plt.hist(y_price_log, bins=30, edgecolor='k')
plt.title('Histogram of log-transformed y_price')
plt.xlabel('Log-Transformed Price')
plt.ylabel('Frequency')
plt.show()




# Ahora se hace un boxplot de los precios para ver cuales son las outliers de manera visual. Se puede ver que hay muchos vuelos encima del precio medio en los datos. A manera simple y visual , se puede ver que los outliers son todos los vuelos mayor de 5000 euros

# %%
plt.figure(figsize=(10, 15))
plt.boxplot(y_price, vert=False, whis=1.5)  # Increasing whisker length
plt.title('Boxplot of y_price with adjusted whiskers')
plt.xlabel('Price')
plt.show()

# %% [markdown]
# Para verificar se utiliza el z score para ver con certeza. Se puede comprobar que los valores si empiezan a ser outliers alrededor de los 5000 euros

# %%
df_cleaned['z_score'] = (df_cleaned['Price'] - df_cleaned
                         ['Price'].mean()) / df_cleaned['Price'].std()


outliers = df_cleaned[np.abs(df_cleaned['z_score']) >=2
                      ]

# Sort outliers by 'y_price' in descending order
sorted_outliers = outliers.sort_values(by='Price', ascending=True)


sorted_outliers


# %%
df_cleaned.drop(columns='z_score',axis=1,inplace=True)

# %% [markdown]
# Se van a eliminar los outliers del dataset. Se eliminan  1211 registros pasando de 22483 registros a 21238 datos

# %%
len(X)

# %%
mask = y_price <= 5000
y_price = y_price[mask]
X = X[mask]


# %%
len(X)

# %% [markdown]
# Ya teniendo el dataframe procesado , se va a calcular las categorias de precio. Ver como se podrian dividir el precio en categorias resulta util ya que permite ver si estan todos los tipos de precios bien representados 

# %%
df_final=pd.concat([X,y_price],axis=1)
df_final

# %%
df_final.describe()

# %%
plt.hist(df_final['Price'], bins=10, edgecolor='k')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Histogram of Prices')
plt.show()

# %%
quantiles = df_final['Price'].quantile([0.33, 0.66])  


bins = [0, quantiles[0.33], quantiles[0.66], np.inf]
labels = [0, 1, 2]
df_final['Price_Category'] = pd.cut(df_final['Price'], bins=bins, labels=labels)

df_final['Price_Category'].value_counts()

# %% [markdown]
# Finalmente se divide en X y en y_price los datos 

# %%
X=df_final.drop(['Price_Category','Price'], axis=1).select_dtypes(include=[float, int])
y_price=df_final['Price']

# %%
X

# %%
y_price

# %%
df_final=df_final.to_csv('processed.csv')

# %%
df_final=pd.read_csv('processed.csv')

# %%
df_final.drop(columns=['Unnamed: 0'],axis=1, inplace=True)

# %% [markdown]
# Tambien se ve como se ven estos grupos usando clusters y aprendizaje no supervisado

# %%
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Supongamos que X es tu conjunto de datos de características relevantes
# X = df[['Duration', 'Number of Stops', 'Distance (km)', 'Hour_of_Departure', 'Month', 'Day']].values

# Escalado de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicación de K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Predicciones de los clústeres
cluster_labels = kmeans.predict(X_scaled)

# Asumiendo que tienes los precios en df['Price'].values
df_final['Cluster'] = cluster_labels

# Calcula el precio promedio por clúster
cluster_means = df_final.groupby('Cluster')['Price'].mean()

print(cluster_means)


