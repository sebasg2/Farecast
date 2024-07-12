# %%
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib

modelo = joblib.load('final_model.pkl')

# %%
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

# %%


def get_airport_coordinates(country):
    import airportsdata
    
    airports = airportsdata.load('IATA')
    iata_code = airports_dict.get(country)
    print(iata_code)
    if iata_code:
        airport = airports.get(iata_code)
        if airport:
            return airport['lat'], airport['lon']
    return None, None


# %%
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# %%
def calculate_distance(row):
    
    origin_lat, origin_lon = get_airport_coordinates(row['Origin_Country'])
    destination_lat, destination_lon = get_airport_coordinates(row['Destination_Country'])
    if origin_lat is not None and destination_lat is not None:
        return haversine(origin_lat, origin_lon, destination_lat, destination_lon)
    return None

# %%


def map_to_continent(country_name):
    
    from Countrydetails import countries
    
    country = countries.all_countries()
    continent_countries_dict=country.countries_in_continents()
    
    try:
        # Iterate through the dictionary to find the continent
        for continent, countries_list in continent_countries_dict.items():
            if country_name in countries_list:
                return continent
        return 'Unknown'  # Return 'Unknown' if country not found in the dictionary
    except Exception as e:
        print(f"Error mapping {country_name} to continent: {e}")
        return 'Unknown'


# %%

def calculate_flight_duration(distance_km):
    # Velocidad promedio de un avión comercial en km/h
    average_speed_kmh = 900
    if distance_km is None:
        return None

    # Calcular la duración en horas
    duration_hours = distance_km / average_speed_kmh

    # Convertir la duración a decimal
    hours = int(duration_hours)
    minutes = (duration_hours - hours) * 60
    duration_decimal = hours + (minutes / 60)

    return round(duration_decimal, 2) 



# %%
columns_with_dummies = [
    'Airline Code_2J',
    'Airline Code_3K',
    'Airline Code_3M',
    'Airline Code_3U',
    'Airline Code_4Y',
    'Airline Code_4Z',
    'Airline Code_5F',
    'Airline Code_5U',
    'Airline Code_5Z',
    'Airline Code_6H',
    'Airline Code_6X',
    'Airline Code_7C',
    'Airline Code_8M',
    'Airline Code_A1',
    'Airline Code_A3',
    'Airline Code_AC',
    'Airline Code_AD',
    'Airline Code_AF',
    'Airline Code_AH',
    'Airline Code_AI',
    'Airline Code_AR',
    'Airline Code_AS',
    'Airline Code_AT',
    'Airline Code_AV',
    'Airline Code_AY',
    'Airline Code_AZ',
    'Airline Code_B6',
    'Airline Code_BG',
    'Airline Code_BI',
    'Airline Code_BJ',
    'Airline Code_BP',
    'Airline Code_BR',
    'Airline Code_BT',
    'Airline Code_BW',
    'Airline Code_CA',
    'Airline Code_CI',
    'Airline Code_CM',
    'Airline Code_CU',
    'Airline Code_CX',
    'Airline Code_CZ',
    'Airline Code_D8',
    'Airline Code_DE',
    'Airline Code_DO',
    'Airline Code_DT',
    'Airline Code_DY',
    'Airline Code_EI',
    'Airline Code_EK',
    'Airline Code_EN',
    'Airline Code_ET',
    'Airline Code_EW',
    'Airline Code_EY',
    'Airline Code_FA',
    'Airline Code_FB',
    'Airline Code_FI',
    'Airline Code_FJ',
    'Airline Code_FM',
    'Airline Code_FN',
    'Airline Code_FZ',
    'Airline Code_G3',
    'Airline Code_GA',
    'Airline Code_GF',
    'Airline Code_GK',
    'Airline Code_H1',
    'Airline Code_HA',
    'Airline Code_HC',
    'Airline Code_HF',
    'Airline Code_HM',
    'Airline Code_HO',
    'Airline Code_HU',
    'Airline Code_HX',
    'Airline Code_HY',
    'Airline Code_IB',
    'Airline Code_ID',
    'Airline Code_IE',
    'Airline Code_J2',
    'Airline Code_JD',
    'Airline Code_JL',
    'Airline Code_JQ',
    'Airline Code_JU',
    'Airline Code_JY',
    'Airline Code_K6',
    'Airline Code_KC',
    'Airline Code_KE',
    'Airline Code_KL',
    'Airline Code_KM',
    'Airline Code_KP',
    'Airline Code_KQ',
    'Airline Code_KR',
    'Airline Code_KU',
    'Airline Code_L6',
    'Airline Code_LA',
    'Airline Code_LG',
    'Airline Code_LH',
    'Airline Code_LO',
    'Airline Code_LX',
    'Airline Code_LY',
    'Airline Code_ME',
    'Airline Code_MF',
    'Airline Code_MH',
    'Airline Code_MK',
    'Airline Code_MS',
    'Airline Code_MU',
    'Airline Code_NF',
    'Airline Code_NH',
    'Airline Code_NK',
    'Airline Code_NZ',
    'Airline Code_OB',
    'Airline Code_OD',
    'Airline Code_OK',
    'Airline Code_OM',
    'Airline Code_OR',
    'Airline Code_OS',
    'Airline Code_OU',
    'Airline Code_OZ',
    'Airline Code_P0',
    'Airline Code_P4',
    'Airline Code_PC',
    'Airline Code_PG',
    'Airline Code_PK',
    'Airline Code_PR',
    'Airline Code_PU',
    'Airline Code_PW',
    'Airline Code_PX',
    'Airline Code_PY',
    'Airline Code_QF',
    'Airline Code_QR',
    'Airline Code_QS',
    'Airline Code_QV',
    'Airline Code_RJ',
    'Airline Code_RN',
    'Airline Code_RO',
    'Airline Code_RQ',
    'Airline Code_SA',
    'Airline Code_SB',
    'Airline Code_SC',
    'Airline Code_SG',
    'Airline Code_SK',
    'Airline Code_SL',
    'Airline Code_SN',
    'Airline Code_SQ',
    'Airline Code_SS',
    'Airline Code_SV',
    'Airline Code_TC',
    'Airline Code_TG',
    'Airline Code_TK',
    'Airline Code_TM',
    'Airline Code_TP',
    'Airline Code_TR',
    'Airline Code_TU',
    'Airline Code_UA',
    'Airline Code_UB',
    'Airline Code_UK',
    'Airline Code_UL',
    'Airline Code_UP',
    'Airline Code_UR',
    'Airline Code_UU',
    'Airline Code_UX',
    'Airline Code_VA',
    'Airline Code_VB',
    'Airline Code_VF',
    'Airline Code_VN',
    'Airline Code_VS',
    'Airline Code_VY',
    'Airline Code_W2',
    'Airline Code_W3',
    'Airline Code_WB',
    'Airline Code_WF',
    'Airline Code_WM',
    'Airline Code_WS',
    'Airline Code_WY',
    'Airline Code_X1',
    'Airline Code_XY',
    'Airline Code_ZB',
    'Airline Code_ZH',
    'Airline Code_ZL',
    'Airline Code_ZP',
    'Origin Continent_Africa',
    'Origin Continent_Asia',
    'Origin Continent_Europe',
    'Origin Continent_North America',
    'Origin Continent_Oceania',
    'Origin Continent_South America',
    'Destination Continent_Africa',
    'Destination Continent_Asia',
    'Destination Continent_Europe',
    'Destination Continent_North America',
    'Destination Continent_Oceania',
    'Destination Continent_South America',
    'Flight_Distance_Category_Long Haul',
    'Flight_Distance_Category_Medium Haul',
    'Flight_Distance_Category_Short Haul',
    'Flight_Duration_Category_Long',
    'Flight_Duration_Category_Medium',
    'Flight_Duration_Category_Short',
 
]



# %% [markdown]
# INCLUIR MES Y DIA

# %%






# Prediction function
def predecir(Hour_of_Departure, airline, Stops, Origin_Country, Destination_Country, day, month, year):
    
    def categorize_flight_duration(duration):
        return pd.cut(duration, bins=[-np.inf, 12, 24, np.inf], labels=['Short', 'Medium', 'Long'])

    def categorize_flight_distance(distance):
        return pd.cut(distance, bins=[-np.inf, 4000, 7500, np.inf], labels=['Short Haul', 'Medium Haul', 'Long Haul'])

    def days_until_flight(departure_time):
        now = datetime.now()
        departure_date = datetime(year, month, day, Hour_of_Departure)
        days_difference = (departure_date - now).days
        return days_difference

    input_data = {
        'Hour_of_Departure': [Hour_of_Departure],
        'Airline': [airline],
        'Number of Stops': [Stops],
        'Origin_Country': [Origin_Country],
        'Destination_Country': [Destination_Country],
        'Month': [month],
        'Day': [day]
    }
    
    input_df = pd.DataFrame(input_data)

    # Calculate distance and flight duration
    input_df['Distance (km)'] = calculate_distance(input_df.iloc[0])
    input_df['Duration'] = calculate_flight_duration(input_df['Distance (km)'])

    # Determine Flight_Duration_Category based on duration
    input_df['Flight_Duration_Category'] = categorize_flight_duration(input_df['Duration'])

    # Determine Direct_Flight based on stops
    input_df['Is_Direct_Flight'] = input_df['Number of Stops'].apply(lambda x: 1 if x == 0 else 0)

    # Determine Flight_Distance_Category based on distance
    input_df['Flight_Distance_Category'] = categorize_flight_distance(input_df['Distance (km)'])

    # Map countries to continents
    input_df['Origin Continent'] = input_df['Origin_Country'].apply(map_to_continent)
    input_df['Destination Continent'] = input_df['Destination_Country'].apply(map_to_continent)

    # Get dummies for categorical variables
    columns_to_dummy = ['Airline', 'Origin Continent', 'Destination Continent', 'Flight_Duration_Category', 'Flight_Distance_Category']
    dummy_df = pd.get_dummies(input_df[columns_to_dummy], drop_first=True, dtype=float)
    
    # Concatenate the original input_df with the dummy columns
    input_df = pd.concat([input_df.drop(columns_to_dummy, axis=1), dummy_df], axis=1)

    # Ensure the input_df has the same columns as the expected columns with dummies
    missing_cols = set(columns_with_dummies) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    # Calculate days until flight
    departure_date = datetime(year, month, day, Hour_of_Departure)
    input_df['Days_Until_Flight'] = days_until_flight(departure_date)

    # Determine if it's an extra long flight
    input_df['Extra_Long_Flight'] = (input_df['Duration'] > 40).astype(float)  # Example threshold for extra long flight

    input_df = input_df[[
        'Duration', 'Number of Stops', 'Distance (km)', 'Hour_of_Departure', 'Is_Direct_Flight',
        'Month', 'Day', 'Airline Code_2J', 'Airline Code_3K', 'Airline Code_3M', 'Airline Code_3U', 
        'Airline Code_4Y', 'Airline Code_4Z', 'Airline Code_5F', 'Airline Code_5U', 'Airline Code_5Z',
        'Airline Code_6H', 'Airline Code_6X', 'Airline Code_7C', 'Airline Code_8M', 'Airline Code_A1',
        'Airline Code_A3', 'Airline Code_AC', 'Airline Code_AD', 'Airline Code_AF', 'Airline Code_AH',
        'Airline Code_AI', 'Airline Code_AR', 'Airline Code_AS', 'Airline Code_AT', 'Airline Code_AV',
        'Airline Code_AY', 'Airline Code_AZ', 'Airline Code_B6', 'Airline Code_BG', 'Airline Code_BI',
        'Airline Code_BJ', 'Airline Code_BP', 'Airline Code_BR', 'Airline Code_BT', 'Airline Code_BW',
        'Airline Code_CA', 'Airline Code_CI', 'Airline Code_CM', 'Airline Code_CU', 'Airline Code_CX',
        'Airline Code_CZ', 'Airline Code_D8', 'Airline Code_DE', 'Airline Code_DO', 'Airline Code_DT',
        'Airline Code_DY', 'Airline Code_EI', 'Airline Code_EK', 'Airline Code_EN', 'Airline Code_ET',
        'Airline Code_EW', 'Airline Code_EY', 'Airline Code_FA', 'Airline Code_FB', 'Airline Code_FI',
        'Airline Code_FJ', 'Airline Code_FM', 'Airline Code_FN', 'Airline Code_FZ', 'Airline Code_G3',
        'Airline Code_GA', 'Airline Code_GF', 'Airline Code_GK', 'Airline Code_H1', 'Airline Code_HA',
        'Airline Code_HC', 'Airline Code_HF', 'Airline Code_HM', 'Airline Code_HO', 'Airline Code_HU',
        'Airline Code_HX', 'Airline Code_HY', 'Airline Code_IB', 'Airline Code_ID', 'Airline Code_IE',
        'Airline Code_J2', 'Airline Code_JD', 'Airline Code_JL', 'Airline Code_JQ', 'Airline Code_JU',
        'Airline Code_JY', 'Airline Code_K6', 'Airline Code_KC', 'Airline Code_KE', 'Airline Code_KL',
        'Airline Code_KM', 'Airline Code_KP', 'Airline Code_KQ', 'Airline Code_KR', 'Airline Code_KU',
        'Airline Code_L6', 'Airline Code_LA', 'Airline Code_LG', 'Airline Code_LH', 'Airline Code_LO',
        'Airline Code_LX', 'Airline Code_LY', 'Airline Code_ME', 'Airline Code_MF', 'Airline Code_MH',
        'Airline Code_MK', 'Airline Code_MS', 'Airline Code_MU', 'Airline Code_NF', 'Airline Code_NH',
        'Airline Code_NK', 'Airline Code_NZ', 'Airline Code_OB', 'Airline Code_OD', 'Airline Code_OK',
        'Airline Code_OM', 'Airline Code_OR', 'Airline Code_OS', 'Airline Code_OU', 'Airline Code_OZ',
        'Airline Code_P0', 'Airline Code_P4', 'Airline Code_PC', 'Airline Code_PG', 'Airline Code_PK',
        'Airline Code_PR', 'Airline Code_PU', 'Airline Code_PW', 'Airline Code_PX', 'Airline Code_PY',
        'Airline Code_QF', 'Airline Code_QR', 'Airline Code_QS', 'Airline Code_QV', 'Airline Code_RJ',
        'Airline Code_RN', 'Airline Code_RO', 'Airline Code_RQ', 'Airline Code_SA', 'Airline Code_SB',
        'Airline Code_SC', 'Airline Code_SG', 'Airline Code_SK', 'Airline Code_SL', 'Airline Code_SN',
        'Airline Code_SQ', 'Airline Code_SS', 'Airline Code_SV', 'Airline Code_TC', 'Airline Code_TG',
        'Airline Code_TK', 'Airline Code_TM', 'Airline Code_TP', 'Airline Code_TR', 'Airline Code_TU',
        'Airline Code_UA', 'Airline Code_UB', 'Airline Code_UK', 'Airline Code_UL', 'Airline Code_UP',
        'Airline Code_UR', 'Airline Code_UU', 'Airline Code_UX', 'Airline Code_VA', 'Airline Code_VB',
        'Airline Code_VF', 'Airline Code_VN', 'Airline Code_VS', 'Airline Code_VY', 'Airline Code_W2',
        'Airline Code_W3', 'Airline Code_WB', 'Airline Code_WF', 'Airline Code_WM', 'Airline Code_WS',
        'Airline Code_WY', 'Airline Code_X1', 'Airline Code_XY', 'Airline Code_ZB', 'Airline Code_ZH',
        'Airline Code_ZL', 'Airline Code_ZP', 'Origin Continent_Africa', 'Origin Continent_Asia',
        'Origin Continent_Europe', 'Origin Continent_North America', 'Origin Continent_Oceania',
        'Origin Continent_South America', 'Destination Continent_Africa', 'Destination Continent_Asia',
        'Destination Continent_Europe', 'Destination Continent_North America', 'Destination Continent_Oceania',
        'Destination Continent_South America', 'Flight_Distance_Category_Long Haul',
        'Flight_Distance_Category_Medium Haul', 'Flight_Distance_Category_Short Haul',
        'Flight_Duration_Category_Long', 'Flight_Duration_Category_Medium', 'Flight_Duration_Category_Short',
        'Days_Until_Flight', 'Extra_Long_Flight'
    ]]

    input_df = input_df.select_dtypes(include=[int, float])
    y_pred = modelo.predict(input_df)

    return y_pred

# Streamlit app
st.title("Prediccion de precio de vuelos")

# Input fields
Hour_of_Departure = st.number_input("Hora", min_value=0, max_value=23, value=14)
airline = st.text_input("Codigo de aerolinea", value='UX')
Stops = st.number_input("Numero de Paradas", min_value=0, max_value=5, value=2)
Origin_Country = st.text_input("Pais de origen", value='Japan')
Destination_Country = st.text_input("Pais de destino", value='Congo')
day = st.number_input("Dia de salida", min_value=1, max_value=31, value=9)
month = st.number_input("Mes de salida", min_value=1, max_value=12, value=8)
year = st.number_input("Año de salida", min_value=2023, max_value=2100, value=2025)



# Predict button
if st.button("Obten tu prediccion"):
    predicted_price = predecir(Hour_of_Departure, airline, Stops, Origin_Country, Destination_Country, day, month, year)
    st.write(f"El precio probable es de : ${predicted_price[0]:.2f}")



