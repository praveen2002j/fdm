import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('new_passengers.csv')
    return data

# Load the models
def load_model(model_name):
    return joblib.load(model_name)

# Define the Streamlit app
st.title("Airline Passenger Satisfaction Prediction System")

# Sidebar options
st.sidebar.header("Options")
selected_option = st.sidebar.selectbox("Choose an option", ["View Data", "Make Prediction"])

# Load data
data = load_data()

# Separate features and target variable
X = data.drop(columns=['satisfaction'])
y = data['satisfaction']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# View Data option
if selected_option == "View Data":
    st.subheader("Dataset: Airline Passenger Satisfaction")
    
    st.image('https://w7.pngwing.com/pngs/902/845/png-transparent-airplane-aircraft-flight-airplane-mode-of-transport-flight-airplane-thumbnail.png', use_column_width=True, caption='See the passenger satisfaction')
    # Display data
    st.write(data.head())

    # Display dataset summary
    st.write("Summary Statistics:")
    st.write(data.describe())

    # Display charts for exploration
    st.subheader("Class Distribution")
    st.bar_chart(data['satisfaction'].value_counts())

# Prediction option
elif selected_option == "Make Prediction":
    st.subheader("Make a Prediction")
    
    # Load models (ensure these files exist in the same directory)
    model_catboost = load_model('catboost_model.joblib')
    model_lightgbm = load_model('lightgbm_model.joblib')
    
    # Feature inputs (based on the dataset)
    st.sidebar.subheader("Input Features")
    gender = st.sidebar.selectbox("Gender (0: Female, 1: Male)", [0, 1])
    age = st.sidebar.slider("Age", 18, 80, 30)
    customer_type = st.sidebar.selectbox("Customer Type (0: First Time, 1: Returning)", [0, 1])
    travel_type = st.sidebar.selectbox("Travel Type (0: Personal, 1: Business)", [0, 1])
    flight_class = st.sidebar.selectbox("Class (0: Business, 1: Economy, 2: Economy Plus)", [0, 1, 2])
    distance = st.sidebar.slider("Flight Distance", 0, 5000, 1000)
    departure_delay_minutes = st.sidebar.slider("Departure Delay Minutes", 0, 180, 0)
    arrival_delay_minutes = st.sidebar.slider("Arrival Delay Minutes", 0, 180, 0)
    dep_val_time_convenient = st.sidebar.slider("Departure Valuation Time Convenient (0: No, 1: Yes)", 0, 1, 1)
    online_booking_service = st.sidebar.slider("Online Booking Service Rating (1-5)", 1, 5, 3)
    checkin_service = st.sidebar.slider("Check-in Service Rating (1-5)", 1, 5, 3)
    online_boarding = st.sidebar.slider("Online Boarding Rating (1-5)", 1, 5, 3)
    gate = st.sidebar.slider("Gate Rating (1-5)", 1, 5, 3)
    onboard_service = st.sidebar.slider("Onboard Service Rating (1-5)", 1, 5, 3)
    seat_comfort = st.sidebar.slider("Seat Comfort Rating (1-5)", 1, 5, 3)
    leg_room_service = st.sidebar.slider("Leg Room Service Rating (1-5)", 1, 5, 3)
    cleanliness = st.sidebar.slider("Cleanliness Rating (1-5)", 1, 5, 3)
    food_drink = st.sidebar.slider("Food & Drink Rating (1-5)", 1, 5, 3)
    inflight_service = st.sidebar.slider("Inflight Service Rating (1-5)", 1, 5, 3)
    wifi_service = st.sidebar.slider("WiFi Service Rating (1-5)", 1, 5, 3)
    baggage_handling = st.sidebar.slider("Baggage Handling Rating (1-5)", 1, 5, 3)

    # User can select which model to use for prediction
    selected_model = st.selectbox("Choose a model", ["CatBoost", "LightGBM"])

    # Input features into a DataFrame for prediction
    input_data = pd.DataFrame({
        'gender': [gender],  
        'age': [age],  
        'customer_type': [customer_type],  
        'travel_type': [travel_type],  
        'class': [flight_class],  
        'distance': [distance],  
        'departure_delay_minutes': [departure_delay_minutes],  
        'arrival_delay_minutes': [arrival_delay_minutes],  
        'dep_val_time_convenient': [dep_val_time_convenient],  
        'online_booking_service': [online_booking_service],  
        'checkin_service': [checkin_service],  
        'online_boarding': [online_boarding],  
        'gate': [gate],  
        'onboard_service': [onboard_service],  
        'seat_comfort': [seat_comfort],  
        'leg_room_service': [leg_room_service],  
        'cleanliness': [cleanliness],  
        'food_drink': [food_drink],  
        'inflight_service': [inflight_service],  
        'wifi_service': [wifi_service],  
        'baggage_handling': [baggage_handling],  
        'entertainment': [0],  # Fill with default or mean value
        # Add other missing features here if necessary
    }, index=[0])  # Ensure the input_data is a DataFrame with one row

    # Ensure the input_data has the same columns as the training data
    input_data = input_data.reindex(columns=X.columns, fill_value=0)  # Fill missing features with default values

    # Apply scaling (use the same scaler as used in training)
    input_data_scaled = scaler.transform(input_data)

    # #Perform the prediction
    if st.button("Predict Satisfaction"):
        if selected_model == "CatBoost":
            prediction = model_catboost.predict(input_data_scaled)
        else:
            prediction = model_lightgbm.predict(input_data_scaled)

       
     # Display the result
        st.write(f"Prediction: {'Satisfied' if prediction[0] == 1 else 'Not Satisfied'}")
