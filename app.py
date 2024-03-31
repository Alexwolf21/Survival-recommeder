import streamlit as st
import pandas as pd
import pickle

# Load the trained model
@st.cache
def load_model():
    with open('logistic_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Define function to make predictions
def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })
    return model.predict(input_data)[0]

# Load the model
model = load_model()

# Custom CSS for styling
st.markdown("""
    <style>
        /* Center-align text */
        .center {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        /* Splash screen animation */
        @keyframes textAnimation {
            0% {
                opacity: 0;
                transform: translateY(-50px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .splash-screen h1, .splash-screen p, .input-content {
            animation: textAnimation 2s;
        }
    </style>
""", unsafe_allow_html=True)

# Splash screen with animation
st.markdown("""
    <div class="center splash-screen">
        <h1>Titanic</h1>
        <p>Made with Streamlit</p>
    </div>
""", unsafe_allow_html=True)

# Input fields with animation
with st.container() as input_container:
    st.markdown("<h2 class='input-content'>Model Inputs</h2>", unsafe_allow_html=True)
    pclass = st.selectbox('Passenger Class', [1, 2, 3])
    sex = st.selectbox('Sex', ['Male', 'Female'])
    age = st.slider('Age', 0, 100, 30)
    sibsp = st.slider('Number of Siblings/Spouses Aboard', 0, 10, 0)
    parch = st.slider('Number of Parents/Children Aboard', 0, 10, 0)
    fare = st.slider('Fare', 0, 600, 50)
    embarked = st.selectbox('Embarked', ['C', 'Q', 'S', 'U'])

# Map select box choices to numerical values
sex = 1 if sex == 'Female' else 0
embarked_mapping = {'C': 0, 'Q': 1, 'S': 2, 'U': 3}
embarked = embarked_mapping[embarked]

# Predict button
if st.button('Predict'):
    # Make prediction using the loaded model
    prediction = predict_survival(pclass, sex, age, sibsp, parch, fare, embarked)
    if prediction == 0:
        st.write('Unfortunately, the model predicts that the person did not survive.')
    else:
        st.write('Congratulations! The model predicts that the person survived.')
