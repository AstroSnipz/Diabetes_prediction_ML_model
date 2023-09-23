import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('deploy_ml_model/trained_model.sav', 'rb')) 

#creating a function for prediction
def diabetes_prediction(input_data):

    # changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)                   # here we didnt use StandardScalar -->  note video
    print(prediction)

    if prediction == 0:
        return "The person in not diabetic"
    else:
        return "The person is diabetic"
    
def main():
    # giving a title
    st.title("Diabetes Prediction Web App")

    # getting the input
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("Blood Pressure value")
    SkinThickness = st.text_input("skin thickness value")
    Insulin = st.text_input("insulin level")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    Age = st.text_input("Age of the person")

    # code for prediction
    diagnosis = ''   # the output of the prediction will be stored here

    # creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)




if __name__ == '__main__':
    main()