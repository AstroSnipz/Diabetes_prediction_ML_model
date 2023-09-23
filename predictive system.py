import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('deploy_ml_model/trained_model.sav', 'rb')) 

# making a predictive system
input_data = (5,117,92,0,0,34.1,0.337,38)

# changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)                   # here we didnt use StandardScalar -->  note video
print(prediction)

if prediction == 0:
    print("The person in not diabetic")
else:
    print("The person is diabetic")
