import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
dt_model = pickle.load(open('Decision_Tree.pkl', 'rb'))
rf_model = pickle.load(open('Random_Forest.pkl', 'rb'))
mlp_model = pickle.load(open('MLP.pkl', 'rb'))
boosted_model = pickle.load(open('Boosted_model.pkl', 'rb'))

# Define the function to preprocess input data
def preprocess_data1(new_data1):
 # Your median values data
 median_values_data = {'TSH': 1.40, 'T3': 1.90, 'TT4': 103.00, 'T4U': 0.96, 'FTI': 108.00}

 # Create a pandas Series
 median_values = pd.Series(median_values_data)

 # Update the new_data1 for None
 for key in new_data1:
    if key == 'sex' and new_data1[key] == None:
        new_data1[key] = 'F'

 for key in new_data1:
    if key in median_values.index and new_data1[key] == None:
        # It's a numerical feature
        new_data1[key] = median_values[key]
    elif key not in median_values.index and new_data1[key]== None:
        # It's a categorical feature, update the corresponding binary columns
        new_data1[key] = 'F'

 return new_data1




# Define the function to preprocess input data
def preprocess_data2(new_data):
 # Your median values data
 median_values_data = {'TSH': 1.40, 'T3': 1.90, 'TT4': 103.00, 'T4U': 0.96, 'FTI': 108.00}

 # Create a pandas Series
 median_values = pd.Series(median_values_data)

 # The columns list of X_train
 columns_list = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'sex_F', 'sex_M',
                'thyroid_surgery_f', 'thyroid_surgery_t', 'on_thyroxine_f',
                'on_thyroxine_t', 'sick_f', 'sick_t', 'goitre_f', 'goitre_t',
                'psych_f', 'psych_t']
 
 # Create a DataFrame for the new data point with the same columns as X_train, initialized to 0
 new_X = pd.DataFrame(0, index=[0], columns=columns_list)

 # Update the DataFrame with the new data
 for key in new_data:
    if key in median_values.index:
        # It's a numerical feature
        new_X[key] = new_data[key]
    else:
        # It's a categorical feature, update the corresponding binary columns
        new_X[key + '_f'] = 1 if new_data[key] == 'f' else 0
        new_X[key + '_t'] = 1 if new_data[key] == 't' else 0

 # Fill missing numerical features with the median values from the original data
 for column in median_values.index:
    if column not in new_data:
        new_X[column] = median_values[column]
        
 # Fill missing categorical features with f which is the highest in original data i.e. mostly negatives
 for column in new_X.columns:
    if column[:-2] not in new_data:
        if column.endswith("_f"):
            new_X[column] = 1

 # Handle the 'sex' feature separately as it's not binary in the original data
 if new_data.get('sex') == None:
    new_X['sex_F'] = 1 
    new_X['sex_M'] = 0
 else:
    new_X['sex_F'] = 1 if new_data.get('sex') == 'F' else 0
    new_X['sex_M'] = 1 if new_data.get('sex') == 'M' else 0
    
 # Ensure the order of columns matches that of X_train
 new_X = new_X.reindex(columns=columns_list)

 return new_X


# Define the function to predict thyroid
def predict_thyroid(new_X):
 

 ensemble_preds_new = boosted_model.predict(new_X)

 return ensemble_preds_new


 
def main():
     st.subheader("Thyroid Classification")
     st.write("Thyroid disease is a significant global health concern, affecting millions of people worldwide. The thyroid gland, a vital organ in our body, plays a crucial role in metabolism, growth, and development. It produces two main hormones, thyroxine (T4) and triiodothyronine (T3), which are essential for the body's metabolic processes. The production of these hormones is regulated by thyroid-stimulating hormone (TSH), which is released by the pituitary gland. An imbalance in these hormones can lead to thyroid diseases.")
     st.write("Hypothyroidism is a condition where the thyroid gland does not produce enough thyroid hormones, leading to symptoms like fatigue, weight gain, and depression. On the other hand, hyperthyroidism is a condition where the thyroid gland produces too much thyroid hormones, leading to symptoms like rapid heart rate, weight loss, and anxiety.")
     
     st.write("Identifying and treating thyroid conditions is crucial for maintaining overall health and preventing complications related to metabolism, energy levels, and various organ functions.")

     # Display normal range table
     st.write("**Normal ranges:**")
     normal_range = {'TSH': '0.5 - 5.0 mIU/L or µIU/mL', 'T3': ' 0.6 - 1.81 ng/mL', 'T4': ' 45 - 130 ng/mL' , 'T4U': ' 0.7 - 1.2', 'FTI': ' 53 - 142' }
     df_range = pd.DataFrame([normal_range])
     st.table(df_range)

     # Add a divider
     st.markdown("<hr>", unsafe_allow_html=True)

      # Create input fields for user input
     st.markdown("**Enter the required information**")
     # age
     age = st.number_input("*Enter your Age",value= None, min_value=0, max_value=102, step=1)

     # sex
     sex = st.radio("*Select Gender", ['Male', 'Female'], index=None)
     sex_mapping = {'Male': 'M', 'Female': 'F'}
     sex = sex_mapping.get(sex, sex)

     #TSH
     TSH = st.number_input("*Enter TSH ( µIU/mL)", value= None, min_value= 0.00, step=0.01)
   
     #T3
     T3 = st.number_input("*Enter T3 ( ng/mL)", value= None, min_value= 0.00, step=0.01)
     
     #T4
     TT4 = st.number_input("*Enter T4 ( ng/mL)", value= None, min_value= 0.0, step=0.1)
     
     
     #T4U
     T4U = st.number_input(" Enter T4U", value= None, min_value= 0.0, step=0.01)

     #FTI
     FTI = st.number_input(" Enter FTI/FT4", value= None, min_value= 0.0, step=0.1)    
     
     
     thyroid_surgery = st.radio("Had Thyroid Surgery before?", ['Yes', 'No'], index= None)
     thyroid_surgery_mapping = {'Yes': 't', 'No': 'f'}
     thyroid_surgery = thyroid_surgery_mapping.get(thyroid_surgery, thyroid_surgery)
     
     on_thyroxine = st.radio("Do you take thyroxine sodium?", ['Yes', 'No'], index=None)
     on_thyroxine_mapping = {'Yes': 't', 'No': 'f'}
     on_thyroxine = on_thyroxine_mapping.get(on_thyroxine, on_thyroxine)
     
     sick = st.radio("Are you sick right now?", ['Yes', 'No'], index=None)
     sick_mapping = {'Yes': 't', 'No': 'f'}
     sick= sick_mapping.get(sick,sick)

     psych = st.radio("Do you psychological effects of thyroid conditions?", ['Yes', 'No'], index=None)
     psych_mapping = {'Yes': 't', 'No': 'f'}
     psych = psych_mapping.get(psych,psych)

     goitre = st.radio("Do you have Goitre?", ['Yes', 'No'], index=None)
     goitre_mapping = {'Yes': 't', 'No': 'f'}
     goitre= goitre_mapping.get( goitre, goitre)
    
     # Making a dictionary for entered datas
     entry_data={'age': age, 'sex': sex, 'TSH': TSH, 'T3': T3, 'TT4': TT4, 'T4U': T4U, 'FTI': FTI, 
                 'thyroid_surgery': thyroid_surgery, 'on_thyroxine': on_thyroxine, 'sick': sick, 
                 'psych': psych, 'goitre': goitre}
     
     # Create three columns
     col1, col2, col3 = st.columns([1,1,1])

     # Put the button in the middle column
     with col2:
        # Operation when pressed check
        if st.button("Check", type= "primary"):
            if TSH is None or T3 is None or TT4 is None or age is None or sex is None:
                st.warning("Please enter values, at least, for: age, Gender, TSH, T3 and T4")
            else:
                new_data= dict()
                new_data = {k: v for k, v in entry_data.items() if v is not None or v is None}
                new_data = preprocess_data1(new_data)
                new_X = preprocess_data2(new_data)
                prediction = predict_thyroid(new_X)

                class_mapping = {0: 'Hyperthyroidism', 1: 'Hypothyroidism', 2: 'Normal'}

                # Map the numeric prediction to the corresponding class label
                prediction_label = class_mapping.get(prediction[0], 'Unknown')

                # Display the prediction
                if prediction_label == 'Normal':
                    st.write("The person is likely to be ", prediction_label)
                else:
                   st.write("The person is likely to have ", prediction_label )

      # Add a divider
     st.markdown("<hr>", unsafe_allow_html=True)


     if st.button("How this was prepared?"):

         st.write("The Models were evaluated on dataset consisting of 7542 samples with thirteen attributes: age, sex, on_thyroxine, sick, thyroid_surgery, goitre, psych, TSH, T3, TT4, T4U, FTI, target")
         st.write("The training set for the models were balanced beforehand to better represent minority classes: which are the positive cases of thyroid")
         st.write("On conducting Four different supervised learning algorithms: Decision Tree,Random Forest, MLP performed well, while SVM was unimpressive. Hence, this app is built using MaxVoting classifier ensembling earlier, three models for Thyroid Classification.")
         
        

# Run the app
if __name__ == '__main__':
    main()




             
                     
                     

         