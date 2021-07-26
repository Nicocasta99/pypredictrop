# Tratamiento de datos
# ==============================================================================
import os
import pandas as pd
import numpy as np

import joblib
import pickle 
import streamlit as st 
import sklearn
from pycaret.regression import *

#cargar modelos 
xg_boost=joblib.load("xgboost_prueba1.pkl")
knn_test_xg=joblib.load("Final xgboost Model 25jul2021.pkl")

#main 
def main(): 
    #Titulo del sitio
    st.title('ROP predictor_ test1')

    #Sidebar 
    st.sidebar.header('Choose parameters')

    #funcion para poner los parametros en el sidebar
    def user_input_parameters():

        
        #ROP_sd = st.sidebar.slider('ROP', 0.0, 110.3, 3.5)
        BIT_DEPTH = st.sidebar.slider('depth', 16193, 18487, 17422)
        HKLD = st.sidebar.slider('Hook Load',295.1, 488.0, 449.1)
        WOB = st.sidebar.slider('WOB', 0.0, 414.0, 16.0)
        TORQUE = st.sidebar.slider('Torque', 0.0, 27.0, 15.5)
        BIT_RPM = st.sidebar.slider('Bit RPM', 0, 1822, 345)
        PUMP = st.sidebar.slider('Pump', 1010, 4561, 2652)
        FLOW_OUT_PC = st.sidebar.slider('Flow out', 0.0, 73.0, 31.5)
        FLOW_IN = st.sidebar.slider('Flow in', 0, 15371, 440)
        OVERBALANCE = st.sidebar.slider('Overbalance', 2.6, 5.6, 3.9)
        dT = st.sidebar.slider('Temperture Diference', -70, 92, 22)
        ROT_TIME = st.sidebar.slider('Rotary Time', 0.0, 147.0, 44.5)
        DESGASTE = st.sidebar.slider('Desgaste', 0, 8, 3)
        
        data = {#'ROP_sd': ROP_sd,
                'BIT_DEPTH': BIT_DEPTH,
                'HKLD': HKLD,
                'WOB': WOB,
                'TORQUE': TORQUE,
                'BIT_RPM': BIT_RPM,
                'PUMP': PUMP,
                'FLOW_OUT_PC': FLOW_OUT_PC,
                'FLOW_IN': FLOW_IN,
                'OVERBALANCE': OVERBALANCE,
                'dT': dT,
                'ROT_TIME': ROT_TIME,
                'DESGASTE': DESGASTE,
                }
        features = pd.DataFrame(data, index=[0])
        return features
    df = user_input_parameters()

    #escritura de parametros seleccionados en la pagina
    st.subheader('User Input Parameters')
    st.write(df)

    #escoger el modelo preferido
    option = ['K Neighbors', 'Xg boosting']
    model = st.sidebar.selectbox('Which model you like to use?', option)
    
    st.subheader('Model applied')
    st.write(model)
    
    if st.button('RUN'):
        if model == 'Xg boosting':

            np1 = predict_model(xg_boost, data=df)
            np1=np1[['Label']]
            st.title("La ROP esperada es de:")
            st.write(np1)
            #st.success(np1)
            st.write("ft/hr")            

        elif model == 'K Neighbors':
            np2 = predict_model(knn_test_xg, data=df)
            np2=np2[['Label']]
            st.title("La ROP esperada es de:")
            st.write(np2)
            #st.success(np2)
            st.write("ft/hr") 

        

    #Final 
if __name__ == '__main__':
    main()

