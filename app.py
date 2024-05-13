import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor

# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Load data from the pickle file
df = pd.read_pickle('C:/Users/HP/CampusX Projects/Laptop Price Predictor/df.pkl')

st.title('Laptop Price Predictor')

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# RAM
RAM = st.selectbox('RAM(in GB)', [2,4,6,8,12,16,24,32,64])

# Weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen Size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800',
                                                '2560x1440', '2304x1440'])

# CPU
cpu = st.selectbox('Brand', df['CPU Brand'].unique())

# HDD
hdd = st.selectbox('HDD(in GB)', [0,128,256,512,1024,2048])

# SSD
ssd = st.selectbox("SSD(in GB)", [0,8,128,256,512,1024])

# GPU
gpu = st.selectbox("GPU", df['Gpu Brand'].unique())

# OS
os = st.selectbox('OS', df['OS'].unique())

if st.button('Predict Price'):

    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2)) ** 0.5 / screen_size

    query = np.array([company, type, RAM, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1,12)
    st.title('The Predicted Price of this configuration is: ' + str(int(np.exp(pipe.predict(query)[0]))))
