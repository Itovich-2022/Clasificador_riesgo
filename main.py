import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
# secciones de la aplicacion
header = st.container()
dataset= st.container()
features = st.container()
model= st.container()
#no me funciono la siguiente funcion
@st.cache
def get_data(filename):
    data=pd.read_csv(filename, sep =';', encoding = 'utf_8')
    return data
#encabezado
with header:
    st.title('Clasificador binario de hipertensión arterial para mayores de 40 años')
    st.text('Determina el riesgo de padecer hipertensión a partir de ciertas variables clínicas')
    
# base de datos de entranamiento
with dataset:
    st.header('Dataset Salud Pública NQN anonimizado filtrado por edad y variables predictoras')
    st.text('Aprox. 8000 registros de consultas de clinica médica durante 2019-2021')   
    
#  data=get_data('df.csv')
    data=pd.read_csv('df.csv', sep =';', encoding = 'utf_8')
    #st.write(type(data))
    X = data.drop('HIPERTENSION_SI', axis = 1)
    y = data.HIPERTENSION_SI
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0, stratify = y)
#    st.write(X.shape)
    X_train=X_train.iloc[:,1:]
#   st.write('nuevo', X_train.shape)
    #st.write(X_test.shape)
#   st.write(df.columns[0:9])
#    st.write(df.columns[9:16])
  #  st.write(X_train.columns[0:9])
#    st.write(X_train.columns[9:17])
#     st.write(df.head(2))
#     st.write(X_train.head(2))
    
 # insertar visualizacion     
    st.subheader('Distribución de consultas por edad')
    edad= pd.DataFrame(data.EDAD.value_counts()).head(40)
    st.bar_chart(edad)
    
# variables predictoras
with features:
    st.header('Variables consideradas para la predicción')
    st.markdown('*Se debe ingresar una serie de datos que son la base de la predicción*')
    
# formulario de informacion para la prediccion de riesgo   
with st.form('Formulario'):

    sel_col, disp_col=st.columns(2)
    
    TAS = sel_col.slider('Tension arterial alta', min_value=40, max_value=240, value=50)
    TAD = sel_col.slider('Tensión arterial baja', min_value=20, max_value=200, value=50)
    COLESTEROL_TOTAL = sel_col.slider('Colesterol Total', min_value=100, max_value=400, value=110)
    CLEARANCE = sel_col.slider('Clearance', min_value=50, max_value=180, value=60)
    EDAD = sel_col.slider('Edad', min_value=0, max_value=105, value=10)
    UTMO_IMC = sel_col.slider('Ultimo Indice de Masa Corporal (IMC)', min_value=15, max_value=45, value=17)
    SEXO_MASCULINO = sel_col.selectbox('Para Sexo masculino elija 1, para Sexo femenino elija 0', options=[0,1])
    input_feature = sel_col.text_input('Escribe tu nombre','')
    OBESIDAD_SI = disp_col.selectbox('Para obesidad SI elija 1, para Obesidad NO elija 0', options = [0,1])
    DIABETES_NO = disp_col.selectbox('Para NO diabético elija 1, para diabético elija 0', options = [0,1])
    DIABETES_DM2 = disp_col.selectbox('Para diabético DM2 elija 1, para diabetico DM1 (o si es NO diabético) elija 0', options = [0,1])
    DISLIPEMIA_SI = disp_col.selectbox('Para dislipemia elija 1, caso contrario elija 0', options = [0,1])
    ENF_CARDIO_ESTABLECIDA_SI = disp_col.selectbox('Para Enfermedad Cardiovascular establecida elija 1, caso contrario elija 0', options = [0,1])
    CLASIFICACION_NORMAL = disp_col.selectbox('Para 20 < IMC <= 25, elija 1', options = [0,1])
    CLASIFICACION_SOBREPESO = disp_col.selectbox('Para 25< IMC <= 30, elija 1', options = [0,1])
    CLASIFICACION_OBESIDAD = disp_col.selectbox('Para 30 < IMC, elija 1', options = [0,1])
    
    
    submitted=st.form_submit_button('Submit')
    
  
    
features=[TAS, TAD, UTMO_IMC, CLEARANCE, COLESTEROL_TOTAL, EDAD, SEXO_MASCULINO, DIABETES_DM2, DIABETES_NO, DISLIPEMIA_SI, OBESIDAD_SI, CLASIFICACION_NORMAL, CLASIFICACION_OBESIDAD, CLASIFICACION_SOBREPESO, ENF_CARDIO_ESTABLECIDA_SI]
# st.write(len(features))

# st.write(features[0])
# escalado de features
features=[(features[i-1]-X_test.mean()[i])/(X_test.std()[i]) for i in range(1,len(features)+1)]
# st.write(features[0])
# st.write(X_test.mean()[1:2])
# st.write(X_test.std()[1:2])
# st.write(features)

with model:
    st.header('Modelo de predición')
    st.markdown('*AdaBoosting Classifier*') 
    
    
with open ('./ABoost.pkl', 'rb') as f_ABoost:
        modelo_ABoost = pickle.load(f_ABoost)
        
def predict(features):
    
  
    final = np.reshape(features, (1, -1))
    
    return modelo_ABoost.predict(final)

image=Image.open('corazon_rojo.jpg')
st.image(image)

pred=predict(features)

if pred == [1]:
    st.write('En base a los datos que ingresaste,', input_feature, ', el test ha arrojado que tienes riesgo de hipertensión arterial. Consulta a un médico')
else:
    st.write('En base a los datos que ingresaste,', input_feature, ', el test ha arrojado que no tienes riesgo de hipertensión arterial. Continua con buenos hábitos de comida y actividad física!')
            

            
            

            
       
       
  