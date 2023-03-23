class Stock:
      def dfClose():
        st.subheader('closing price vs time chart')
        fig=plt.figure(figsize=(12,6))
        plt.plot(df.Close,'b',label='Closing price')
        plt.xlabel('Time Period')
        plt.ylabel('Price in rupees')
        #plt.plot(df.Close)
        plt.legend()
        st.pyplot(fig)
      def ExponentialAverage():
        st.subheader('ExponentialAverage')
        ma7=df.Close.ewm(span=7).mean()
        ma21=df.Close.ewm(span=21).mean()
        ma50=df.Close.ewm(span=50).mean()
        ma100=df.Close.ewm(span=100).mean()
        ma200=df.Close.ewm(span=200).mean()
        fig1=plt.figure(figsize=(12,6))
        plt.xlabel('Time Period')
        plt.ylabel('Price in rupees')
        plt.plot(df.Close,'b',label='Closing price')
        plt.plot(ma7,'r',label='ewm=7')
        plt.plot(ma21,'g',label='ewm=21')
        plt.plot(ma50,'m',label='ewm=50')
        plt.plot(ma100,'y',label='ewm=100')
        plt.plot(ma200,'k',label='ewm=200')
        plt.legend()
        st.pyplot(fig1)
      def SuperTrend():
        st.subheader('SuperTrend')
        ta.supertrend(high=df.High,low=df.Low,close=df.Close,period=20,multiplier=2)
        df['Sup']=ta.supertrend(high=df.High,low=df.Low,close=df.Close,period=20,multiplier=2)['SUPERT_7_2.0']
        fig2=plt.figure(figsize=(12,6))
        plt.xlabel('Time Period')
        plt.ylabel('Price in rupees')
        plt.plot(df['Close'])
        plt.plot(df['Sup'])
        st.pyplot(fig2)
      def MovingAverage():
         st.subheader('MovingAverage')
         fast=100
         slow=200
         mov_avg_fast=df.Close.rolling(fast).mean()
         mov_avg_slow=df.Close.rolling(slow).mean()
         fig3=plt.figure(figsize=(12,6))
         plt.xlabel('Time Period')
         plt.ylabel('Price in rupees')
         plt.plot(df.Close,'b',label="df_close")
         plt.plot(mov_avg_fast,'g',label="mov_avg_fast")
         plt.plot(mov_avg_slow,'r',label="mov_avg_slow")
         plt.legend()
         st.pyplot(fig3)
      def Vol_Wht_Avg_Prc():
        st.subheader('Volume Weight Average Price')
        df['vwap']=ta.vwap(high=df.High,low=df.Low,close=df.Close,volume=df.Volume)
        fig4=plt.figure(figsize=(12,6))
        plt.xlabel('Time Period')
        plt.ylabel('Price in rupees')
        plt.plot(df.vwap,'g',label="VWAP")
        plt.legend()
        st.pyplot(fig4)
      def prediction():            
            # spliting data into training and testing
            data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.75)])
            data_testing=pd.DataFrame(df['Close'][int(len(df)*0.75):])
            from sklearn.preprocessing import MinMaxScaler
            scaler=MinMaxScaler(feature_range=(0,1))
            data_training_array=scaler.fit_transform(data_training)
            # spliting data into training and testing
            data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.75)])
            data_testing=pd.DataFrame(df['Close'][int(len(df)*0.75):])
            from sklearn.preprocessing import MinMaxScaler
            scaler=MinMaxScaler(feature_range=(0,1))
            data_training_array=scaler.fit_transform(data_training)         
            #load my model
            model=load_model('keras_model.h5')
            #testing
            past_100_days=data_training.tail(100)
            final_df=past_100_days.append(data_testing,ignore_index=True)
            input_data=scaler.fit_transform(final_df)
            x_test=[]
            y_test=[]
            for i in range(100,input_data.shape[0]):
                x_test.append(input_data[i-100:i])
                y_test.append(input_data[i,0])
            x_test, y_test = np.array(x_test), np.array(y_test)
            y_predicted=model.predict(x_test)
            scal=scaler.scale_
            scaler_factor=1/scal
            y_predicted=y_predicted*scaler_factor
            y_test=y_test*scaler_factor         
            #final graph
            st.subheader('Prediction vs Orignal')
            fig5=plt.figure(figsize=(12,6))
            plt.plot(y_test,'b',label='Orignal price')
            plt.plot(y_predicted,'r',label='Predicted price')
            plt.xlabel('Time Period')
            plt.ylabel('Price in rupees')
            plt.legend()
            st.pyplot(fig5)
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fix_yahoo_finance as fyf
import pandas_ta as ta
from keras.models import load_model
st.title("stock Analysis")
user_input=st.text_input("enter the proper ticker",'SBIN.NS')
df=fyf.download(user_input)
Stock.dfClose()
Stock.ExponentialAverage()
Stock.SuperTrend()
Stock.MovingAverage()
Stock.Vol_Wht_Avg_Prc()
Stock.prediction()









