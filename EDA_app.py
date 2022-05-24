import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


# Web app title
st.markdown(''' # **EDA web app**
This app is developed by me called **EDA app**
''')
# upload file from pc
with st.sidebar.header("Upload dataset(.csv)"):
    uploaded_file =  st.sidebar.file_uploader("upload file", type=['csv'])
    df = sns.load_dataset('titanic')
    st.sidebar.mar
    
# profilinng report for pandas
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv

    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**inputDF**')
    st.write(df)
    st.write('...')
    st.header("**Profile report with pd**")
    st_profile_report(pr)
else:
    st.info('Awaiting for csv file')
    if st.button('Press to use example data'):
    # example dataset 
     @st.cache
     def load_data():
          a = pd.DataFrame(np.random.rand(100, 4),
                        columns=['age','banana','codanics','Eye'])
          return a      
     df=load_data()
    pr = ProfileReport(df, explorative=True)
    st.header('**inputDF**')
    st.write(df)
    st.write('...')
    st.header("**Profile report with pd**")
    st_profile_report(pr)