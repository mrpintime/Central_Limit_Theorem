#Central Limit Theorem

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st



st.title('Central Limit Theorem (CLT)') #todo: add icon and header animation here
st.header('Theorem Description')

Theorem_Description = """
The central limit theorem states that if you have a population with mean $$μ$$ and standard deviation $$σ$$ and take 
sufficiently large random samples from the population with replacement, then the distribution of the sample means will be approximately normally distributed.
\n check out this: [Boston University](https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_probability/BS704_Probability12.html)

"""

st.info(Theorem_Description)
st.header('App Description')
st.write("""
Here we try to use statistics and generating proccess to experiment Central Limit Theorem using power of python.
We have Four steps:""")
st.markdown("""
         1. Choose your distribution for sampling
         2. Define size of your sample
         3. Sample from your distribution several times
         4. Generate histogram from your new distibution by click on `Start Simulation` button 
         """)

st.header('Business Usage')
st.markdown("""
> **Hypothesis Testing**: CLT is often used in hypothesis testing. When dealing with sample means, you can assume that the sampling distribution of the mean is approximately normal. This allows you to perform hypothesis tests and calculate confidence intervals.

> **Quality Control**: In manufacturing and production processes, companies often take random samples of products to assess quality. By applying the CLT, they can make inferences about the population's quality based on the sample means.

> **A/B Testing**: In digital marketing, A/B testing is used to compare the performance of two or more versions of a webpage, advertisement, or product feature. The CLT is used to analyze the results and determine if there is a statistically significant difference between the groups.

> **Risk Assessment**: In risk management, businesses assess various risks, such as operational, market, and credit risks. The CLT can be used to model the distribution of potential losses and make risk management decisions. 

Some extra usages are :
**Inventory Management**, **Credit Scoring**, **Customer Satisfaction Surveys**, **Market Research**
            
In all these cases, **The Central Limit Theorem** provides a solid foundation for making statistical inferences and decisions based on sample data. It allows businesses to draw conclusions about populations, make predictions, and make data-driven decisions, which are crucial for effective operations and decision-making.
""")



def simulate(sample_size: int, number: int, dist=list) -> list:
    """
    dist = your distribution
    sample_size = size of each random sampling proccedure
    number = number of sampling you want to do from you distribution 
    """
    sample_mean = []
    for i in range(number):
        sampling = np.random.choice(dist, size=sample_size, replace=True) 
        sample_mean.append(np.mean(sampling))
    return sample_mean

dist_select = st.sidebar.selectbox('Select you distribution', options=['normal', 'binomial', 'exponential', 'upload personal data'])
number_sampling = st.sidebar.number_input('Select number of sampling', min_value=1, max_value=5000, value=100)
samp_size = st.sidebar.slider('Select size of each sample', min_value=1, max_value=3000, value=300)

if dist_select == 'upload personal data':
    # upload personal dataset
    upload_files = st.sidebar.file_uploader(label='Upload your own data', help='You can upload your data with unknown distribution', type=['csv'])
    if upload_files is not None:
        df = pd.read_csv(upload_files)
        columns = df.columns
        select_column = st.sidebar.selectbox('Choose your Column', options=columns)
        # check columns type
        try:
            df = df[select_column].astype(np.int64)
            data = simulate(sample_size=samp_size, number=number_sampling, dist=df)
        except:
            st.error('This column is not number!!', icon='🚨')
            st.stop()



elif dist_select == 'normal':
    # sampling from normal distribution
    generated_dist = np.random.normal(loc=0, scale=1, size=5000)
    data = simulate(sample_size=samp_size, number=number_sampling, dist=generated_dist)
    
elif dist_select == 'binomial':
    # sampling from binomial distribution
    generated_dist = np.random.binomial(500, 0.5, 5000)
    data = simulate(sample_size=samp_size, number=number_sampling, dist=generated_dist)
    
elif dist_select == 'exponential':
    # sampling from exponential distribution
    generated_dist = np.random.exponential(scale=1, size=5000)
    data = simulate(sample_size=samp_size, number=number_sampling, dist=generated_dist)    

bt_simulate = st.button('Start Simulation')
if bt_simulate:
    if dist_select == 'upload personal data' and upload_files is None:
        st.warning('Select your dataset first!!', icon="🤖")
        st.stop()
    new_dist = data
    # plot the distribution using plotly
    fig, ax = plt.subplots()
    ax = px.histogram(new_dist)
    st.plotly_chart(ax)

