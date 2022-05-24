from tkinter.messagebox import NO
from turtle import color
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import  PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import accuracy_score
# Heading
st.write(''' # Explore different ML models and datasets
We are going to see which model is best
''')
# make a box foe datasets
dataset_name = st.sidebar.selectbox(
    'Please select one dataset',('iris','Breast cancer','wine')
)
# make a select box for classifier too
classifier_name= st.sidebar.selectbox(
'Please select one classifier',('KNN','SVM','Random Forest')
)
# define a function to load datastes
def get_datastes(dataset_name):
    data= None
    if dataset_name == 'iris':
        data = datasets.load_iris()
    elif dataset_name == 'wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()

    x = data.data
    y=data.target
    return x,y  

# Now call a function
x,y = get_datastes(dataset_name)

# Now print the shape of datasets on app

st.write('Shape of dataset:',x.shape)
st.write("Number of classifier",len(np.unique(y)))

# Define the parameter of classifiers in user input
def add_parameter_ui(classifier_name):
    params= dict()
    if classifier_name =='SVM':
        c = st.sidebar.slider('c',0.01,10.0)
        params['c']= c
    elif classifier_name=='KNN':
        k =st.sidebar.slider('k',1,15)
        params['k']=k
    else:
        max_depth = st.sidebar.slider('max_depth',2,15)
        params['max_depth']= max_depth
        n_esitmators = st.select_slider('n_estimators',1,100)
        params['n_estimators'] = n_esitmators
    return params
    
# Call function
params = add_parameter_ui(classifier_name)

# Define classifier based on classifier names ans params
def get_classifier(classifier_name,params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(c=params['c'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['k'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'],random_state=1234)
    return clf      


# Call our classifier fuction
clf = get_classifier(classifier_name,params)
    
# split our dataset in test and train test
X_train, X_test ,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1234)
    
# Training and prediction of classifiers
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

# Check accuracy
acc = accuracy_score(y_test,y_pred)
st.write(f'classifier = {classifier_name}')
st.write(f'Accuracy=',acc)

# ploting ('scatter plot')
pca = PCA(2)
X_projected = pca.fit_transform(x)
# Now slice dataset in "0" and "1"
x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2,color= 'red', alpha=0.8,cmap='viridis')
plt.xlabel('Principal component 1') 
plt.ylabel('Principal component 2')
plt.colorbar()

st.pyplot(fig)