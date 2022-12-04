#!/usr/bin/env python
# coding: utf-8

# # Detecting Parkinson's Disease using Machine Learning Algorithms

# By: Najmeh Moazzen

# Student ID: 961309

# Date: 1400.10.10
# 

# Version: Python 3.9.6

# ## Get the data from PC

# In[1]:


# make dataframe

import numpy as np
import pandas as pd
df= pd.read_csv (r"C:\Users\Njm\Downloads\parkinsons\parkinsons.data")


# In[2]:


df.head()


# In[3]:


df.tail()


# In[4]:


df


# In[5]:


df.info()


# In[6]:


#features name

print (" MDVP:Fo(Hz) : Average vocal fundamental frequency\n"+
       " MDVP:Fhi(Hz) : Maximum vocal fundamental frequency\n"+
       " MDVP:Flo(Hz) : Minimum vocal fundamental frequency\n"+
       " MDVP:Jitter(%) : Five measures of variation in fundamental frequency\n"
       " MDVP:Jitter(Abs) \n"+
       " MDVP:RAP \n"+
       " MDVP:PPQ \n"+
       " Jitter:DDP \n"+
       " MDVP:Shimmer : six measures of variation in amplitude\n"+
       " MDVP:Shimmer (db)\n"+
       " Shimmer:APQ3\n"+
       " Shimmer:APQ5\n"+
       " MDVP:APQ\n"+
       " Shimmer:DDA\n"+
       " NHR : two measures of ratio of noise to tonal components in the voice\n"+
       " HNR\n"
       " RPDE : two nonlinear dynamical complexity measures\n"+
       " D2\n"+
       " DFA : signal fractal scaling exponent\n"+
       " Spread1 : three nonlinear measures of fundamental frequncy variation\n"+
       " Spread2\n"+
       " PPE\n"+
       " Status : Health state of the subject: Parkinson's ---> 1\n"+
       "                                       Healthy     ---> 0 "
)


# In[7]:


df["Jitter:DDP"].value_counts()


# In[8]:


#Number of Parkinson's and healthy cases

df["status"].value_counts()


# ### 1 ---> Parkinson's Positive
# 
# 
# ### 0 ---> Healthy

# In[9]:


df.describe()


# In[10]:


# Grouping the data based on the target variable
df.groupby('status').mean()


# # Visualization

# In[11]:


#Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(20,15))
save_fig("attribute_histogram_plots")
plt.show()


# In[13]:


# Data Visualization of Correlation matrix
import seaborn as sns
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.9, fmt= '.1f',ax=ax)
df.shape


# In[14]:


import seaborn as sns

fig, (axis1, axis2) = plt.subplots(1,2,figsize=(14,8))

sns.violinplot(x="status", y="D2", data=df, ax=axis1, palette="Set2")
axis1.set_title("status vs D2")

sns.violinplot(x="status", y="PPE", data=df, ax=axis2, palette="Set3")
axis2.set_title("status vs PPE")


# In[15]:


import seaborn as sns

fig, (axis1, axis2) = plt.subplots(1,2,figsize=(14,8))

sns.violinplot(x="status", y="RPDE", data=df, ax=axis1, palette="Set2")
axis1.set_title("status vs RPDE")

sns.violinplot(x="status", y="DFA", data=df, ax=axis2, palette="Set3")
axis2.set_title("status vs DFA")


# In[16]:


fig, (axis1, axis2) = plt.subplots(1,2,figsize=(14,8))

sns.boxplot(x="status", y="NHR", data=df, ax=axis1)
axis1.set_title("status vs NHR")

sns.boxplot(x="status", y="HNR", data=df, ax=axis2, palette="Set2")
axis2.set_title("status vs HNR")


# In[17]:


sns.set(rc = {"figure.figsize": (8, 6)})
sns.countplot(x="status", data=df)
# 48 healthy & 147 desease 


# In[18]:


df.plot(kind="scatter", x="status", y="D2", alpha=1)
save_fig("good_visualization_plot")


# # Data pre-processing

# preparing data

# In[19]:


#Length
#Number of all cases
len(df.name.unique())


# In[20]:


len(df.DFA.unique())


# In[21]:


len(df.status.unique())


# In[22]:


df


# In[23]:


df.columns


# In[24]:


# to delete a name column in dataframe

df=df.drop(['name'],axis=1)
df


# In[25]:


df.columns


# In[26]:


#number of posetive parkinson's desease cases
print('Number of posetive parkinsons desease cases: ')
df[df['status']==1].shape


# In[27]:


#number of healthy cases
print('Number of healthy cases: ')
df[df['status']==0].shape


# In[28]:


# Data Standardization

#برای اینکه مقادیر مختلفی و بسیار متفاوت باهم که در جدول هستند را اسکیل کنیم و در یک حدود باشند از این قسمت استفاده میکنیم

from sklearn.preprocessing import StandardScaler
x = df.drop(['status'],axis=1)
y = df['status']


# In[29]:


#آرایه ایکس بین -2 تا 2 افتاده حدودا 


stdscaler = StandardScaler()
x = np.array(stdscaler.fit_transform(x))


# In[30]:


x


# In[31]:


y


# In[90]:


# Train and Test Split
# 80% train data and 20% test data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=10)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[91]:


X_train = np.array(x_train)
len(X_train)


# In[92]:


X_test = np.array(x_test)
len(X_test)


# In[93]:


Y_train = np.array(y_train)
len(Y_train)


# In[94]:


Y_test = np.array(y_test)
len(Y_test)


# # Apply Algorithms

# ## Linear Regression Model

# In[95]:


from sklearn.linear_model import LinearRegression
model1 = LinearRegression()


# In[96]:


model1.fit (X_train,Y_train)


# In[97]:


Y_predmod1 = model1.predict(X_test)
Y_predmod1


# In[98]:


for i,j in enumerate(Y_predmod1):
    if( j < 0.5 ):
        Y_predmod1[i]=0
    else:
        Y_predmod1[i]=1
print(Y_predmod1)
Y_predmod1.shape


# In[99]:


Y_test = np.array(Y_test)
print(Y_test)
Y_test.shape


# In[100]:


print('Y_pred          Y_test')
for i,j in zip(Y_predmod1,Y_test):
    print(i,"             ",j)


# In[101]:


model1.coef_


# In[102]:


model1.intercept_


# In[103]:


#Confusion Matrix

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn                               #to make heat map

cm = confusion_matrix(Y_test,Y_predmod1)
plt.figure(figsize=(8,6))
fg = sn.heatmap(cm, annot=True, cmap='Oranges')
figure =fg.get_figure()
figure.savefig('Linear_Regression.jpg',dpi=400)
plt.xlabel("Predicted")
plt.ylabel('Truth')


# از بین 39 مورد تست شده:
#هشت نفر واقعا سالم هستن و ماشین هم پیش بینی کرده واقعا سالمن
# صفر نفر واقعا بیمارن و ماشین اشتباها پیش بینی کرده که سالم
# سه نفر واقعا سالمن اما ماشین اشتباها پیش بینی کرده که بیمارن
# بیست و هشت نفر واقعا بیمارن و ماشین به درستی پیش بینی کرده که بیمارن


# N: healthy       P:parkinson

# [TN   FP]
# [FN   TP]


# In[104]:


from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
print ("confusion_matrix")
print(confusion_matrix(Y_test,Y_predmod1))
print("\naccuracy:",accuracy_score(Y_test,Y_predmod1))
print("\nrecall:",recall_score(Y_test,Y_predmod1, average = None))
print("\nprecision:",precision_score(Y_test,Y_predmod1, average = None))


# # Logistic Regression Model

# In[105]:


from sklearn.linear_model import LogisticRegression
model2 = LogisticRegression()


# In[106]:


model2.fit(X_train,Y_train)


# In[107]:


Y_predmod2 = model2.predict(X_test)
Y_predmod2


# In[108]:


Y_test


# In[109]:


print('Y_Pred         Y_test')
for i,j in zip(Y_predmod2,Y_test):
    print(i,"             ",j)


# In[110]:


print("Logistic Regression Accuracy : ")
model2.score(X_test,Y_test)


# In[111]:


model2.predict_proba(X_test)


# In[112]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn                               #to make heat map


cm = confusion_matrix(Y_test,Y_predmod2)
plt.figure(figsize=(8,6))
fg = sn.heatmap(cm, annot=True)
figure =fg.get_figure()
figure.savefig('Logistic_Regression.jpg',dpi=400)
plt.xlabel("Predicted")
plt.ylabel('Truth')


# از بین 39 مورد تست شده:
# یازده نفر واقعا سالم هستن و ماشین هم پیش بینی کرده واقعا سالمن
# یک نفر واقعا بیمارن ولی ماشین اشتباها پیش بینی کرده که سالم
# صفر نفر واقعا سالمن اما ماشین اشتباها پیش بینی کرده که بیمارن
# بیست و هفت نفر واقعا بیمارن و ماشین به درستی پیش بینی کرده که بیمارن


# N: healthy       P:parkinson

# [TN   FP]
# [FN   TP]


# In[113]:


from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
print ("confusion_matrix")
print(confusion_matrix(Y_test,Y_predmod2))
print("\naccuracy:",accuracy_score(Y_test,Y_predmod2))
print("\nrecall:",recall_score(Y_test,Y_predmod2, average = None))
print("\nprecision:",precision_score(Y_test,Y_predmod2, average = None))


# ## Decision Tree Model

# In[114]:


from sklearn import tree
model3 = tree.DecisionTreeClassifier()


# In[115]:


model3.fit(X_train,Y_train)


# In[116]:


Y_predmod3 = model3.predict(X_test)
Y_predmod3


# In[117]:


Y_test


# In[118]:


print('Y_Pred          Y_test')
for i,j in zip(Y_predmod3,Y_test):
    print(i,"             ",j)


# In[119]:


print("Decision Tree Accuracy : ")
model3.score(X_test,Y_test)


# In[120]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn                               #to make heat map


cm = confusion_matrix(Y_test,Y_predmod3)
plt.figure(figsize=(8,6))
fg = sn.heatmap(cm, annot=True,cmap='coolwarm')
figure =fg.get_figure()
figure.savefig('Decision_tree.jpg',dpi=400)
plt.xlabel("Predicted")
plt.ylabel('Truth')


# از بین 39 مورد تست شده:
# نه نفر واقعا سالم هستن و ماشین هم پیش بینی کرده واقعا سالمن
# یک نفر واقعا بیماره ولی ماشین اشتباها پیش بینی کرده که سالم
# دو نفر واقعا سالمن اما ماشین اشتباها پیش بینی کرده که بیمارن
# بیست و هفت نفر واقعا بیمارن و ماشین به درستی پیش بینی کرده که بیمارن
# درایه های روی قطر به درستی پیش بینی شده اند

# N: healthy       P: parkinson

# [TN   FP]
# [FN   TP]


# In[121]:


from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
print ("confusion_matrix")
print(confusion_matrix(Y_test,Y_predmod3))
print("\naccuracy:",accuracy_score(Y_test,Y_predmod3))
print("\nrecall:",recall_score(Y_test,Y_predmod3, average = None))
print("\nprecision:",precision_score(Y_test,Y_predmod3, average = None))


# ## Support Vector Machine

# In[122]:


from sklearn.svm import SVC

model4 = SVC()
#model4 = SVC(C=3)


# In[123]:


model4.fit(X_train,Y_train)


# In[124]:


Y_predmod4 = model4.predict(X_test)
Y_predmod4


# In[125]:


Y_test


# In[126]:


print('Y_Pred          Y_test')
for i,j in zip(Y_predmod4,Y_test):
    print(i,"             ",j)


# In[127]:


print("Support Vector Machine Accuracy : ")
model4.score(X_test,Y_test)


# In[128]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn                               #to make heat map


cm = confusion_matrix(Y_test,Y_predmod4)
plt.figure(figsize=(8,6))
fg = sn.heatmap(cm, annot=True,cmap="YlGnBu")
figure =fg.get_figure()
figure.savefig('SVM.jpg',dpi=400)
plt.xlabel("Predicted")
plt.ylabel('Truth')


# از بین 39 مورد تست شده:
#نه نفر واقعا سالم هستن و ماشین هم پیش بینی کرده واقعا سالمن
# صفر نفر واقعا بیماره ولی ماشین اشتباها پیش بینی کرده که سالم
# دو نفر واقعا سالمن اما ماشین اشتباها پیش بینی کرده که بیمارن
# بیست و هشت نفر واقعا بیمارن و ماشین به درستی پیش بینی کرده که بیمارن
# درایه های روی قطر به درستی پیش بینی شده اند

# N: healthy       P: parkinson

# [TN   FP]
# [FN   TP]


# In[129]:


from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
print ("confusion_matrix")
print(confusion_matrix(Y_test,Y_predmod4))
print("\naccuracy:",accuracy_score(Y_test,Y_predmod4))
print("\nrecall:",recall_score(Y_test,Y_predmod4, average = None))
print("\nprecision:",precision_score(Y_test,Y_predmod4, average = None))


# ## Random Forrest Classifier

# In[130]:


from sklearn.ensemble import RandomForestClassifier

model5 = RandomForestClassifier()


# In[131]:


model5.fit(X_train,Y_train)


# In[132]:


Y_predmod5 = model5.predict(X_test)
Y_predmod5


# In[133]:


Y_test


# In[134]:


print('Y_Pred          Y_test')
for i,j in zip(Y_predmod5,Y_test):
    print(i,"             ",j)


# In[135]:


print("Random Forrest Accuracy : ")
model5.score(X_test,Y_test)


# In[138]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


cm = confusion_matrix(Y_test,Y_predmod5)
import seaborn as sn                               #to made heat map
plt.figure(figsize=(8,6))
fg = sn.heatmap(cm, annot=True,cmap="Greens")
figure =fg.get_figure()
figure.savefig('Random_Forrest.jpg',dpi=400)
plt.xlabel("Predicted")
plt.ylabel('Truth')


# از بین 39 مورد تست شده:
# ده نفر واقعا سالم هستن و ماشین هم پیش بینی کرده واقعا سالمن
# یک نفر واقعا بیماره ولی ماشین اشتباها پیش بینی کرده که سالم
# یک نفر واقعا سالمن اما ماشین اشتباها پیش بینی کرده که بیمارن
# بیست و هفت نفر واقعا بیمارن و ماشین به درستی پیش بینی کرده که بیمارن
# درایه های روی قطر به درستی پیش بینی شده اند

# N: healthy       P: parkinson

# [TN   FP]
# [FN   TP]


# In[139]:


from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
print ("confusion_matrix")
print(confusion_matrix(Y_test,Y_predmod5))
print("\naccuracy:",accuracy_score(Y_test,Y_predmod5))
print("\nrecall:",recall_score(Y_test,Y_predmod5, average = None))
print("\nprecision:",precision_score(Y_test,Y_predmod5, average = None))


# ## Extreme Gradient Boosting Model

# In[140]:


from xgboost import XGBClassifier

model6 = XGBClassifier()


# In[141]:


model6.fit(X_train,Y_train)


# In[142]:


Y_predmod6 = model6.predict(X_test)
Y_predmod6


# In[143]:


Y_test


# In[144]:


print('Y_Pred          Y_test')
for i,j in zip(Y_predmod6,Y_test):
    print(i,"             ",j)


# In[145]:


print("Extreme Gradient Boosting Accuracy : ")
model6.score(X_test,Y_test)


# In[147]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn                               #to make heat map


cm = confusion_matrix(Y_test,Y_predmod6)
plt.figure(figsize=(8,6))
fg = sn.heatmap(cm, annot=True,cmap="BuPu")
figure =fg.get_figure()
figure.savefig('XGB_Classifier.jpg',dpi=400)
plt.xlabel("Predicted")
plt.ylabel('Truth')


# از بین 39 مورد تست شده:
# یازده نفر واقعا سالم هستن و ماشین هم پیش بینی کرده واقعا سالمن
# صفر نفر واقعا بیماره ولی ماشین اشتباها پیش بینی کرده که سالم
# صفر نفر واقعا سالمن اما ماشین اشتباها پیش بینی کرده که بیمارن
# بیست و هشت نفر واقعا بیمارن و ماشین به درستی پیش بینی کرده که بیمارن
# درایه های روی قطر به درستی پیش بینی شده اند

# N: healthy       P: parkinson

# [TN   FP]
# [FN   TP]


# In[148]:


from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
print ("confusion_matrix")
print(confusion_matrix(Y_test,Y_predmod6))
print("\naccuracy:",accuracy_score(Y_test,Y_predmod6))
print("\nrecall:",recall_score(Y_test,Y_predmod6, average = None))
print("\nprecision:",precision_score(Y_test,Y_predmod6, average = None))


# # Neural Network Model

# In[161]:


import tensorflow
import keras
from keras.layers import Dense


# In[162]:


model7 = keras.Sequential()
model7.add(Dense(1,input_dim=22,activation='sigmoid'))
model7.add(Dense(20,activation='sigmoid'))
model7.add(Dense(12,activation='sigmoid'))
model7.add(Dense(9,activation='sigmoid'))
model7.add(Dense(5,activation='sigmoid'))
model7.add(Dense(1,activation='sigmoid'))
model7.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model7.fit(X_train,Y_train,epochs=300)


# In[163]:


model7.evaluate(X_test,Y_test)


# In[164]:


print(model7.summary())


# In[165]:


Y_predmod7 = model7.predict(X_test)
Y_predmod7 


# In[166]:


for i,j in enumerate (Y_predmod7):
    if ( j<0.5 ):
        Y_predmod7[i]=0
    else:
        Y_predmod7[i]=1
Y_predmod7


# In[167]:


Y_predmod7.reshape(1,39)


# In[168]:


Y_test


# In[169]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(Y_test,Y_predmod7)
import seaborn as sn                               #to made heat map
plt.figure(figsize=(8,6))
fg = sn.heatmap(cm, annot=True,cmap="PiYG")
figure =fg.get_figure()
figure.savefig('NN.jpg',dpi=400)
plt.xlabel("Predicted")
plt.ylabel('Truth')
plt.show()

# از بین 39 مورد تست شده:
# ده نفر واقعا سالم هستن و ماشین هم پیش بینی کرده واقعا سالمن
#دو نفر واقعا بیماره ولی ماشین اشتباها پیش بینی کرده که سالم
# یک نفر واقعا سالمن اما ماشین اشتباها پیش بینی کرده که بیمارن
# بیست و شش نفر واقعا بیمارن و ماشین به درستی پیش بینی کرده که بیمارن
# درایه های روی قطراصلی به درستی پیش بینی شده اند

# N: healthy       P: parkinson

# [TN   FP]
# [FN   TP]


# In[170]:


from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
print ("confusion_matrix")
print(confusion_matrix(Y_test,Y_predmod7))
print("\naccuracy:",accuracy_score(Y_test,Y_predmod7))
print("\nrecall:",recall_score(Y_test,Y_predmod7, average = None))
print("\nprecision:",precision_score(Y_test,Y_predmod7, average = None))


# # K-Nearest Neighbor Model

# In[171]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[172]:


model8 = KNeighborsClassifier()
model8.fit(X_train, Y_train)


# In[173]:


print("K-Nearest Neighbor Model Accuracy : ")
model8.score(X_test,Y_test)


# In[174]:


#KNN Grid search

params_dict = {'n_neighbors':[3, 5, 7, 9], 'p':[1, 2, 3], 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}
gs = GridSearchCV(model8, param_grid=params_dict, verbose=10, cv=10)


# In[175]:


gs.fit(X_train, Y_train)


# In[176]:


print(gs.best_params_)


# In[177]:


Y_predmod8 = model8.predict(X_test)
Y_predmod8


# In[178]:


Y_test


# In[179]:


print('Y_Pred          Y_test')
for i,j in zip(Y_predmod8,Y_test):
    print(i,"             ",j)


# In[180]:


print(gs.best_estimator_)


# In[182]:


new_model8 = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
             n_neighbors=7, p=1,
             weights='uniform')
new_model8.fit(X_train, Y_train)


# In[183]:


print("KNN - k=7, got {}% accuracy on the test set.".format(accuracy_score(Y_test, new_model8.predict(X_test))*100))


# In[184]:


# precision score on test and train

from sklearn.metrics import precision_score

model8_precision_test = precision_score(Y_test,Y_predmod8,average='weighted')
model8_precision_test


# In[185]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(Y_test,Y_predmod8)
import seaborn as sn                               #to made heat map
plt.figure(figsize=(8,6))
fg = sn.heatmap(cm, annot=True,cmap="YlGnBu")
figure =fg.get_figure()
figure.savefig('KNN.jpg',dpi=400)
plt.xlabel("Predicted")
plt.ylabel('Truth')
plt.show()

# از بین 39 مورد تست شده:
# هشت نفر واقعا سالم هستن و ماشین هم پیش بینی کرده واقعا سالمن
#صفر نفر واقعا بیماره ولی ماشین اشتباها پیش بینی کرده که سالم
# سه نفر واقعا سالمن اما ماشین اشتباها پیش بینی کرده که بیمارن
# بیست و هشت نفر واقعا بیمارن و ماشین به درستی پیش بینی کرده که بیمارن
# درایه های روی قطراصلی به درستی پیش بینی شده اند

# N: healthy       P: parkinson

# [TN   FP]
# [FN   TP]


# In[186]:


from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
print ("confusion_matrix")
print(confusion_matrix(Y_test,Y_predmod8))
print("\naccuracy:",accuracy_score(Y_test,Y_predmod8))
print("\nrecall:",recall_score(Y_test,Y_predmod8, average = None))
print("\nprecision:",precision_score(Y_test,Y_predmod8, average = None))


# ## Naive Bayes Model

# In[187]:


from sklearn.naive_bayes import GaussianNB

model9 = GaussianNB()
model9.fit(X_train,Y_train)


# In[188]:


Y_predmod9 = model9.predict(X_test)
print(Y_predmod9)


# In[189]:


Y_test


# In[190]:


print('Y_Pred          Y_test')
for i,j in zip(Y_predmod9,Y_test):
    print(i,"             ",j)


# In[191]:


print("Naive Bayes Model Accuracy : ")
model9.score(X_test,Y_test)


# In[192]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(Y_test,Y_predmod9)
import seaborn as sn                               #to made heat map
plt.figure(figsize=(8,6))
fg = sn.heatmap(cm, annot=True,cmap="Oranges")
figure =fg.get_figure()
figure.savefig('Naive_Bayes.jpg',dpi=400)
plt.xlabel("Predicted")
plt.ylabel('Truth')
plt.show()

# از بین 39 مورد تست شده:
#یازده نفر واقعا سالم هستن و ماشین هم پیش بینی کرده واقعا سالمن
#ده نفر واقعا بیماره ولی ماشین اشتباها پیش بینی کرده که سالم
# صفر نفر واقعا سالمن اما ماشین اشتباها پیش بینی کرده که بیمارن
# هجده نفر واقعا بیمارن و ماشین به درستی پیش بینی کرده که بیمارن
# درایه های روی قطراصلی به درستی پیش بینی شده اند

# N: healthy       P: parkinson

# [TN   FP]
# [FN   TP]


# In[193]:


# precision score on test and train
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
print ("confusion_matrix")
print(confusion_matrix(Y_test,Y_predmod9))
print("\naccuracy:",accuracy_score(Y_test,Y_predmod9))
print("\nrecall:",recall_score(Y_test,Y_predmod9, average = None))
print("\nprecision:",precision_score(Y_test,Y_predmod9, average = None))

#model9_precision_test = precision_score(Y_test,Y_predmod9,average= 'weighted')


# # All Models Accuracy

# In[194]:


from sklearn.metrics import accuracy_score

print("\n Linear Regression accuracy:",accuracy_score(Y_test,Y_predmod1))
print("\n Logistic Regression accuracy:",accuracy_score(Y_test,Y_predmod2))
print("\n Decision Tree accuracy:",accuracy_score(Y_test,Y_predmod3))
print("\n Support Vector Machine accuracy:",accuracy_score(Y_test,Y_predmod4))
print("\n Random Forrest accuracy:",accuracy_score(Y_test,Y_predmod5))
print("\n XGBClassifier accuracy:",accuracy_score(Y_test,Y_predmod6))
print("\n Neural Network accuracy:",accuracy_score(Y_test,Y_predmod7))
print("\n k-nearest Neighbor accuracy:",accuracy_score(Y_test,Y_predmod8))
print("\n Naive Bayes accuracy:",accuracy_score(Y_test,Y_predmod9))


# In[195]:


from sklearn.metrics import confusion_matrix, recall_score, precision_score

print("\n Linear Regression accuracy:\n",confusion_matrix(Y_test,Y_predmod1))
print("\n Logistic Regression accuracy:\n",confusion_matrix(Y_test,Y_predmod2))
print("\n Decision Tree accuracy:\n",confusion_matrix(Y_test,Y_predmod3))
print("\n Support Vector Machine accuracy:\n",confusion_matrix(Y_test,Y_predmod4))
print("\n Random Forrest accuracy:\n",confusion_matrix(Y_test,Y_predmod5))
print("\n XGBClassifier accuracy:\n",confusion_matrix(Y_test,Y_predmod6))
print("\n Neural Network accuracy:\n",confusion_matrix(Y_test,Y_predmod7))
print("\n k-nearest Neighbor accuracy:\n",confusion_matrix(Y_test,Y_predmod8))
print("\n Naive Bayes accuracy:\n",confusion_matrix(Y_test,Y_predmod9))


# In[196]:


from sklearn.metrics import recall_score, precision_score

print("\n Linear Regression accuracy:\n",recall_score(Y_test,Y_predmod1))
print("\n Logistic Regression accuracy:\n",recall_score(Y_test,Y_predmod2))
print("\n Decision Tree accuracy:\n",recall_score(Y_test,Y_predmod3))
print("\n Support Vector Machine accuracy:\n",recall_score(Y_test,Y_predmod4))
print("\n Random Forrest accuracy:\n",recall_score(Y_test,Y_predmod5))
print("\n XGBClassifier accuracy:\n",recall_score(Y_test,Y_predmod6))
print("\n Neural Network accuracy:\n",recall_score(Y_test,Y_predmod7))
print("\n k-nearest Neighbor accuracy:\n",recall_score(Y_test,Y_predmod8))
print("\n Naive Bayes accuracy:\n",recall_score(Y_test,Y_predmod9))


# In[197]:


from sklearn.metrics import precision_score

print("\n Linear Regression accuracy:\n", precision_score(Y_test,Y_predmod1))
print("\n Logistic Regression accuracy:\n", precision_score(Y_test,Y_predmod2))
print("\n Decision Tree accuracy:\n", precision_score(Y_test,Y_predmod3))
print("\n Support Vector Machine accuracy:\n", precision_score(Y_test,Y_predmod4))
print("\n Random Forrest accuracy:\n", precision_score(Y_test,Y_predmod5))
print("\n XGBClassifier accuracy:\n", precision_score(Y_test,Y_predmod6))
print("\n Neural Network accuracy:\n", precision_score(Y_test,Y_predmod7))
print("\n k-nearest Neighbor accuracy:\n", precision_score(Y_test,Y_predmod8))
print("\n Naive Bayes accuracy:\n", precision_score(Y_test,Y_predmod9))

