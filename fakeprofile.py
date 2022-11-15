#!C:\Users\kaiva\AppData\Local\Programs\Python\Python310\python.exe
#!C:\Python27\python.exe
#!C:\Python310\python.exe
#!C:\Python39\python.exe
#!C:\Program Files\Python39\python.exe    


import cgi ,cgitb
form = cgi.FieldStorage()
status=form["status"].value
followers=form["followers"].value
friends=form["friends"].value
fav=form["fav"].value
lang_num=form["lang_num"].value
listed_count=form["listed_count"].value
geo=form["geo"].value
pic=form["pic"].value

userinput=np.array([[status,followers,friends,fav,lang_num,listed_count,geo,pic]])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_profiling as pp
from pandas_profiling import ProfileReport
import keras as k
from sklearn.model_selection import train_test_split
import keras
from keras import regularizers
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Activation
from keras import optimizers
from numpy.random import seed
seed(1)     


# # Read dataset

# In[2]:


df_users = pd.read_csv("dataset/users.csv")
df_fusers = pd.read_csv("dataset/fusers.csv")


# In[3]:


df_fusers.shape


# In[4]:


df_users.shape


# # Add isFake Column

# In[5]:


#for df_users
isNotFake = np.zeros(3474)

#for df_fusers
isFake = np.ones(3351)


# In[6]:


#adding is fake or not column to make predictions for it
df_fusers["isFake"] = isFake
df_users["isFake"] = isNotFake


# # Combine different datasets into one

# In[7]:


df_allUsers = pd.concat([df_fusers, df_users], ignore_index=True)
df_allUsers.columns = df_allUsers.columns.str.strip()


# In[8]:


#to shuffle the whole data
df_allUsers = df_allUsers.sample(frac=1).reset_index(drop=True)


# In[9]:


df_allUsers.describe()


# In[10]:


df_allUsers.head()


# # Distribution of Data in X and Y

# In[11]:


Y = df_allUsers.isFake


# In[12]:


df_allUsers.drop(["isFake"], axis=1, inplace=True)
X = df_allUsers


# In[13]:


profile = ProfileReport(X, title="Pandas Profiling Report")
profile


# In[14]:


Y.reset_index(drop=True, inplace=True)


# In[15]:


print(Y.shape)


# In[16]:


X.head()


# In[17]:


lang_list = list(enumerate(np.unique(X["lang"])))
lang_dict = {name : i for i, name in lang_list}
X.loc[:, "lang_num"] = X["lang"].map(lambda x: lang_dict[x]).astype(int)

X.drop(["name"], axis=1, inplace=True)


# # Feature Selection

# In[18]:


X = X[[
    "statuses_count",
    "followers_count",
    "friends_count",
    "favourites_count",
    "lang_num",
    "listed_count",
    "geo_enabled",
    "profile_use_background_image"
                        ]]


# In[19]:


profile = ProfileReport(X, title="Pandas Profiling Report")
profile


# In[20]:


X = X.replace(np.nan, 0) #To replace the missing boolean values with zeros as it means false


# In[21]:


profile = ProfileReport(X, title="Pandas Profiling Report")
profile


# # Import Data

# In[22]:


train_X, test_X, train_y, test_y = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, train_size=0.8, test_size=0.2, random_state=0)


# In[23]:


print(train_X.shape)
print(test_X.shape)
print(train_y.shape)
print(test_y.shape)


# # Design Model

# In[24]:


model = Sequential()
model.add(Dense(32, activation='relu', input_dim=8))
model.add(Dense(64, input_dim=32, activation='relu'))
model.add(Dense(64, input_dim=64, activation='relu'))
model.add(Dense(32,input_dim=64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# # Compile Model

# In[25]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_X, train_y,
                    epochs=15,
                    verbose=1,
                    validation_data=(val_X,val_y))

prediction = model.predict(userinput)
prediction = prediction[0]
print('Prediction\n',prediction)
print('\nThresholded output\n',(prediction>0.5)*1)

if ((prediction>0.5)*1)==1:
    result="The Profile is Fake"
else:
    result="The Profile is real"



print ("Content-type:text/html\r\n\r\n")
print("<html>")
print("<head>")
print("<title>Result</title>")
print("</head>")
print("<body>")
print(result)
print("</body>")
print ("</html>")                                   




