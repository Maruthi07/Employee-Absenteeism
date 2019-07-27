#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import math
from scipy.stats import chi2_contingency
import os
import seaborn as sns #for plotting

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


os.getcwd()
d1=pd.read_excel("Absenteeism_at_work_Project (1).xls")


# In[6]:


d1.head
d2=d1.copy()


# In[7]:


d2['Reason for absence'] = d2['Reason for absence'].fillna(d2['Reason for absence'].median())
d2['Month of absence'] = d2['Month of absence'].fillna(d2['Month of absence'].median())
d2['Transportation expense'] = d2['Transportation expense'].fillna(d2['Transportation expense'].median())
d2['Distance from Residence to Work'] = d2['Distance from Residence to Work'].fillna(d2['Distance from Residence to Work'].median())
d2['Service time']= d2['Service time'].fillna(d2['Service time'].median())
d2['Age'] = d2['Age'].fillna(d2['Age'].median())
d2['Work load Average/day ']= d2['Work load Average/day '].fillna(d2['Work load Average/day '].median())
d2['Hit target']= d2['Hit target'].fillna(d2['Hit target'].median())
d2['Disciplinary failure']= d2['Disciplinary failure'].fillna(d2['Disciplinary failure'].median())
d2['Education']= d2['Education'].fillna(d2['Education'].median())
d2['Son']= d2['Son'].fillna(d2['Son'].median())
d2['Social drinker']= d2['Social drinker'].fillna(d2['Social drinker'].median())
d2['Social smoker']= d2['Social smoker'].fillna(d2['Social smoker'].median())
d2['Pet']= d2['Pet'].fillna(d2['Pet'].median())
d2['Weight']= d2['Weight'].fillna(d2['Weight'].median())
d2['Height']= d2['Height'].fillna(d2['Height'].median())
d2['Body mass index']= d2['Body mass index'].fillna(d2['Body mass index'].median())
d2['Absenteeism time in hours']= d2['Absenteeism time in hours'].fillna(d2['Absenteeism time in hours'].median())


# In[8]:


d3=d2


# In[9]:


#converting dataframe into numeric

d2['Reason for absence'] = d2['Reason for absence'].astype(int)
d2['Month of absence'] = d2['Month of absence'].astype(int)
d2['Transportation expense'] = d2['Transportation expense'].astype(int)
d2['Distance from Residence to Work'] = d2['Distance from Residence to Work'].astype(int)
d2['Service time'] = d2['Service time'].astype(int)
d2['Age'] = d2['Age'].astype(int)
d2['Work load Average/day '] = d2['Work load Average/day '].astype(int)
d2['Hit target'] = d2['Hit target'].astype(int)
d2['Disciplinary failure'] =d2['Disciplinary failure'].astype(int)
d2['Education'] = d2['Education'].astype(int)
d2['Son'] = d2['Son'].astype(int)
d2['Age'] = d2['Age'].astype(int)
d2['Social drinker'] = d2['Social drinker'].astype(int)
d2['Social smoker'] = d2['Social smoker'].astype(int)
d2['Pet'] = d2['Pet'].astype(int)
d2['Weight'] = d2['Weight'].astype(int)
d2['Height'] = d2['Height'].astype(int)
d2['Body mass index'] = d2['Body mass index'].astype(int)
d2['Absenteeism time in hours'] = d2['Absenteeism time in hours'].astype(int)


# In[10]:


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(d2['Transportation expense'], [75 ,25])

#Calculate_IQR
iqr = q75 - q25

#Calculate inner and outer fence
mini = q25 - (iqr*1.5)
maxi = q75 + (iqr*1.5)

#Replace with NA
d2.loc[d2['Transportation expense'] < mini,:'Transportation expense'] = np.nan
d2.loc[d2['Transportation expense'] > maxi,:'Transportation expense'] = np.nan


# In[11]:


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(d2['Month of absence'], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
mini = q25 - (iqr*1.5)
maxi = q75 + (iqr*1.5)

#Replace with NA
d2.loc[d2['Month of absence'] < mini,:'Month of absence'] = np.nan
d2.loc[d2['Month of absence'] > maxi,:'Month of absence'] = np.nan


# In[ ]:


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(d2['Distance from Residence to Work'], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
mini = q25 - (iqr*1.5)
maxi = q75 + (iqr*1.5)

#Replace with NA
d2.loc[d2['Distance from Residence to Work'] < mini,:'Distance from Residence to Work'] = np.nan
d2.loc[d2['Distance from Residence to Work'] > maxi,:'Distance from Residence to Work'] = np.nan


# In[ ]:


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(d2['Service time'], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
mini = q25 - (iqr*1.5)
maxi = q75 + (iqr*1.5)

#Replace with NA
d2.loc[d2['Service time'] < mini,:'Service time'] = np.nan
d2.loc[d2['Service time'] > maxi,:'Service time'] = np.nan


# In[ ]:


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(d2['Age'], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
mini = q25 - (iqr*1.5)
maxi = q75 + (iqr*1.5)

#Replace with NA
d2.loc[d2['Age'] < mini,:'Age'] = np.nan
d2.loc[d2['Age'] > maxi,:'Age'] = np.nan


# In[ ]:


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(d2['Work load Average/day '], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
mini = q25 - (iqr*1.5)
maxi = q75 + (iqr*1.5)

#Replace with NA
d2.loc[d2['Work load Average/day '] < mini,:'Work load Average/day '] = np.nan
d2.loc[d2['Work load Average/day '] > maxi,:'Work load Average/day '] = np.nan


# In[ ]:


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(d2['Hit target'], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
mini = q25 - (iqr*1.5)
maxi = q75 + (iqr*1.5)

#Replace with NA
d2.loc[d2['Hit target'] < mini,:'Hit target'] = np.nan
d2.loc[d2['Hit target'] > maxi,:'Hit target'] = np.nan


# In[ ]:


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(d2['Son'], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
mini = q25 - (iqr*1.5)
maxi = q75 + (iqr*1.5)

#Replace with NA
d2.loc[d2['Son'] < mini,:'Son'] = np.nan
d2.loc[d2['Son'] > maxi,:'Son'] = np.nan


# In[ ]:


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(d2['Pet'], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
mini = q25 - (iqr*1.5)
maxi = q75 + (iqr*1.5)

#Replace with NA
d2.loc[d2['Pet'] < mini,:'Pet'] = np.nan
d2.loc[d2['Pet'] > maxi,:'Pet'] = np.nan


# In[ ]:


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(d2['Height'], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
mini = q25 - (iqr*1.5)
maxi = q75 + (iqr*1.5)

#Replace with NA
d2.loc[d2['Height'] < mini,:'Height'] = np.nan
d2.loc[d2['Height'] > maxi,:'Height'] = np.nan


# In[ ]:


#Detect and replace with NA
#Extract quartiles
qu75, qu25 = np.percentile(d2['Weight'], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
mini = qu25 - (iqr*1.5)
maxi = qu75 + (iqr*1.5)

#Replace with NA
d2.loc[d2['Weight'] < mini,:'Weight'] = np.nan
d2.loc[d2['Weight'] > maxi,:'Weight'] = np.nan


# In[ ]:


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(d2['Body mass index'], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
mini = q25 - (iqr*1.5)
maxi = q75 + (iqr*1.5)

#Replace with NA
d2.loc[d2['Body mass index'] < mini,:'Body mass index'] = np.nan
d2.loc[d2['Body mass index'] > maxi,:'Body mass index'] = np.nan


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(d2['Absenteeism time in hours'], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
mini = q25 - (iqr*1.5)
maxi = q75 + (iqr*1.5)

#Replace with NA
d2.loc[d2['Absenteeism time in hours'] < mini,:'Absenteeism time in hours'] = np.nan
d2.loc[d2['Absenteeism time in hours'] > maxi,:'Absenteeism time in hours'] = np.nan


# In[12]:


missing_val=pd.DataFrame(d2.isnull().sum())


# In[13]:


missing_val


# In[14]:


d2['Reason for absence'] = d2['Reason for absence'].fillna(d2['Reason for absence'].median())
d2['Month of absence'] = d2['Month of absence'].fillna(d2['Month of absence'].median())
d2['Transportation expense'] = d2['Transportation expense'].fillna(d2['Transportation expense'].median())
d2['Distance from Residence to Work'] = d2['Distance from Residence to Work'].fillna(d2['Distance from Residence to Work'].median())
d2['Service time']= d2['Service time'].fillna(d2['Service time'].median())
d2['Age'] = d2['Age'].fillna(d2['Age'].median())
d2['Work load Average/day ']= d2['Work load Average/day '].fillna(d2['Work load Average/day '].median())
d2['Hit target']= d2['Hit target'].fillna(d2['Hit target'].median())
d2['Disciplinary failure']= d2['Disciplinary failure'].fillna(d2['Disciplinary failure'].median())
d2['Education']= d2['Education'].fillna(d2['Education'].median())
d2['Son']= d2['Son'].fillna(d2['Son'].median())
d2['Social drinker']= d2['Social drinker'].fillna(d2['Social drinker'].median())
d2['Social smoker']= d2['Social smoker'].fillna(d2['Social smoker'].median())
d2['Pet']= d2['Pet'].fillna(d2['Pet'].median())
d2['Weight']= d2['Weight'].fillna(d2['Weight'].median())
d2['Height']= d2['Height'].fillna(d2['Height'].median())
d2['Body mass index']= d2['Body mass index'].fillna(d2['Body mass index'].median())
d2['Absenteeism time in hours']= d2['Absenteeism time in hours'].fillna(d2['Absenteeism time in hours'].median())


# In[15]:


d2['ID'] = d1['ID']
d2['Day of the week'] = d1['Day of the week']
d2['Seasons'] = d1['Seasons']


# In[16]:


d2.isnull().sum()


# In[17]:


Missing = pd.DataFrame(d2.isnull().sum())
Missing


# In[18]:


d2.info()

d2['ID'] = d2['ID'].astype('category')
d2['Reason for absence'] = d2['Reason for absence'].astype('category')
d2['Month of absence'] = d2['Month of absence'].astype('category')
d2['Day of the week'] = d2['Day of the week'].astype('category')
d2['Seasons'] = d2['Seasons'].astype('category')
d2['Disciplinary failure'] = d2['Disciplinary failure'].astype('category')
d2['Education'] = d2['Education'].astype('category')
d2['Social drinker'] = d2['Social drinker'].astype('category')
d2['Social smoker'] = d2['Social smoker'].astype('category')


# In[22]:


cnames = ['Transportation expense', 'Distance from Residence to Work',
                    'Service time', 'Age', 'Work load Average/day ', 'Hit target', 'Son',
                    'Pet', 'Weight', 'Height', 'Body mass index']


# In[23]:


df_corr=d2.loc[:,cnames]


# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')
#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(7, 5))

#generate correlation matrix
corr=df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap='rainbow',annot=True,
            square=True, ax=ax)


# In[25]:


d2 = d2.drop(['ID','Day of the week','Seasons','Hit target','Education', 'Social drinker','Social smoker', 'Pet', 'Weight'], axis=1)
# = subset(d1, select = -c(id,education,day.of.the.week, pet,hit.target, seasons,social.smoker,
                            #social.drinker,weight))


# In[26]:


cnames=["Transportation expense", "Distance from Residence to Work",
                    "Service time","Age","Work load Average/day ","Son",
                    "Height","Body mass index","Absenteeism time in hours"]

#'ID', 'Reason for absence', 'Month of absence', 'Day of the week',
 #      'Seasons', 'Transportation expense', 'Distance from Residence to Work',
 #      'Service time', 'Age', 'Work load Average/day ', 'Hit target',
 #      'Disciplinary failure', 'Education', 'Son', 'Social drinker',
 #     'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index',
  #     'Absenteeism time in hours'


# In[27]:


d2.info


# In[28]:


#normalization
for i in cnames:
    print(i)
    d2[i]=(d2[i]-min(d2[i]))/(max(d2[i])-min(d2[i]))


# In[49]:


from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor


# In[50]:


#divide data into train and test
from sklearn.model_selection import train_test_split
train,test=train_test_split(d2,test_size=0.2)


# In[51]:


#Decision tree for regresion
fit=DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,0:9],train.iloc[:,9])


# In[52]:


#apply model on the test data
prediction_DT=fit.predict(test.iloc[:,0:9])


# In[53]:


actual=test['Absenteeism time in hours']
predicted=pd.DataFrame(prediction_DT)
actual=pd.DataFrame(actual)


# In[71]:


predicted["predicted"]=pd.DataFrame(prediction_DT)


# In[72]:


#calculate mape
#def rmse(predict, act):
#    return np.sqrt(((predict- act) ** 2).mean())
def rmse(predictions, targets):

    differences = predictions - targets                       #the DIFFERENCEs.

    differences_squared = differences ** 2                    #the SQUAREs of ^

    mean_of_differences_squared = differences_squared.mean()  #the MEAN of ^

    rmse_val = np.sqrt(mean_of_differences_squared)           #ROOT of ^

    return rmse_val


# In[73]:


actual


# In[74]:


rmse(predicted["predicted"],actual["Absenteeism time in hours"])


# In[75]:


train


# In[76]:


train['Reason for absence']=train['Reason for absence'].astype(float)
train['Month of absence']=train['Month of absence'].astype(float)
train['Disciplinary failure']=train['Disciplinary failure'].astype(float)
train['Height']=train['Height'].astype(float)

#(id,education,day.of.the.week, pet,hit.target, seasons,social.smoker,
 #                           #social.drinker,weight))
    
    
    #'ID', 'Reason for absence', 'Month of absence', 'Day of the week',
 #      'Seasons', 'Transportation expense', 'Distance from Residence to Work',
 #      'Service time', 'Age', 'Work load Average/day ', 'Hit target',
 #      'Disciplinary failure', 'Education', 'Son', 'Social drinker',
 #     'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index',
  #     'Absenteeism time in hours'


# In[77]:


#import libraries for LR
import statsmodels.api as sm

#Train the model using the training sets
model=sm.OLS(train.iloc[:,9],train.iloc[:,0:9]).fit()


# In[78]:


#print out the statistics
model.summary()


# In[79]:


from sklearn.model_selection import train_test_split

X = d2.drop('Absenteeism time in hours',axis=1)
y = d2['Absenteeism time in hours']


# In[80]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[81]:


from sklearn.linear_model import LinearRegression


# In[82]:


lm = LinearRegression()
lm.fit(X_train,y_train)


# In[83]:


predictions = lm.predict(X_test)


# In[84]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.distplot(d2['Absenteeism time in hours'])
plt.show()


# In[85]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df

plt.scatter(y_test,predictions)


sns.distplot((y_test-predictions),bins=50);


# In[86]:


from sklearn import metrics
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[87]:


from sklearn.tree import DecisionTreeRegressor
fit = DecisionTreeRegressor()
fit.fit(X_train,y_train)


# In[88]:


prediction_dtree = fit.predict(X_test)


# In[89]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction_dtree)))


# In[90]:


from sklearn.ensemble import RandomForestRegressor


# In[91]:


# Create a random forest Regressor
RFR = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)

# Train the classifier
RFR.fit(X_train, y_train)


# In[92]:


prediction_RFR = RFR.predict(X_test)


# In[93]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction_RFR)))


# In[94]:


LOSS_DF = d3[['Month of absence','Absenteeism time in hours','Work load Average/day ','Service time']]


# In[95]:


LOSS_DF["Loss"]=(LOSS_DF['Work load Average/day ']*LOSS_DF['Absenteeism time in hours'])/LOSS_DF['Service time']


# In[96]:



LOSS_DF["Loss"] = np.round(LOSS_DF["Loss"]).astype('int64')


# In[97]:


NO = LOSS_DF[LOSS_DF['Month of absence'] == 0]['Loss'].sum()
Jan = LOSS_DF[LOSS_DF['Month of absence'] == 1]['Loss'].sum()
Feb = LOSS_DF[LOSS_DF['Month of absence'] == 2]['Loss'].sum()
Mar = LOSS_DF[LOSS_DF['Month of absence'] == 3]['Loss'].sum()
April =LOSS_DF[LOSS_DF['Month of absence'] == 4]['Loss'].sum()
may = LOSS_DF[LOSS_DF['Month of absence'] == 5]['Loss'].sum()
Jun = LOSS_DF[LOSS_DF['Month of absence'] == 6]['Loss'].sum()
Jul = LOSS_DF[LOSS_DF['Month of absence'] == 7]['Loss'].sum()
Aug = LOSS_DF[LOSS_DF['Month of absence'] == 8]['Loss'].sum()
Sep = LOSS_DF[LOSS_DF['Month of absence'] == 9]['Loss'].sum()
Oct = LOSS_DF[LOSS_DF['Month of absence'] == 10]['Loss'].sum()
Nov = LOSS_DF[LOSS_DF['Month of absence'] == 11]['Loss'].sum()
Dec = LOSS_DF[LOSS_DF['Month of absence'] == 12]['Loss'].sum()


# In[98]:


data = {'No Absent': NO, 'Janaury': Jan,'Febraury': Feb,'March': Mar,
       'April': April, 'May': may,'June': Jun,'July': Jul,
       'August': Aug,'September': Sep,'October': Oct,'November': Nov,
       'December': Dec}


# In[99]:


data = {'No Absent': NO, 'Janaury': Jan,'Febraury': Feb,'March': Mar,
   'April': April, 'May': may,'June': Jun,'July': Jul,
   'August': Aug,'September': Sep,'October': Oct,'November': Nov,
   'December': Dec}


# In[100]:


WorkLoss = pd.DataFrame.from_dict(data, orient='index')


# In[101]:


WorkLoss.rename(index=str, columns={0: "Work Load Loss/Month"})

