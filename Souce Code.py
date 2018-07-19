# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:39:06 2018

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 23:04:17 2018

@author: Lenovo
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

# data explorartion
train.head(5)
train.describe()
train.isnull().sum()

# univariate and multivariate analysis 
train.pivot_table(values='Item_Outlet_Sales',index=['Item_Fat_Content'],aggfunc = lambda x: x.mean()).plot(kind='bar',color='red')
# we can see that Low Fat, low fat,lf are same category written differently so we have to correct it

train.pivot_table(values='Item_Outlet_Sales',index=['Outlet_Identifier'],aggfunc = lambda x: x.mean()).plot(kind='bar',color='red')
# one thing to note that OUT027 have more ave sale than other outlets

train.pivot_table(values='Item_Outlet_Sales',index=['Outlet_Establishment_Year'],aggfunc = lambda x: x.mean()).plot(kind='bar',color='red')
# one thing to note that avg sale of outlet open in 1998 are much less the other outlets

train.pivot_table(values='Item_Outlet_Sales',index=['Outlet_Size'],aggfunc = lambda x: x.mean()).plot(kind='bar',color='red')
# ave sale of mediun size outlet are more medium >high>small

train.pivot_table(values='Item_Outlet_Sales',index=['Outlet_Type'],aggfunc = lambda x: x.mean()).plot(kind='bar',color='red')
# ave sale of grocery is much less and Supermarket type_3 is high

# boxplot to check outlier
sns.boxplot(y="Item_Outlet_Sales", data=train)# Outlier present but it is possible that some item have high weight
sns.boxplot(y="Item_Visibility", data=train)# Outlier present but it is possible for item giving more profit

sns.distplot(train['Item_Weight'].dropna(), kde=True)
sns.distplot(train['Item_Visibility'], kde=True) # rightly skewed distribution
sns.distplot(train['Item_MRP'], kde=True)
sns.distplot(train['Item_Outlet_Sales'], kde=True)# rightly skewed distribution


sns.jointplot(x="Item_MRP", y="Item_Outlet_Sales", data=train)

f, ax = plt.subplots(figsize=(10, 8))
corr = train.corr()
sns.heatmap(corr,
            mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

test = pd.read_csv("Test.csv")
target = train.Item_Outlet_Sales
train.drop(['Item_Outlet_Sales'],axis=1,inplace=True)

# combining test and train data for feature engineering
combine = train.append(test)

# low fat , Low Fat and LF are same category
Item_fat_dictionary = {
    "Low Fat": "Low Fat",
    "Regular": "Regular",
    "LF": "Low Fat",
    "reg": "Regular",     
    "low fat": "Low Fat",
}
# map the Item fat content
combine["Item_Fat_Content"] = combine.Item_Fat_Content.map(Item_fat_dictionary)

# check missing values
combine.isnull().sum()

# Filling missing value of Item wqeight
Item_avg_weight= combine.pivot_table(values='Item_Weight',index=['Item_Identifier'])

#Get a boolean variable specifying missing Item_Weight values
miss_bool = combine['Item_Weight'].isnull() 

#Impute data and check #missing values before and after imputation to confirm
print ('Orignal #missing: %d'% sum(miss_bool))
combine.loc[miss_bool,'Item_Weight'] = combine.loc[miss_bool,'Item_Identifier'].apply(lambda x: Item_avg_weight.loc[x])
# df.loc[row,col]
combine.isnull().sum()

# we have seen some value of visibility values are zero which cannot be possible as item was sold
sum(combine.Item_Visibility==0)
# replace zero by nan
combine.Item_Visibility = combine.Item_Visibility.replace(0,np.nan)
sum(combine.Item_Visibility==0)


# Filling missing value of visibility
avg_item_visi = combine.pivot_table(values='Item_Visibility',index=['Item_Identifier'])

#Get a boolean variable specifying missing Item_Weight values
miss_bool = combine['Item_Visibility'].isnull() 

print ('Orignal #missing: %d'% sum(miss_bool))
#Impute data and check #missing values before and after imputation to confirm
combine.loc[miss_bool,'Item_Visibility'] = combine.loc[miss_bool,'Item_Identifier'].apply(lambda x: avg_item_visi.loc[x])
# df.loc[row,col]
combine.isnull().sum()

temp = pd.crosstab(combine['Outlet_Type'],combine['Outlet_Size'])
temp.plot(kind='bar',stacked=True,color=['red','blue','green'])
# we can infer that all Grocery store are small and supermarket2 &3 are medium in size

temp2 = pd.crosstab(combine['Outlet_Location_Type'],combine['Outlet_Size'])
temp2.plot(kind='bar',stacked=True,color=['red','blue','green'])
# we can infer that all outlet in tier 2 cities are small

# filling Outlet_Size 
combine.loc[ (pd.isnull(combine['Outlet_Size'])) & (combine['Outlet_Type'] =='Grocery Store'),'Outlet_Size'] = 'Small'
combine.loc[ (pd.isnull(combine['Outlet_Size'])) & (combine['Outlet_Type'] =='Supermarket Type2'),'Outlet_Size'] = 'Medium'
combine.loc[ (pd.isnull(combine['Outlet_Size'])) & (combine['Outlet_Type'] =='Supermarket Type3'),'Outlet_Size'] = 'Medium'
combine.loc[ (pd.isnull(combine['Outlet_Size'])) & (combine['Outlet_Location_Type'] =='Tier 2'),'Outlet_Size'] = 'Small'

# checking 
combine.isnull().sum()


# treating skewed disribution
combine['Sqrt_Item_Weight'] = np.sqrt(combine['Item_Weight'])
combine['log_Item_Visibility'] = np.log(combine['Item_Visibility'])



# feature engineering
# we can use hypothesis that more the visibility more will be sale of product
combine['Item_Visibility_MeanRatio'] = combine.apply(lambda x: x['Item_Visibility']/avg_item_visi.loc[x['Item_Identifier']], axis=1)
combine.head()

# process Item_fat_Content
fat_content_dummies= pd.get_dummies(combine['Item_Fat_Content'],prefix="Fat_Content")
combine = pd.concat([combine,fat_content_dummies],axis=1)
combine.drop('Item_Fat_Content',axis=1,inplace=True)

# Process Item_Identifier
# we can see that item identifier either starting with FD or DR or NC
combine['Broad_Item_Category'] = combine['Item_Identifier'].apply(lambda row: row[0:2])

broad_cat_dummies= pd.get_dummies(combine['Broad_Item_Category'],prefix="Item_Category")
combine = pd.concat([combine,broad_cat_dummies],axis=1)
combine.drop('Broad_Item_Category',axis=1,inplace=True)

# process Outlet_IOdentifier
Outlet_identity_dummies= pd.get_dummies(combine['Outlet_Identifier'],prefix="NUM")
combine = pd.concat([combine,Outlet_identity_dummies],axis=1)
combine.drop('Outlet_Identifier',axis=1,inplace=True)

# process outlet_Size
Outlet_size_dummies= pd.get_dummies(combine['Outlet_Size'],prefix="OUT_SIZE")
combine = pd.concat([combine,Outlet_size_dummies],axis=1)
combine.drop('Outlet_Size',axis=1,inplace=True)

# Outlet_location
Outlet_loc_dummies= pd.get_dummies(combine['Outlet_Location_Type'],prefix="CITY")
combine = pd.concat([combine,Outlet_loc_dummies],axis=1)
combine.drop('Outlet_Location_Type',axis=1,inplace=True)

# Outlet_Type
Outlet_type_dummies= pd.get_dummies(combine['Outlet_Type'])
combine = pd.concat([combine,Outlet_type_dummies],axis=1)
combine.drop('Outlet_Type',axis=1,inplace=True)

combine.to_csv("combine_stage1.csv",index=False,header=True)
combine = pd.read_csv("combine_stage1.csv")
# incase we lost our data


from sklearn.preprocessing import LabelEncoder
# label encoding on item identifier  so that information does not lost
le= LabelEncoder()
combine['Item_Identity'] = le.fit_transform(combine['Item_Identifier'])

# drop unneccesary variables
combine.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)


# Though we know that making new fearture using from response variable id not a good practice 
# but we can do it in this dataset because all the item in train set are also present in test set
train = pd.concat([train,target],axis=1)
avg_Item_Sales = train.pivot_table(values="Item_Outlet_Sales",index=["Item_Identifier"])
#avg_Item_Sales['Item_Identifier'] = avg_Item_Sales.index
avg_Item_Sales.columns = ["avg_Item_Sales"]



train = combine.iloc[:8523]
test = combine.iloc[8523:]

# add avg_item_sale to train and test dataset
train = train.join(avg_Item_Sales,on="Item_Identifier")
test = test.join(avg_Item_Sales,on="Item_Identifier")
# y = target variable


# drop unnecssary variable
train.drop('Item_Identifier',axis=1,inplace=True)
test.drop('Item_Identifier',axis=1,inplace=True)

train.to_csv("train_processed.csv",index=False,header=True)
test.to_csv("test_processed.csv",index=False,header=True)
train = pd.read_csv("train_processed.csv")
test = pd.read_csv("test_processed.csv")

# we will store Item_identifier & outlet_Identifier in a data frame because we will use it in submission
df = pd.read_csv("Test.csv")
sub = pd.DataFrame(df[['Item_Identifier','Outlet_Identifier']])



# Feature importance 
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = 50,max_features = 'sqrt')
model = model.fit(train,target)
feat_imp = pd.Series(model.feature_importances_, train.columns).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show()

from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor



    
# Model Building
def martModel(model,train,test,predictors,target,sub,filename):
    #Fit the algorithm on the data
    model.fit(train[predictors],target)
        
    #Predict training set:
    train_predictions = model.predict(train[predictors])

    #Perform cross-validation:
    cv_score = cross_val_score(model, train[predictors],target, cv=20, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    R2_score =  r2_score(target, train_predictions)
    
    #Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(target, train_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    print("R2_Score : %s" % "{0:.4%}".format(R2_score))
    
    #Predict on testing data:
    test_target = model.predict(test[predictors])
    
    #Export submission file:
    sub['Item_Outlet_Sales'] = test_target
    sub.to_csv(filename, index=False)
    


predictors = ['Item_Weight', 'Item_Visibility', 'Item_MRP',
       'Fat_Content_Low Fat', 'Fat_Content_Regular', 'Item_Category_DR',
       'Item_Category_FD', 'Item_Category_NC', 'NUM_OUT010', 'NUM_OUT013',
       'NUM_OUT017', 'NUM_OUT018', 'NUM_OUT019', 'NUM_OUT027', 'NUM_OUT035',
       'NUM_OUT045', 'NUM_OUT046', 'NUM_OUT049', 'OUT_SIZE_High',
       'OUT_SIZE_Medium', 'OUT_SIZE_Small', 'CITY_Tier 1', 'CITY_Tier 2',
       'CITY_Tier 3', 'Grocery Store', 'Supermarket Type1',
       'Supermarket Type2', 'Supermarket Type3', 'Item_Identity',
       'avg_Item_Sales']

#


# linear rgression with normalisation
model1 = LinearRegression(normalize=True) 
martModel(model1, train, test, predictors, target,sub, 'alg1.csv')
coef1 = pd.Series(model1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')
# leaderboard  = 1271.26 

predictors = ['Item_MRP', 'Sqrt_Item_Weight',
       'log_Item_Visibility', 'Item_Visibility_MeanRatio',
       'Fat_Content_Low Fat', 'Fat_Content_Regular', 'Item_Category_DR',
       'Item_Category_FD', 'Item_Category_NC', 'NUM_OUT010', 'NUM_OUT013',
       'NUM_OUT017', 'NUM_OUT018', 'NUM_OUT019', 'NUM_OUT027', 'NUM_OUT035',
       'NUM_OUT045', 'NUM_OUT046', 'NUM_OUT049', 'OUT_SIZE_High',
       'OUT_SIZE_Medium', 'OUT_SIZE_Small', 'CITY_Tier 1', 'CITY_Tier 2',
       'CITY_Tier 3', 'Grocery Store', 'Supermarket Type1',
       'Supermarket Type2', 'Supermarket Type3', 'Item_Identity',
       'avg_Item_Sales']

# linear rgression with normalisation
model2 = LinearRegression(normalize=True) 
martModel(model2, train, test, predictors, target,sub, 'alg2.csv')
coef2 = pd.Series(model2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')
# leaderboard = 1270


# Ridge regression
model3 = Ridge(alpha=0.05,normalize=True)
martModel(model3,train,test,predictors,target,sub,'alg3.csv')
coef3 = pd.Series(model3.coef_, predictors).sort_values()
coef3.plot(kind='bar', title='Model Coefficients')
# leaderboard = 1253.526

from sklearn.ensemble import RandomForestRegressor


model5 = RandomForestRegressor(n_estimators = 200,max_depth=8,min_samples_leaf=100, max_features=0.3
                                ,random_state=3,n_jobs=-1,)
martModel(model5, train, test, predictors, target, sub, 'alg11.csv')
coef11 = pd.Series(model5.feature_importances_, predictors).sort_values(ascending=False)
coef11.plot(kind='bar', title='Feature Importances')
# leaderboard = 1198.96






