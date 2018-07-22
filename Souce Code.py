# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 22:02:30 2018

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
combine.drop(['Item_Type'],axis=1,inplace=True)


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
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge




    
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
    

# linear rgression with normalisation
model = LinearRegression(normalize=True) 
martModel(model, train, test, target,sub, 'M1.csv')
coef1 = pd.Series(model.coef_, train.columns).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')
# leaderboard score = 1270


# Ridge regression
model2 = Ridge(alpha=0.05,normalize=True)
martModel(model2,train,test,target,sub,'M2.csv')
coef2 = pd.Series(model2.coef_, train.columns).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')
# leaderboard score = 1253


model3 = RandomForestRegressor(n_estimators = 250,max_depth=8,min_samples_split=50, max_features='sqrt',
                               min_samples_leaf=20 ,random_state=3,n_jobs=-1)
martModel(model3, train, test,target, sub, 'M3.csv')
coef3 = pd.Series(model3.feature_importances_, train.columns).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')
# leaderboard score = 1193.96


# now we wil use GBM Boosting algorithm and tune its parameters by grid Search
# Model Building
def martModel(model,train,predictors,target):
    #Fit the algorithm on the data
    model.fit(train[predictors],target)
        
    #Predict training set:
    train_predictions = model.predict(train[predictors])

    #Perform cross-validation:
    cv_score = cross_val_score(model, train[predictors],target, cv=5, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    R2_score =  r2_score(target, train_predictions)
    
    #Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(target, train_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    print("R2_Score : %s" % "{0:.4%}".format(R2_score))
    
    feat_imp = pd.Series(model.feature_importances_, predictors).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    

    
def pred(model,test,predictors,sub,filename):
    test_target = model.predict(test[predictors])
    
    #Export submission file:
    sub['Item_Outlet_Sales'] = test_target
    sub.to_csv(filename, index=False)
    


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from pprint import pprint

# Start by creating a baseline model using GBM without anytuning

predictors = [x for x in train.columns ]
gbm0 = GradientBoostingRegressor(random_state=10)
martModel(gbm0, train, predictors,target)
pred(gbm0,test,predictors,sub,'gbm0.csv')
#  Leaderboard score =  1236

# First we tune Number of trees
predictors = [x for x in train.columns]
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, 
                                    min_samples_split=80,min_samples_leaf=50,max_depth=8,
                                    max_features='sqrt',subsample=0.8,random_state=10), 
                                    param_grid = param_test1, scoring='neg_mean_squared_error',
                                    n_jobs=4,iid=False, cv=3)

gsearch1.fit(train[predictors],target)
pprint(gsearch1.grid_scores_)
pprint(gsearch1.best_params_)
print(gsearch1.best_score_)
# n tree = 40

# The order of tuning variables should be decided carefully. 
# You should take the variables with a higher impact on outcome first

param_test2 = {'max_depth':range(3,16,2)}
gsearch2 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=40, max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test2, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=3)

gsearch2.fit(train[predictors],target)
pprint(gsearch2.grid_scores_)
pprint(gsearch2.best_params_)
print(gsearch2.best_score_)
# max depth= 4


param_test3 = {'min_samples_split':range(40,201,10)}
gsearch3 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=40,max_depth=5, 
                       max_features='sqrt', subsample=0.8, random_state=10), 
           param_grid = param_test3, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=3)

gsearch3.fit(train[predictors],target)
pprint(gsearch3.grid_scores_)
pprint(gsearch3.best_params_)
print(gsearch3.best_score_)
# min_samples_split = 140

martModel(gsearch3.best_estimator_,train,predictors,target)
pred(gsearch3.best_estimator_,test,predictors,sub,'gbm4.csv')
# Leaderboard score = 1183.84


param_test4 = {'min_samples_leaf':range(20,140,10)}
gsearch4 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=40,max_depth=5, 
                      min_samples_split=140, max_features='sqrt', subsample=0.8, random_state=10), 
           param_grid = param_test4, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=3)

gsearch4.fit(train[predictors],target)
pprint(gsearch4.grid_scores_)
pprint(gsearch4.best_params_)
print(gsearch4.best_score_)
# min_samples_leaf = 30


martModel(gsearch4.best_estimator_,train,predictors,target)
pred(gsearch4.best_estimator_,test,predictors,sub,'gbm5.csv')
# 1186.08




