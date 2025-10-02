from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
df = pd.read_csv('assurance_maladie.csv')
df.describe().round(3)

df.info()
df.head()
df.describe
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.duplicated().sum()
df.isna().sum()
df.isnull().sum()
sns.histplot(data=df , x="age", binwidth=3)
plt.show()

selected_columns_nums = ['age' , 'bmi' , 'children' , 'charges']
df_num = df[selected_columns_nums]
sns.heatmap(df_num.corr() , annot=True , cmap="Blues" , fmt=".2f" )
plt.show()

selected_columns_nums = ['age' , 'bmi' , 'children' , 'charges']

imputer_mean = SimpleImputer(strategy='mean')
df[selected_columns_nums] = imputer_mean.fit_transform(df[selected_columns_nums])


selected_columns_string = ['sex' , 'smoker' , 'region']
for i in range(len(selected_columns_string)):
    
    mode_category = df[selected_columns_string[i]].mode()[0]

    # Fill missing values with the calculated mode
    df[selected_columns_string].fillna(mode_category, inplace=True)
    
print(df)

    


df.duplicated().sum()

df.boxplot('charges')
sns.boxplot(df["bmi"])
print(df.shape)
from scipy.stats import zscore

for i in range(len(selected_columns_nums)):
    df['Z-score'] = zscore(df[selected_columns_nums[i]])
    df = df[df['Z-score'].abs() <= 3]  

df = df.drop(columns=['Z-score'])     
print(df)



#outliers = df[df['Z-score'].abs() > 3]
#print(outliers)

print(df.columns)

Q1 = df['bmi'].quantile(0.25)
Q3 = df['bmi'].quantile(0.75)
IQR = Q3 - Q1

# Calculate bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect outliers
df = df[(df['bmi'] >= lower_bound) & (df['bmi'] <= upper_bound)]


df['charges'] = np.log(df['charges'])








print(df.shape)
df_cleaned = df
df_cleaned.boxplot('charges')

catg_cols = df.select_dtypes(include='object').columns.to_list()
print(catg_cols)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False , handle_unknown='ignore' )
encoder.fit(df[catg_cols])

encoded_cols = list(encoder.get_feature_names_out(catg_cols))
df[encoded_cols] = encoder.transform(df[catg_cols])
print(df[encoded_cols])
from sklearn.model_selection import train_test_split
X = df_cleaned.drop(columns=['bmi' , 'sex' , 'region' , 'smoker'])
Y = df_cleaned['bmi']
X.head()
Y.head()
X.shape
Y.shape
X_train , X_test , Y_train , Y_test = train_test_split(X ,Y , random_state=11 , test_size=0.2)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape
X_train.describe().round(3)




X1 = df_cleaned[selected_columns_nums]
X2 = df_cleaned[selected_columns_nums]
from sklearn.preprocessing import StandardScaler
scaleStandard = StandardScaler()
X1 = scaleStandard.fit_transform(X1)
X1 = pd.DataFrame(X1 , columns=selected_columns_nums)
X1.head()
X1.describe().round(3)


from sklearn.preprocessing import MinMaxScaler
scaleminmax = MinMaxScaler(feature_range=(0,1))
X2 = scaleminmax.fit_transform(X2)
X2 = pd.DataFrame(X2 , columns=selected_columns_nums)
X2.describe().round(3)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
lm = LinearRegression()
lm.fit(X_train , Y_train)
lm.coef_
cdf = pd.DataFrame(lm.coef_ , X.columns , columns=['Coef'])
#print(cdf)

y_pred = lm.predict(X_test)
mean_absolute_error(Y_test, y_pred)





mean_squared_error(Y_test , y_pred)
r2_score(Y_test , y_pred)
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=13)
rfr.fit(X_train , Y_train)
y_pred = rfr.predict(X_test)
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
mean_absolute_error(Y_test, y_pred)

mean_squared_error(Y_test , y_pred)
r2_score(Y_test , y_pred)
from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train , Y_train)
y_pred = xgb.predict(X_test)
mean_absolute_error(Y_test, y_pred)

y_pred_train = xgb.predict(X_train)
mean_absolute_error( Y_train, y_pred_train)
mean_squared_error(Y_test , y_pred)
r2_score(Y_test , y_pred)
from sklearn.svm import SVR

svr = SVR()
svr.fit(X_train , Y_train)

y_pred = svr.predict(X_test)
mean_absolute_error(Y_test , y_pred)
mean_squared_error(Y_test , y_pred)
r2_score(Y_test , y_pred)
