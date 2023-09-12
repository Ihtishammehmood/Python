
> **Table of Content**

- [Random forest (ML) Model](#random-forest-ml-model)
- [Steps](#steps)
  - [import Liabraries](#import-liabraries)
  - [Data load](#data-load)
  - [checking null Values](#checking-null-values)
  - [Dropping columns with majority of null values](#dropping-columns-with-majority-of-null-values)
  - [Dealing with Outlier in IQR( Interquartile Range)](#dealing-with-outlier-in-iqr-interquartile-range)
  - [High correlation between variables](#high-correlation-between-variables)
    - [use of correlation matrix](#use-of-correlation-matrix)
  - [Normality check](#normality-check)
  - [normalizing data](#normalizing-data)
  - [One  Hot Endocding](#one--hot-endocding)
  - [check for overfitting](#check-for-overfitting)
  - [Data splitting](#data-splitting)
  - [Random Forest](#random-forest)
    - [Data spliting](#data-spliting)
    - [Model Training](#model-training)
    - [Model Efficiency check](#model-efficiency-check)
   








---
# Random forest (ML) Model
The theory behind the random forest machine learning model is based on the concept of ensemble learning. Ensemble learning is a technique that combines multiple learning models to improve the performance of the overall model. In the case of random forest, the individual learning models are decision trees.
This following data set has the details about Celestial objects like Astroid

>[Dataset](https://www.kaggle.com/datasets/sakhawat18/asteroid-dataset)

The main purpose of using random forest is to predict the PHA(Potentially Hazardous Astroid) in a given data set.

# Steps

## import Liabraries

The first step before working on different data sets is to import some libraries. The important libraries are as follows.
``````python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
``````

## Data load

The second important step is to load a data set into p pandas dataframe. Because I was working in kaggle environment so, it has different method of loading a dataset
``````python
df = pd.read_csv('/kaggle/input/asteroid-dataset/dataset.csv')
``````
## checking null Values 

Null values can be very problematic when it comes to data analysis or data creating a machine learning models. Dealing with null values should be the primary concern.

``````python
df.isnull().sum()
``````

This line of code gives the list of all the null values in the series(columns) of dataframe. In this cas we can get a summary about the total number of NaNs in each columns

## Dropping columns with majority of null values

Dropping columns that has  most null values. These kind of variables are not helpful in model building onr data data analysis
I removed the columns that has a lot of null values with the following line of code

``````python
column_drop = ['name','prefix','diameter','albedo','diameter_sigma']
df= df.drop(columns=column_drop)
``````

After that I removed other smaller count of null values by removing rows that has null values in it

``````python
clean_df = df.dropna()
``````

## Dealing with Outlier in IQR( Interquartile Range)

Outliers can be very problematic for the creating machine learning models. Outliers can skew the result. There are many ways to deal with the outliers
one of the most common ways is the IQR method
``````python
data = clean_df

def find_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers

def cap_outliers_in_dataframe(data):
    capped_df = data.copy()

    for column in capped_df.columns:
        if np.issubdtype(capped_df[column].dtype, np.number):  # Check if the column contains numeric data
            outliers = find_outliers_iqr(capped_df[column])

            # Cap the outliers by replacing them with lower/upper bounds
            Q1 = capped_df[column].quantile(0.25)
            Q3 = capped_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            capped_df[column] = np.where(outliers, np.clip(capped_df[column], lower_bound, upper_bound), capped_df[column])

    return capped_df

capped_df = cap_outliers_in_dataframe(data)

# Print the DataFrame with capped values
print(capped_df)
``````

This line code first create upper limit and lower limit with the name of Q1 and Q3 then then the  formula is created which is IQR = Q3-Q1
The main purpose of creating this upper and lower limit is to bound the data frame within this limit above or below the limit is considered as outlier 

``````Python
   lower_bound = Q1 - 1.5 * IQR
   upper_bound = Q3 + 1.5 * IQR
``````

## High correlation between variables

When there are highly correlated columns in a data frame, it means that those columns are linearly related to each other. This means that they carry almost the same information, making it redundant to include all of them in a model.

Including highly correlated features in a model can cause a number of problems, including:

* Reduced accuracy
* Increased variance
* Multicollinearity

### use of correlation matrix

A correlation matrix is a square matrix that shows the correlation coefficients between all pairs of variables in a dataset. The correlation coefficient is a measure of the linear relationship between two variables.

It can be used to identify the variables which are highly  correlated with each other 
``````python
# Select only numeric columns
numeric_columns = capped_df.select_dtypes(include=[np.number])

# Compute the correlation matrix for numeric columns
corr_matrix = numeric_columns.corr().abs()

# Create a mask for selecting the upper triangle of the correlation matrix
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find and drop highly correlated columns
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] >= 0.99)]
capped_df.drop(to_drop, axis=1, inplace=True)
``````

I have given a threshold in a code to see remvoe all those columns that has correlation of >=0.99 which is considered as a high correlation 

## Normality check

Normality check helps to see if data is normaly distributed or not 
``````python
from scipy import stats
from scipy.stats import shapiro
numeric_df = capped_df.select_dtypes(include=['float', 'int'])

for column in numeric_df:
    stat, p_value = shapiro(capped_df[column])
    
    print('Shapiro test for column {}: p-value = {}'.format(column, p_value))
``````
 In this case Shapiro-Wilk test has been use to check the normality of the data
 Shapiro wilk test is applies only to numeric data with the help of for loop function

## normalizing data 

Normalizing a data can be done in various ways like normalization and standardization. In this case z-score is used to normalize the data .

``````python
# Create a list of all the float and numeric columns
float_columns = [column for column in capped_df.columns if pd.api.types.is_numeric_dtype(capped_df[column])]

# Normalize the data using z-score
for column in float_columns:
    capped_df[column] = capped_df[column] - capped_df[column].mean() / capped_df[column].std()
``````
## One  Hot Endocding

Machine does not understand any text formant to make it understand any data first, it needs to be converted into numerical formant. One hot Encoding does the same. It converts the objects (strings ) into float.

``````python
from sklearn.preprocessing import OneHotEncoder


one_hot_encoder = OneHotEncoder(sparse=False, drop='first')
capped_df['neo'] = one_hot_encoder.fit_transform(capped_df[['neo']])
capped_df['pha'] = one_hot_encoder.fit_transform(capped_df[['pha']])
capped_df['orbit_id'] = one_hot_encoder.fit_transform(capped_df[['orbit_id']])
capped_df['class'] = one_hot_encoder.fit_transform(capped_df[['class']])
capped_df['equinox'] = one_hot_encoder.fit_transform(capped_df[['equinox']])
``````

before proceeding, important library needs to be imported

``````python
from sklearn.preprocessing import OneHotEncoder
``````
Different columns with object formant needs to be placed within the code as shown above. 
One Hot Encoding works well wiht only few categorical variables

## check for overfitting

Over fitting is used when labels are not in balance
``````python
capped_df.pha.value_counts()/capped_df.shape[0]
``````
by running this code we can see the following results

PHA | Ratio
------- | -------
0.0 | 0.997784
1.0 | 0.002216

it can be clearly seen that the percentage of 0.0(N) in 99% which is not good in order to create a balance in predicted values following line of code has been used:

``````python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state= 42)
x_train, y_train = sm.fit_resample(x_train, y_train.ravel())
``````

## Data splitting

Its time of split the data into Training and Testing 
``````python
# Data preprocessing
x = capped_df.drop(['pha','full_name','id','pdes',], axis=1)
y = capped_df['pha'] 
``````

In this case x variables are the features and y variable is a label that need to be predicted which is PHA( Potentially Hazardous Asteroids).

## Random Forest

``````python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
``````

These liabraries needs to be imported inorder to train and test the model 

### Data spliting
``````python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
``````

The code is for splitting the data into a training set and a test set. The train_test_split() function from the scikit-learn library is used to do this. The function takes four arguments:

x: The features of the dataset.
y: The labels of the dataset.
test_size: The proportion of the data that should be used for the test set.
random_state: A seed for the random number generator.

### Model Training

Now its time to train the model  by selecting and training the model

``````python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
``````

### Model Efficiency check

It is important to check the efficiency of model to see how it is performing

``````python
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# Model evaluation
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
``````

After running this python command we can see the summary of results in following table

**Accuracy: 0.9982195240980978**

**Classification Report:**

``````
      
       precision  recall  f1-score   support

         0.0       1.00      1.00      1.00    186080
         1.0       0.70      0.25      0.37       387

    accuracy                           1.00    186467
   macro avg       0.85      0.63      0.68    186467
weighted avg       1.00      1.00      1.00    186467
``````


