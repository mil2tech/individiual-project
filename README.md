# About the Project

## Project Goal

###  The goal of this project is to  accurately predict Consumer Price Index for  Metropolitan area with time series analysis.

## Project Description

### The cost of living and inflation is on the rise.  United States citizens want to find the home that works best for them.  We will analayze data of historical Consumer Price Index (CPI) data from the years 2009 to 2021. The data will be helpful to forecast future CPI. 

### The Consumer Price Index (CPI) is a measure of the average change in prices over time in a fixed market basket of goods and services

## Initial Questions

- Does the Houston's CPI ?

- Is there seasonality in the CPI?

- What times of the year does CPI change?


## Data Dictionary

| Feature    | Description                                        |
|------------|----------------------------------------------------|
| Jan        | January                                            |
| Feb        | February                                           |
| Mar        | March                                              |
| Apr        | April                                              |
| May        | May                                                |
| Jun        | June                                               |
| Jul        | July                                               |
| Aug        | August                                             |
| Sep        | September                                          |
| Oct        | October                                            |
| Nov        | November                                           |
| Dec        | December                                           |
| CPI        | Consumer Price Index                               |
| HALF1      | CPI average for first half of corresponding year.  |
| HALF2      | CPI average for second half of corresponding year. |
| Annual     | CPI average of corresponding year.                 |
| Year       | Year in numeric format                             |
| year_month | Time format as Year and month                      |


### Libraries and files used

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

- import files
import acquire
- working with dates
from datetime import datetime

- to evaluated performance using rmse
from sklearn.metrics import mean_squared_error
from math import sqrt 


- holt's linear trend model. 
from statsmodels.tsa.api import Holt

- setting up plot figure
plt.rc('figure', figsize=(13, 7))
plt.rc('font', size=16)
plt.rc('figure', figsize=(13, 7))
plt.rc('font', size=16)



## Steps to Reproduce

### The Plan
- Acquire the data
- Wrangle (prepare and clean) the data
- Split the data into three samples: `train`, `validate`, and `test`
- Conduct exploration of `train` sample and find relationships in data by plotting visuals and statistical testing
- Explore the data
- Generate various models from  `train` sample
- Generate model predictions by fitting models to only the `train` and `validate` samples to avoid data leakage into `test` sample
- Evaluate the performance of the models with root mean square error (RMSE) calculation and pick the best performing model
- Evaluate the performance of best model on the `test` sample.
- Draw conclusion


### Acquistion of Houston CPI data

To acquire the data, I used the CPI for All Urban Consumers (CPI-U) table on  https://data.bls.gov/PDQWeb/cu  

I set the parameters Houston-The Woodlands-Sugar Land, TX and All Items Range from 2009 to 2021 for the table.

Download .xsls file locally.

Next, I imported the .xsls using the the function below.

'''
def get_houston_overall_cpi():
    # imports  downloaded xlsx file from BLS website to python
    houston_overall_cpi = pd.read_excel('houston_metro_CPI_all_item_09_to_21.xlsx',index_col=None, header= 11)
    return houston_overall_cpi
'''

After importing the data:

- There is 16 columns and 13 rows.

- Six of the month columns are filled with nulls due to the agency's regional office only recording CPI every other month.

- Every value in every column besides `Year` represents CPI measurment recorded per time period. 

Things to do:

- Imput nulls 

- Drop unnecessary columns

- Transform data by melting columns by year and month to expand data vertically

- Set index to time

### Preparation of data

To clean the data I did the following things in order

'''
def prep_houston_overall():
    import acquire
    houston_overall_cpi = acquire.get_houston_overall_cpi()
    # Fill Nulls for first half of the year months with CPI values of HALF1 of that corresponding year
    houston_overall_cpi['Jan'] = houston_overall_cpi['HALF1']
    houston_overall_cpi['Mar'] = houston_overall_cpi['HALF1']
    houston_overall_cpi['May'] = houston_overall_cpi['HALF1']

    # Fill Nulls for second half of the year months with CPI values of HALF2 of that corresponding year
    houston_overall_cpi['Jul'] = houston_overall_cpi['HALF2']
    houston_overall_cpi['Sep'] = houston_overall_cpi['HALF2']
    houston_overall_cpi['Nov'] = houston_overall_cpi['HALF2']

    # Drop following columns
    houston_overall_cpi.drop(columns= ["Annual", "HALF1", "HALF2"], inplace=True)
    # melt data frame by year with the month of corresponding year being the  observation which expands dataframe vertically
    df = pd.melt(houston_overall_cpi, id_vars=["Year"], value_vars=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], var_name="month", value_name="CPI").sort_values(by=['Year'])
    # Convert Year into Integer
    df['Year'] = df.Year.astype(int)
    # Create new column with Year and Month
    df['year_month']=df.apply(lambda x:'%s-%s' % (x['Year'],x['month']),axis=1)
    # Convert combined into data time object
    df.year_month = pd.to_datetime(df.year_month)
    # set index to year_month
    df = df.set_index('year_month').sort_index()
    return df
'''
Actions taken:

- Replaced the months in the of first half of the year with the value  of `HALF1` for the correpsonding year

- Replaced the months in the  of second half of the year with the value  of `HALF2` for the correpsonding year

- Dropped columns "Annual", "HALF1", "HALF2"

- Utilized pd.melt() to expand data vertically

- Converted data index to date time column and sorted in ascending order

Overview of Refined data:

- There is 3 columns and 156 rows.

- No nulls values


### Split the data

- Split the refined dataframe into 3 samples using 

'''
def split_data(df):
    # Train split having years 2009 to 2014
    train = df[:'2014']
    # Validate split having years 2015 to 2020
    validate = df['2015':'2020']
    # Test split having only the year 2021
    test = df['2021']
    print(f" df - {df.shape}.")
    print(f" train - {train.shape}.")
    print(f" validate - {validate.shape}.")
    print(f" test - {test.shape}.")
    return train, validate, test
'''

- Data is to be split into three samples
    - Train 3 columns and 72 rows   equivalent to 6 years of data (2009 - 2014)
    - Validate 3 columns and 72 rows equivelent to 6 years of data (2015 - 2020)
    - Test 3 columns and 12 rows equivelent to one year of data (2021)

### Explore the `Train` sample

- Make a data frame with only the target variable 'CPI'
- The data using '''import statsmodels.api as sm '''
- Auto Correlation plots will provide how strong are past periods of data when resampling data.
    ''' pd.plotting.autocorrelation_plot() '''
- Decomposing the data will show various plots. It will show the trend of the data, seasonality and the  variance of resampled data.
    ''' sm.tsa.seasonal_decompose().plot()'''
- A box plot will show the categorial data like month and year. 

### Modeling the data and evaluating which model performs the best with  `train` and `validate` samples
- Create models performance dataframe
- Create multiple models:
    - Last Observed value
        - Applies last observed CPI value to `validate`
        - Evaluate RMSE
        - Append model evaluation to models performance dataframe
    - Simple Average
        - Applies the mean of CPI to `Validate`
        - Evaluate RMSE
        - Append model evaluation to models performance dataframe
    - Moving Average
        - Create a list of periods
        - Apply the mean of CPI for each period to `Validate`
        - Evaluate RMSE
        - Append model evaluation to models performance dataframe
    - Holts Linear Trend
        - Create Holts object
        - Fit Holts object
        - Make predictions to `validate`
        - Evaluate RMSE
        - Append model evaluation to models performance dataframe
    - Previous Cylce
        - Create a range object (72)
        - loop through every number in range object
        - add the mean of CPI difference to the `train` CPI and apply the `validate` index to the predictions
        - Evaluate RMSE
        - Append model evaluation to models performance dataframe

- Review the results and pick the model with the lowest RMSE to run on scaled `test` sample

### Running best model on `test` sample

- Create Holts model object using `train` CPI
- Fitting the model to an optimized object
- Create prediction dataframe for `test`
- Use the model object to make prediction on  `test`  sample.
- Add model predictions to dataframe
- Evaluate the performce of the model by using root mean square error (RMSE) on the predictions
- Print RMSE
- Plot actual obeservations and predections 
- Review the results of model performance on  `test` sample

### Conclusion

- Summary of the project
    - The Holts model performed on `test` sample by a margin of error of 8.
    - Looking at the predictions plot, The model is underperforming and will continue to increase with a margin of error over time.
    - I belive the decrease in the models performance has to do with the vass difference of `train` and `test`. The `train` sample size is six time greater.
    - I believe the model initally performed with a small margin of error due to both `train` and `validate` are the same size.


- Recommendation
    - Train the holts model on smaller periods of time to improve efficiency

- Next Steps
    - Add other elements of CPI for categories like rent, unleaded gasoline, electricity, 
    - Figure a way to incorparte population to the data. Population tends to impact the CPi
    - Information about wages of workers in region
    - Acquire 2022 year data to do further analysis once modifications of model is completed

