import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# working with dates
from datetime import datetime

# to evaluated performance using rmse
from sklearn.metrics import mean_squared_error
from math import sqrt 


# holt's linear trend model. 
from statsmodels.tsa.api import Holt

#### Acquisition of data

# Overall CPI for Houston Metro Area
def get_houston_overall_cpi():
    # imports  downloaded xlsx file from BLS website to python
    houston_overall_cpi = pd.read_excel('houston_metro_CPI_all_item_09_to_21.xlsx',index_col=None, header= 11)
    return houston_overall_cpi

#### Preparation and cleaning of data

#
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

# Spliting data into three samples

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

### Modeling

# evaluate() will compute the Mean Squared Error and the Root Mean Squared Error to evaluate.
def evaluate( target_var):
    import acquire
    houston_overall_cpi = acquire.get_houston_overall_cpi()
    df = acquire.prep_houston_overall()
    train, validate, test = acquire.split_data(df)
    '''
    This function will take the actual values of the target_var from validate, 
    and the predicted values stored in yhat_df, 
    and compute the rmse, rounding to 0 decimal places. 
    it will return the rmse. 
    '''
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 3)
    return rmse

# plot_and_eval() will use the evaluate function and also plot train and test values with the predicted values in order to compare performance.
def plot_and_eval(target_var):
    '''
    This function takes in the target var name (string), and returns a plot
    of the values of train for that variable, validate, and the predicted values from yhat_df. 
    it will als lable the rmse. 
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label='Train', linewidth=1)
    plt.plot(validate[target_var], label='Validate', linewidth=1)
    plt.plot(yhat_df[target_var])
    plt.title(target_var)
    rmse = evaluate(target_var)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.show()

# function to store the rmse so that we can compare
def append_eval_df(model_type, target_var):
    '''
    this function takes in as arguments the type of model run, and the name of the target variable. 
    It returns the eval_df with the rmse appended to it for that model and target_var. 
    '''
    rmse = evaluate(target_var)
    d = {'model_type': [model_type], 'target_var': [target_var],
        'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)  


def model_results():
    # create an empty dataframe
    eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'rmse'])
    return eval_df

#Apply simple average predictions to our observations
def make_predictions(sales=None, quantity=None):
    yhat_df = pd.DataFrame({'CPI': [avg_cpi]}, index=validate.index)
    return yhat_df

#Apply moving average predictions to our observations
def moving_avg_make_predictions(sales=None, quantity=None):
    yhat_df = pd.DataFrame({'CPI': [rolling_cpi]}, index=validate.index)
    return yhat_df


#### Best Model testing

# evaluation function to compute rmse
def evaluate_test(target_var):
    rmse = round(sqrt(mean_squared_error(test[target_var], yhat_df[target_var])), 0)
    return rmse

def plot_and_eval_test(target_var):
    '''
    This function takes in the target var name (string), and returns a plot
    of the values of train for that variable, validate, and the predicted values from yhat_df. 
    it will als lable the rmse. 
    '''
    plt.figure(figsize = (12,4))
    plt.plot(test[target_var], label='Actual', linewidth=1)
    plt.plot(yhat_df[target_var], label='Prediction', linewidth=1)
    plt.title(target_var)
    rmse = evaluate_test(target_var)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.legend()
    plt.show()

