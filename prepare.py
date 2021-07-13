import seaborn as sns
import os
from pydataset import data
from scipy import stats
import pandas as pd
import numpy as np

#train validate, split
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

# metrics and confusion
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#model classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ignore warnings

# ignore warnings
import warnings
warnings.filterwarnings("ignore")







#function to look create df which shows records containing nulls
def view_null_records(df, variable):
    """
    function allows you to records which contain null, nan values.
    REMEMBER, will only work for individual column and if that columns has nulls, 
    otherwise will return empty dataframe
    """
    df[df[variable].isna()]
    
    return df[df[variable].isna()]


def create_dummies(df):
    '''
    This function is used to create dummy columns for my non binary columns
    '''
    # create dummies for payment_type, internet_service_type, and contract_type
    time_dummies = pd.get_dummies(df.time_of_day, drop_first=False)
    wind_dummies = pd.get_dummies(df.wind_direction, drop_first=False)
    

    # now we concatenate our dummy dataframes with the original
    df = pd.concat([df, time_dummies], axis=1)
    df = pd.concat([df, wind_dummies], axis=1)
    

    return df

####################################
                #outlier finding###
#####################################

def outlier_bound_calculation(df, variable):
    '''
    calcualtes the lower and upper bound to locate outliers in variables
    '''
    quartile1, quartile3 = np.percentile(df[variable], [25,75])
    IQR_value = quartile3 - quartile1
    lower_bound = quartile1 - (1.5 * IQR_value)
    upper_bound = quartile3 + (1.5 * IQR_value)
    '''
    returns the lowerbound and upperbound values
    '''
    return print(f'For {variable} the lower bound is {lower_bound} and  upper bound is {upper_bound}')


def detect_outliers(df, k, col_list):
    ''' get upper and lower bound for list of columns in a dataframe 
        if desired return that dataframe with the outliers removed
    '''
    
    odf = pd.DataFrame()
    
    for col in col_list:

        q1, q2, q3 = df[f'{col}'].quantile([.25, .5, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound
        
        # print each col and upper and lower bound for each column
        print(f"{col}: Median = {q2} lower_bound = {lower_bound} upper_bound = {upper_bound}")

        # return dataframe of outliers
        odf = odf.append(df[(df[f'{col}'] < lower_bound) | (df[f'{col}'] > upper_bound)])
            
    return odf



def show_outliers(df, k, columns):
    '''
    calculates the lower and upper bound to locate outliers and displays them
    recommended k be 1.5 and entered as integer
    '''
    for i in columns:
        quartile1, quartile3 = np.percentile(df[i], [25,75])
        IQR_value = quartile3 - quartile1
        lower_bound = (quartile1 - (k * IQR_value))
        upper_bound = (quartile3 + (k * IQR_value))
        print(f'For {i} the lower bound is {lower_bound} and  upper bound is {upper_bound}')
        
        
def remove_outliers(df, k, columns):
    '''
    calculates the lower and upper bound to locate outliers in variables and then removes them.
    recommended k be 1.5 and entered as integer
    '''
    for i in columns:
        quartile1, quartile3 = np.percentile(df[i], [25,75])
        IQR_value = quartile3 - quartile1
        lower_bound = (quartile1 - (k * IQR_value))
        upper_bound = (quartile3 + (k * IQR_value))
        print(f'For {i} the lower bound is {lower_bound} and  upper bound is {upper_bound}')
        df = df[(df[i] <= upper_bound) & (df[i] >= lower_bound)]
        print('-----------------')
        print('Dataframe now has ', df.shape[0], 'rows and ', df.shape[1], 'columns')
    return df

def remove_outliers_noprint(df, k, columns):
    '''
    calculates the lower and upper bound to locate outliers in variables and then removes them.
    recommended k be 1.5 and entered as integer
    '''
    for i in columns:
        quartile1, quartile3 = np.percentile(df[i], [25,75])
        IQR_value = quartile3 - quartile1
        lower_bound = (quartile1 - (k * IQR_value))
        upper_bound = (quartile3 + (k * IQR_value))

        df = df[(df[i] <= upper_bound) & (df[i] >= lower_bound)]
        
    return df
        
        


    
  
    
##############################################################################
# extract objects or numerical based columns 


def get_object_cols(df):
    """
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names.
    """
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()

    return object_cols


def get_numeric_X_cols(X_train, object_cols):
    """
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects.
    """
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]

    return numeric_cols

# Generic splitting function for continuous target.
def split_continuous(df):
    '''
    Takes in a df
    Returns train, validate, and test DataFrames
    '''
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, 
                                        test_size=.2, 
                                        random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, 
                                   test_size=.3, 
                                   random_state=123)

    # Take a look at your split datasets

    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')
    return train, validate, test





##############################################################################
def model_metrics(X_train, y_train, X_validate, y_validate):
    '''
    this function will score models and provide confusion matrix.
    returns classification report as well as evaluation metrics.
    '''
    lr_model = LogisticRegression(random_state =1349)
    dt_model = DecisionTreeClassifier(max_depth = 2, random_state=1349)
    rf_model = RandomForestClassifier(max_depth=4, min_samples_leaf=3, random_state=1349)
    kn_model = KNeighborsClassifier()
    models = [lr_model, dt_model, rf_model]
    for model in models:
        #fitting our model
        model.fit(X_train, y_train)
        #specifying target and features
        train_target = y_train
        #creating prediction for train and validate
        train_prediction = model.predict(X_train)
        val_target = y_validate
        val_prediction = model.predict(X_validate)
        # evaluation metrics
        TN_t, FP_t, FN_t, TP_t = confusion_matrix(y_train, train_prediction).ravel()
        TN_v, FP_v, FN_v, TP_v = confusion_matrix(y_validate, val_prediction).ravel()
        #calculating true positive rate, false positive rate, true negative rate, false negative rate.
        tpr_t = TP_t/(TP_t+FN_t)
        fpr_t = FP_t/(FP_t+TN_t)
        tnr_t = TN_t/(TN_t+FP_t)
        fnr_t = FN_t/(FN_t+TP_t)
        tpr_v = TP_v/(TP_v+FN_v)
        fpr_v = FP_v/(FP_v+TN_v)
        tnr_v = TN_v/(TN_v+FP_v)
        fnr_v = FN_v/(FN_v+TP_v)
        
        
        
        print('--------------------------')
        print('')
        print(model)
        print('train set')
        print('')
        print(f'train accuracy: {model.score(X_train, y_train):.2%}')
        print('classification report:')
        print(classification_report(train_target, train_prediction))
        print('')
        print(f'''
        True Positive Rate:{tpr_t:.2%},  
        False Positive Rate :{fpr_t:.2%},
        True Negative Rate: {tnr_t:.2%},  
        False Negative Rate: {fnr_t:.2%}''')
        print('------------------------')
        
        print('validate set')
        print('')
        print(f'validate accuracy: {model.score(X_validate, y_validate):.2%}')
        print('classification report:')
        print(classification_report(y_validate, val_prediction))
        print('')
        print(f'''
        True Positive Rate:{tpr_v:.2%},  
        False Positive Rate :{fpr_v:.2%},
        True Negative Rate: {tnr_v:.2%},  
        False Negative Rate: {fnr_v:.2%}''')
        print('')
        print('------------------------')
        
        
        ####################################################################
# Train, validate, split which doesnt exclude target from train. 
# target is categorical 


#genreal split when categorical
def general_split(df, stratify_var):
    '''
    This function take in the telco_churn_data acquired by get_telco_churn,
    performs a split and stratifies total_charges column. Can specify stratify as None which will make this useful for continous.
    Returns train, validate, and test dfs.
    '''
    #20% test, 80% train_validate
    train_validate, test = train_test_split(df, test_size=0.2, 
                                        random_state=1349, stratify = stratify_var)
    # 80% train_validate: 30% validate, 70% train.
    train, validate = train_test_split(train_validate, train_size=0.7, 
                                   random_state=1349, stratify = stratify_var)
    """
    returns train, validate, test 
    """
    return train, validate, test






##################################################
# train, validate, split #
#  generates features and target.
################################################

def train_validate_test(df, target, stratify):
    """
    this function takes in a dataframe and splits it into 3 samples,
    a test, which is 20% of the entire dataframe,
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe.
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable.
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    """
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123, stratify = stratify)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123, stratify = stratify)

    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]

    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    '''    
    Returns X_train, y_train, X_validate, y_validate, X_test, y_test
    '''

    return X_train, y_train, X_validate, y_validate, X_test, y_test

################################################################




########################################
#                scaling

def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    """
    this function takes in 3 dataframes with the same columns,
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler.
    it returns 3 dataframes with the same column names and scaled values.
    """
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).

    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    # scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train.
    #
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=numeric_cols).set_index(
        [X_train.index.values]
    )

    X_validate_scaled = pd.DataFrame(
        X_validate_scaled_array, columns=numeric_cols
    ).set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=numeric_cols).set_index(
        [X_test.index.values]
    )
    
    """
    returns X_train_scaled, X_validate_scaled, X_test_scaled 
    """

    return X_train_scaled, X_validate_scaled, X_test_scaled
        
    
 
def prep_solar(df):
    # will not utilize UNIXTime
    df.drop(columns = 'UNIXTime', inplace = True)
    # Data Needs to be renamed to Date
    df.rename(columns = {'Data':'Date', 'Radiation':'solar_irradiance'}, inplace = True)
    
    #added in column in datetime format, will convert these new columns to a hour and min then change to string and combine them.
    df['Date'] = pd.to_datetime(df.Date)
    df['Timehr'] = pd.to_datetime(df.Time).dt.hour
    df['Timehr'] = df.Timehr.astype(str)
    #######
    df['Timemin'] = pd.to_datetime(df.Time).dt.minute
    df['Timemin'] = df.Timemin.astype(str)
    #######
    df['Timesunhr'] = pd.to_datetime(df.TimeSunRise).dt.hour
    df['Timesunmin'] = pd.to_datetime(df.TimeSunRise).dt.minute
    df['Timesunhr'] = df.Timesunhr.astype(str)
    df['Timesunmin'] = df.Timesunmin.astype(str)
    #####
    df['Timesethr'] = pd.to_datetime(df.TimeSunSet).dt.hour
    df['Timesetmin'] = pd.to_datetime(df.TimeSunSet).dt.minute
    df['Timesethr'] = df.Timesethr.astype(str)
    df['Timesetmin'] = df.Timesetmin.astype(str)
    
    #combining 
    df['Time'] = (df.Timehr + '.' + df.Timemin)
    df['TimeSunRise'] = (df.Timesunhr + '.' + df.Timesunmin)
    df['TimeSunSet'] = (df.Timesethr + '.' + df.Timesetmin)
    #dropping several columns
    df.drop(columns = {'Timehr', 'Timemin', 'Timesunhr', 'Timesunmin', 'Timesethr',
       'Timesetmin'}, inplace = True)
    cols = ['TimeSunRise', 'TimeSunSet', 'Time']
    df[cols] = df[cols].astype(float)
    
    #creating new column which is day, if 0 then night, if 1 is day
    df['day'] = np.where((df.Time < df.TimeSunSet) & (df.Time > df.TimeSunRise), 1, 0)
    #dropping Sunset and Rise columns
    df.drop(columns = ['TimeSunRise', 'TimeSunSet'],inplace = True)
    ## remove outliers
    cols = df.drop(columns = ['day','Date','Time'])
    cols = cols.columns.tolist()
    df = remove_outliers(df,1.5,cols)
    
    #binning degrees to have specificy wind direction
    df['wind_direction'] = pd.cut(df['WindDirection(Degrees)'], [0,23,68,113,158,203,248,293,336,360], labels = ['N','NE','E','SE','S','SW','W','NW','N'], ordered = False)
    #drop wind direction
    df.drop(columns = 'WindDirection(Degrees)', inplace = True)
    #bin day timeframes
    df['time_of_day'] = pd.cut(df['Time'], [-0.01, 06.00, 10.00, 12.00,14.00,16.00,18.00,24.00], labels = ['0.00-06.00', '06.00-10.00','10.00-12.00','12.00-14.00','14.00-16.00','16.00-18.00','18.00--24.00'])
    #drop time
    df.drop(columns = {'Time'}, inplace = True)
    #create dummies
    df =create_dummies(df)
    #drop columns
    df.drop(columns = ['wind_direction', 'time_of_day'], inplace = True)
    
    return df



    
