import pandas as pd
import requests

def API_single_inputs(url, name):
    '''
    Can input url and then look at keys and specify certain keys and feature.
    '''
    response = requests.get(url)
    data = response.json()
    print('please enter key')
    print(data.keys())
    print(data[str(input())].keys())
    print('please input key and key of nested')
    df = pd.DataFrame(data[str(input())][str(input())])
    df.to_csv(name + '.csv')
    
    

import pandas as pd
import requests
import os




def json_to_dataframe(url, page_name, key1, key2= None, key3= None):
    ''' 
    Takes in a json api and returns json api as a dataframe.
    json_to_dataframe(
    url: base url,
    page_name: the page name to be added to api url 
    key1: key from dataframe,
    key2= None,
    key3= None,
    )
    '''
    items_list = []
    
    # Let's take an example url and make a get request
    response = requests.get(url)
    #create dictionary object
    data= response.json()
    
    n = data[key1][key2]
    
    if (key2 != None) & (key3 != None):
        #Adding 1 here so the last digit is not cut off (not inclusive)
        for i in range(1,n+1):
            url = f'https://python.zach.lol/api/v1/{page_name}?page='+str(i)
            response = requests.get(url)
            data = response.json()
            page_items = data[key1][key3]
            items_list += page_items
    else:
        for i in range(1,n+1):
            url = f'https://python.zach.lol/api/v1/{page_name}?page='+str(i)
            response = requests.get(url)
            data = response.json()
            page_items = data[key1]
            items_list += page_items
    
    df= pd.DataFrame(items_list)
    return df




def get_data(csv, url, page_name, key1, key2= None, key3= None, cached=False):
    '''
    This function reads in df using json_to_dataframe function and writes data to
    a csv file if cached == False or if cached == True reads in df from csv file present
    and returns df.
    '''
    if cached == False or os.path.isfile(csv) == False:
        
        #Read fresh data from db into a DataFrame.
        df= json_to_dataframe(url, page_name, key1, key2, key3)
        
        # Write DataFrame to a csv file.
        df.to_csv(csv)
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv(csv, index_col=0)
        
    return df




def combined_dataframes():
    '''
    This function reads in items, stores, and sales csv files, creates a them as 
    dataframes using pandas. The dataframes are joined using merge function from pandas,
    newly joined dataframes are returned as one dataframe.
    '''
    #Bring in items csv using pandas
    items_df_csv = pd.read_csv("items_df.csv")
    #Bring in store csv using pandas
    stores_df_csv = pd.read_csv("stores_df.csv")
    #Bring in sales csv using pandas
    sales_df_csv = pd.read_csv("sales_df.csv")
    #Merging sales df and store df using pandas
    sales_and_stores_df= pd.merge(sales_df_csv, stores_df_csv, left_on='store', right_on='store_id', how='left')
    #Merging sales_and_store_df with items_df using pandas
    sales_stores_items_df= pd.merge(sales_and_stores_df, items_df_csv, left_on='store_id', right_on='store_id', how='left')
    
    return sales_stores_items_df





def get_germany_data(url, cached=False):
    '''
    This function reads in url for germany data and writes data to
    a csv file if cached == False or if cached == True reads in germany df from
    a csv file, returns df.
    '''
    germany_df = pd.read_csv(url)
    
    if cached == False or os.path.isfile('germany_df.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = germany_df
        
        # Write DataFrame to a csv file.
        df.to_csv('germany_df.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('germany_df.csv', index_col=0)
        
    return df


####################

def solar_data():
    '''
    This function reads the solar_predictions data from the csv file which is available on kaggle.
    link to data: https://www.kaggle.com/dronio/SolarEnergy?select=SolarPrediction.csv
    '''
    # Read in DataFrame from csv downloaded from kaggle.
    df = pd.read_csv('SolarPrediction.csv')
    
    return df




def get_solar(cached=False):
    '''
    This function reads in url for solar_prediction data and writes data to
    a csv file if cached == False or if cached == True reads in solar_df from a csv file, returns df.
    '''
    solar_df = solar_data()
    if os.path.isfile('SolarPrediction.csv') == False:
        # Read fresh data from db into a DataFrame.
        df = solar_df
        # Write DataFrame to a csv file.
        df.to_csv('SolarPrediction.csv')
    else:
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('SolarPrediction.csv', index_col=0)
    return df

