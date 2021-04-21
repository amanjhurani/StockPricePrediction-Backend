
import pandas_datareader as pdr
import pandas as pd

def downloadStockData(company):
    key="APIKEY"
    df = pdr.get_data_tiingo(company, api_key=key)
    df.to_csv('./StockData/'+ company+'.csv')

def getStockData(company):
    df=pd.read_csv('./StockData/'+ company+'.csv')
    return df