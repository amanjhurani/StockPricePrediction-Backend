
import pandas_datareader as pdr
import pandas as pd

def downloadStockData(company):
    key="df4cbd2f21573774d53a88d482c1ae50785b1c0f"
    df = pdr.get_data_tiingo(company, api_key=key)
    df.to_csv('./StockData/'+ company+'.csv')

def getStockData(company):
    df=pd.read_csv('./StockData/'+ company+'.csv')
    return df