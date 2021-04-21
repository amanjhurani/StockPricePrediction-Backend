import model_train
import data


df = data.getStockData('AAPL')

# model_train.trainStockModel('AAPL', df)
model_train.getStockPrediction('AAPL', df)


