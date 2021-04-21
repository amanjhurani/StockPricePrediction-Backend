import model_train
import data


df = data.getStockData('GOOGL')

model_train.trainStockModel('GOOGL', df)


