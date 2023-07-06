from matplotlib import pyplot as plt
from matplotlib import dates as dt
from pathlib import Path
import pandas as pd
import numpy as np
import os
from datetime import date
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#simple moving average computing
def SMA(close, timeframe):
    sma = [0.0]*timeframe
    aux = 0.0
    for i in range(0,timeframe):
        aux = aux + close[i]
    sma.append(aux/timeframe)
    for i in range(timeframe+1,len(close)):
        sma.append((sma[i-1]*timeframe + close[i] - close[i-timeframe])/timeframe)
    return sma


#smple moving average plotting for 10, 20, 50 and 200 days
def SMAplot(type, name):
    #function that plots the simple moving average for timeframes 10, 20, 50, 200
    #it must read the desired file, analyse the data and plot the SMAs
    str = os.path.join("Data", type + 's', name + ".us.txt")
    data = pd.read_csv(str)
    date_array  = pd.to_datetime(data['Date'])
    close_array = data['Close']
    sma10 = SMA(close_array, 10)
    sma20 = SMA(close_array, 20)
    sma50 = SMA(close_array, 50)
    sma200 = SMA(close_array, 200)
    
    ax = plt.subplot()
    ax.plot(date_array, close_array, linestyle = 'dotted', color = 'black', label = 'close')
    ax.plot(date_array, sma10, linestyle = 'solid', color = 'red', label = '10 days')
    ax.plot(date_array, sma20, linestyle = 'solid', color = 'blue', label = '20 days')
    ax.plot(date_array, sma50, linestyle = 'solid', color = 'green', label = '50 days')
    ax.plot(date_array, sma200, linestyle = 'solid', color = 'yellow', label = '200 days')

    plt.title(
    name + ".us.txt",
    fontsize='large',
    loc='center',
    fontweight='bold',
    style='italic',
    family='monospace')
    ax.legend()
    plt.show()

#plots a set random ETFs and Stocks
def plots():
    SMAplot("Stock","wyy")
    SMAplot("Stock","googl")
    SMAplot("Stock","aaap")
    SMAplot("Stock","ibm")
    SMAplot("Stock","evep")
    SMAplot("ETF","bbh")
    SMAplot("ETF","fex")
    SMAplot("ETF","rtm")
    SMAplot("ETF","tur")
    SMAplot("ETF","zroz")

#correlation function
#calculate the correlation between stocks during the specified timeframe
#default value for ending date is the end of the dataset
#it can also calculate correlation between a chosen set of stocks, passed as parameters 

def correlation(startingDate, listOfStocks = [], endingDate = '2017-11-10'):

    #we will calculate correlation only between the starting and end dates of the stocks 
    #with the same number of entries in the timeframe.

    #if they do not have data in the starting and ending dates, we do not count it as a valid stock
    #we then only get the closing values in the timeframe

    #array to store the names of the stocks (order may change )
    stockNames = []
    #matrix to store the prices of each stock in the timeframe
    stocks = [[]]
    #index of stocks
    a = 0

    
    bdays = pd.tseries.offsets.BDay()
    startDate = date(int(startingDate[0:4]),int(startingDate[5:7]),int(startingDate[8:])) + 0*bdays
    endDate = date(int(endingDate[0:4]),int(endingDate[5:7]),int(endingDate[8:])) + 0*bdays
    start = str(startDate.year)+'-'+str(startDate.month).zfill(2)+'-'+str(startDate.day).zfill(2)
    end = str(endDate.year)+'-'+str(endDate.month).zfill(2)+'-'+str(endDate.day).zfill(2)

    #read data from the specified stocks
    if(len(listOfStocks) != 0 ):
        for s in listOfStocks:
            st = os.path.join("Data", 'Stocks', s + ".us.txt")
            data = pd.read_csv(st)
            if start in data['Date'].values and end in data['Date'].values:
                    data = data[data['Date'] >= startingDate]
                    data = data[data['Date'] <= endingDate]
                    close_array = data['Close']
                    stockNames.append(s)
                    stocks.append([0,0])
                    stocks[a] = close_array
                    print(stocks[a])
                    a = a + 1

    #read whole database of stocks
    else:
        for child in Path('Data\Stocks').iterdir():
            if child.is_file():
                name = str(child)[14:-7]    
                st = os.path.join(child)
                data = pd.read_csv(st)
                if start in data['Date'].values and end in data['Date'].values:
                    data = data[data['Date'] >= startingDate]
                    data = data[data['Date'] <= endingDate]
                    close_array = data['Close']
                    stockNames.append(name)
                    stocks.append([0,0])
                    stocks[a] = close_array
                    a = a + 1

    corrMatrix = np.zeros((len(stockNames),len(stockNames)))

    #calculates the correlation
    #if the number of data does not match, sets an invalid value
    for j in range(0,len(stockNames)):
        for i in range(j,len(stockNames)):
            if len(stocks[i]) != len(stocks[j]):
                corrMatrix[i][j] = -2
            else:
                corrMatrix[i][j] = np.corrcoef(stocks[j],stocks[i])[1][0]


    maxCorr = [(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]
    minCorr = [(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]
    
    #calculate the 5 greatest and lowest correlations
    for j in range(0,len(stockNames)):
        for i in range(j+1,len(stockNames)):
            if corrMatrix[i][j] > maxCorr[4][0]:
                maxCorr[4] = (corrMatrix[i][j],stockNames[i],stockNames[j])
                maxCorr.sort(key=lambda x: x[0],reverse=True)
            if corrMatrix[i][j] < minCorr[4][0] and corrMatrix[i][j]!=-2:
                minCorr[4] = (corrMatrix[i][j],stockNames[i],stockNames[j])
                minCorr.sort(key=lambda x: x[0])

    #print the results
    print(stockNames) #used together with the correlation matrix 
    print(corrMatrix) #the rows and columns are ordered by the stockNames vector

    print(maxCorr)
    print(minCorr)

def regression(name, train_size):
    # Read the stock data from the file
    filepath = f"Data/Stocks/{name}.us.txt"
    stock = pd.read_csv(filepath)
    stock['Date'] = pd.to_datetime(stock['Date'])

    # Extract the closing values
    close_values = stock['Close'].values.reshape(-1, 1)

    # Split the closing values into train and test sets
    train_end_index = int(train_size * len(close_values))
    train_close = close_values[:train_end_index]
    test_close = close_values[train_end_index:]

    # Prepare the training data
    X_train = np.arange(len(train_close)).reshape(-1, 1)
    y_train = train_close

    # Prepare the test data
    X_test = np.arange(len(train_close), len(close_values)).reshape(-1, 1)
    y_test = test_close

    # Perform polynomial regression
    # Degrees used: AAPL - 4, MSFT - 4, GOOGL  - 2, GOOG - 2 
    #(i would guess that higher degrees have better prediction, but must have more data to work well) 
    polynomial_features = PolynomialFeatures(degree=2)
    X_train_poly = polynomial_features.fit_transform(X_train)
    X_test_poly = polynomial_features.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Predict the next closing values
    predicted_values = model.predict(X_test_poly)

    # Plot actual and predicted closing values
    plt.plot(stock['Date'], close_values, label='Actual')
    plt.plot(stock['Date'][train_end_index:], predicted_values, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('Stock Price Prediction - ' + str(name).upper())
    plt.legend()
    plt.show()


def main():
    plots()

    correlation('2015-03-15')
    correlation('2017-01-03',{'goog','googl','msft','aapl','ibm','ge','aal','twtr','f','fb','amzn'})

    regression('goog',0.8)



main()
