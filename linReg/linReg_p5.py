import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

import pickle

style.use('ggplot')


df = quandl.get('WIKI/GOOGL')

# take a look at the data set
#print(df.head())

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

# define 'special' relationships between features to eliminate redundancy
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100.0


#define new data frame with new combined features
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]


forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)


forecast_out = int(math.ceil(0.1*len(df)))
print(forecast_out)


df['label'] = df[forecast_col].shift(-forecast_out)




X = np.array(df.drop(['label', 'Adj. Close'],1))
X = preprocessing.scale(X)
# we will predict against X_lately (last 30 days of data)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]


df.dropna(inplace=True)
y = np.array(df['label'])




X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)


clf = LinearRegression(n_jobs=-1)
clf.fit(X_train,y_train)
with open('linearregression.pickle','wb') as f:
	pickle.dump(clf,f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test,y_test)

#print(accuracy)

#predictions
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# iterating through the forecast set, taking each forecast and day
# setting those as the values in the df, basically making the future features nan
# the last line takes all of the first column and sets them to nan the sets the 
# last column to whatever i is.
for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	# next date is a datestamp and next_date is the index of the df
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
