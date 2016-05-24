import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression


df = quandl.get('WIKI/GOOGL')

# take a look at the data set
#print(df.head())

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

# define 'special' relationships between features to eliminate redundancy
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100.0


#define new data frame with new combined features
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

# prediction
forecast_col = 'Adj. Close'
# replaces NAN data with value - cannot work with 
# better than getting rid of row 
df.fillna(-99999, inplace=True)

#try to predict out 10% of the data frame
forecast_out = int(math.ceil(0.01*len(df)))

# create label
# basically shifting the column
# shifting the columns negatively (by forecast_out) -> the label column
# for each row will be the adjusted close price 10 days into the future
# "our features are these attributes of what we think may cause the adjusted close 
# price in 10 to change"
df['label'] = df[forecast_col].shift(-forecast_out)

# drops NAN where we dont have full label data bc of shift
df.dropna(inplace=True)


# features represented by X
# df.drop returns a new data frame that is converted into a np array
X = np.array(df.drop(['label'],1))

# label array
y = np.array(df['label'])

# scaling X before feeding to classifier
X = preprocessing.scale(X)
y = np.array(df['label'])



X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)


#train classifier on training set using LinearRegression
clf = LinearRegression(n_jobs=-1)

# train classifier on training set using SVM
#clf = svm.SVR()
clf.fit(X_train,y_train)

# try out new classifier on test set
accuracy = clf.score(X_test,y_test)

print(accuracy)