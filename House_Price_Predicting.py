import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
matplotlib.use('TkAgg')

Dir = '/Users/Yu/Desktop/Machine Learning/home_data.csv'

# exact columns using pandas
data = pd.read_csv(Dir)
zip_price = list(zip(data.zipcode, data.price))   # get zipcode and price
df = pd.DataFrame(data=zip_price, columns=['Zipcode', 'Price'])         # raw data
sumPrice = df.groupby('Zipcode', as_index=False).aggregate(np.average)  # get average price in every area

# 1, GET THE MOST EXPENSIVE AVERAGE PRICE AREA AND IT'S AVERAGE PRICE
maxAveragePrice = 0
maxZipCodeArea = 0
for idx, row in sumPrice.iterrows():
    if row['Price'] > maxAveragePrice:
        maxAveragePrice = row['Price']
        maxZipCodeArea = int(row['Zipcode'])


print('Zipcode of Max average house price area: ', maxZipCodeArea)
print('It\'s maximum average house Price: ', maxAveragePrice)

# Plot the result and visualize it
newList = list(zip(sumPrice['Zipcode'], sumPrice['Price']))
x_zip = []
y_price = []
for ele in newList:
    x_zip.append(ele[0])
    y_price.append(ele[1])

# histogram
# bins = []   # range of y coordinates that will be displayed
# plt.hist(y_price, bins, histtype='bar', rwidth=0.2)

# bars
fig = plt.figure()
plt.bar(x_zip, y_price, label='zip-avg_price', color='b')

plt.xlabel('Zipcode')
plt.ylabel('Price')
plt.title('Area - Price Distribution')
# plt.show()
plt.savefig('zip_price.png')

# 2, GET FRACTION OF 'sqrt_living' = [2000, 4000] OUT OF ALL HOUSES

df2 = pd.DataFrame(data=data)
inRange = df2[(df2['sqft_living'] > 2000) & (df2['sqft_living'] <= 4000)]       # house in range[2000, 4000]
fraction = len(inRange.index) / len(df2.index)
print('Ratio of square feet house range[2000, 4000] out of all houses: ', fraction)

# 3, BUILDING A REGRESSION MODEL WITH SEVERAL FEATURES

my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
advanced_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode', 'condition', 'grade',
                     'waterfront', 'view', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
                     'sqft_living15', 'sqft_lot15']

df4 = pd.DataFrame(data=data)
train, test = train_test_split(df4, test_size=0.2)          # Split data to test and train
regr = linear_model.LinearRegression()                      # define a regression obj

# TRAIN WITH 6 FEATURES
train_X1 = train[my_features]               # get x and y
train_Y1 = train['price']
test_X1 = test[my_features]
test_Y1 = test['price']

regr.fit(train_X1, train_Y1)
pred_1 = regr.predict(test_X1)

print('Estimated intercept coefficient: \n', regr.coef_)
print('Number of coefficients: ', len(regr.coef_))
print("Mean squared error: %.2f" % mean_squared_error(test_Y1, pred_1))
print('Variance score: %.2f' % r2_score(test_Y1, pred_1))
'''
# scatter plot bathrooms - price
plt.scatter(df4['sqft_living'], df4['price'])

plt.xlabel("sqft_living")
plt.ylabel("House price")
plt.title("sqft_living - house price")
plt.show()
'''
# TRAIN WITH MANY FEATURES

train_X2 = train[advanced_features]
train_Y2 = train['price']
test_X2 = test[advanced_features]
test_Y2 = test['price']

regr.fit(train_X2, train_Y2)
pred_2 = regr.predict(test_X2)
'''
print('Estimated intercept coefficient: \n', regr.coef_)
print('Number of coefficients: ', len(regr.coef_))

# Plot
plt.scatter(test_X2['bedrooms'], test_Y2,  color='black')
plt.plot(test_X2['bedrooms'], pred_2, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
'''

print("MSE diff: ", mean_squared_error(test_Y1, pred_1) - mean_squared_error(test_Y2, pred_2))