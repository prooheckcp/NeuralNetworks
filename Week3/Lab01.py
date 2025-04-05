# for array computations and loading data
import numpy as np

# for building linear regression models and preparing data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# for building and training neural networks
import tensorflow as tf

# custom functions
import utils

# reduce display precision on numpy arrays
np.set_printoptions(precision=2)

# suppress warnings
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# Load the dataset from the text file
data = np.loadtxt('./data/data_w3_ex1.csv', delimiter=',')

# Split the inputs and outputs into separate arrays
x = data[:,0]
y = data[:,1]

# Convert 1-D arrays into 2-D because the commands later will require it
x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)

# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=1)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

# Delete temporary variables
del x_, y_

# Pre-processor classes
scaler_linear = StandardScaler()
scaler_poly = StandardScaler()
poly = PolynomialFeatures(degree=2, include_bias=False)

# Linear regression model
X_train_scaled = scaler_linear.fit_transform(x_train) # Compute the mean and standard deviation of the training set then transform it
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
X_cv_scaled = scaler_linear.transform(x_cv)
yhat = linear_model.predict(X_cv_scaled)

#
X_train_mapped_scaled = scaler_poly.fit_transform(x_train)
model = LinearRegression()
model.fit(X_train_mapped_scaled, y_train)
yhat = model.predict(X_train_mapped_scaled)

X_cv_mapped = poly.fit_transform(x_cv)
X_cv_mapped_scaled = scaler_poly.fit_transform(X_cv_mapped)
yhat = model.predict(X_cv_mapped_scaled)
# Doesn't work