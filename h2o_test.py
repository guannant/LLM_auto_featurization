import h2o
from h2o.estimators import H2OGradientBoostingEstimator

# Start an H2O cluster
h2o.init()

# Load a sample dataset from H2O
data = h2o.import_file("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# Split into train and test
train, test = data.split_frame(ratios=[0.8], seed=1234)

# Define predictors and response
x = data.columns[:-1]   # all columns except last
y = data.columns[-1]    # species

# Convert response column to categorical
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

# Train a Gradient Boosting model
model = H2OGradientBoostingEstimator(ntrees=50, max_depth=5, seed=1234)
model.train(x=x, y=y, training_frame=train)

# Evaluate model performance
perf = model.model_performance(test_data=test)
print(perf)

h2o.shutdown(prompt=False)
 