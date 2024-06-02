import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("./cars_CO2_emission.csv")

features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']
data = df[features]

train, test = train_test_split(data, test_size=0.2, random_state=42)

x_train = train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
y_train = train['CO2EMISSIONS']
x_test = test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
y_test = test['CO2EMISSIONS']

model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

print('Coefficients:', model.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, predictions))
print('Variance score: %.2f' % r2_score(y_test, predictions))

plt.scatter(y_test, predictions)
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("Actual vs. Predicted CO2 Emissions")
plt.show()

plt.scatter(data['ENGINESIZE'], data['CO2EMISSIONS'], color='blue')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.title("Engine Size vs. CO2 Emissions")
plt.show()

plt.scatter(data['CYLINDERS'], data['CO2EMISSIONS'], color='green')
plt.xlabel("Cylinders")
plt.ylabel("CO2 Emissions")
plt.title("Cylinders vs. CO2 Emissions")
plt.show()

plt.scatter(data['FUELCONSUMPTION_COMB'], data['CO2EMISSIONS'], color='red')
plt.xlabel("Fuel Consumption (Combined)")
plt.ylabel("CO2 Emissions")
plt.title("Fuel Consumption vs. CO2 Emissions")
plt.show()
