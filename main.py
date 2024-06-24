import pandas as pd
import numpy as np
import math
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from joblib import dump, load

data = pd.read_excel("Net_Worth_Data.xlsx")

if __name__ == "__main__":
    # print the first few rows
    print(data.head())

    # print the last few rows
    print(data.tail())

    # print the shape of the data
    print(data.shape)

    # print a summary of the dataset
    data.info()

# anonymise the data
anonymised_data = data.drop(["Client Name", "Client e-mail", "Profession", "Education", "Country"], axis=1)
data_input = anonymised_data.drop(["Net Worth"], axis=1).values
data_output = anonymised_data["Net Worth"].values.reshape(-1, 1)

if __name__ == "__main__":
    # print out data shape
    print(data_input.shape)
    print(data_output.shape)

input_scalar = StandardScaler()
scaled_input = input_scalar.fit_transform(data_input)

output_scalar = StandardScaler()
scaled_output = output_scalar.fit_transform(data_output)

if __name__ == "__main__":
    random_state = 42
    train_input, test_input, train_output, test_output = train_test_split(scaled_input, scaled_output, test_size=0.2, random_state=random_state)

    # the models to train
    models = [
        LinearRegression(),
        Lasso(random_state=random_state),
        Ridge(random_state=random_state),
        SVR(),
        RandomForestRegressor(random_state=random_state),
        GradientBoostingRegressor(random_state=random_state),
        XGBRegressor(),
        DecisionTreeRegressor(random_state=random_state),
        AdaBoostRegressor(random_state=random_state),
        ExtraTreesRegressor(random_state=random_state),
    ]

    # initial values, lowest_error is infinity so that any error that a model has will be the new minimum
    lowest_error = math.inf
    best_model = None

    for i, model in enumerate(models):
        print(model)
        model.fit(train_input, train_output)

        predict = model.predict(test_input)
        error = root_mean_squared_error(test_output, predict)
        print(f"Test Error: {error}")

        # if the error is the new best
        if error < lowest_error:
            # make that the new lowest error and record the index
            lowest_error = error
            best_model = i

        print()

    print(f"The best model is {models[best_model]}")

    predict = models[best_model].predict(scaled_input)
    error = root_mean_squared_error(scaled_output, predict)
    print(f"Error on full dataset: {error}")

    dump(models[best_model], "model.bin")
    loaded_model = load("model.bin")

    gender = int(input("Enter gender (0 for female, 1 for male): "))
    age = int(input("Enter age: "))
    income = float(input("Enter income: "))
    dept = float(input("Enter credit card dept: "))
    healthcare = int(input("Enter healthcare cost: "))
    inherited = float(input("Enter amount inherited: "))
    stocks = float(input("Enter stocks: "))
    bonds = float(input("Enter Bonds: "))
    funds = float(input("Enter mutual funds: "))
    etf = float(input("Enter ETFs: "))
    reit = float(input("Enter REITs: "))

    predict = loaded_model.predict(np.array([
        gender,
        age,
        income,
        dept,
        healthcare,
        inherited,
        stocks,
        bonds,
        funds,
        etf,
        reit,
    ]).reshape(1, -1))
    print(f"Predicted net worth: {output_scalar.transform(predict)}")
