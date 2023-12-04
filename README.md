# Logistic Regression on Iris dataset
Implementation of a Logistic Regression model to classify flowers from the Iris dataset, made as a work for the AI class.

## Dataset
The used dataset was the Iris dataset, which contains 150 flower samples, with 4 attributes each, and 3 different flower classes. The dataset can be found [here](https://github.com/LuanRoger/logistic_regression/blob/main/Iris.csv)

## Implementation
The algorithm (on `LogisticRegression` class) takes as parameters the number of iterations and the learning rate, after you can call the `train` method to train the model and the `predict` method to predict the class of a given sample.

The program also contains a `main` function that reads the dataset, splits it into random train and test sets, 80% of the set is for train and the 30% is for test, so after, it trains each individual model for each class using 100000 interations to do a binary classification (one vs all). At the end, the program prints the accuracy, precision and recall of the model on the test set using the function `calculate_metrics` on the `regression_utils.py` file.

## Execution
To execute the program, you can run the `main.py` file, it will print the metrics for each model.
```bash
python main.py
```
> The script does not have any external dependencies.

## Result
The algorithm has been run 3 times to get diferents samples from the random train/test split and the results was being put on a spreadsheet that can be accessed here: [Results](https://docs.google.com/spreadsheets/d/1bsN92PP5TCqqemPfw0nXc56DgCIDMK41AjimR5w_1Fw/edit?usp=sharing)