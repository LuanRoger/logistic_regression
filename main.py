from logistic_regression import LogisticRegression
from regression_utils import calculate_metrics
from utils import IrisClasification, split_into_samples, convert_into_bynary_output
from csv_reader import read_csv

iris_data = read_csv("./Iris.csv", True, ignore_column=[0])

train_percentage = 0.8
train_size = int(len(iris_data) * train_percentage)
test_size = len(iris_data) - train_size

X_train, Y_train = split_into_samples(iris_data, 0, train_size, features=4, label=4, randomize=True)
X_test, Y_test = split_into_samples(iris_data, train_size, len(iris_data), features=4, label=4, randomize=True)

setosa_clasification = LogisticRegression()
setosa_y_train = convert_into_bynary_output(Y_train, IrisClasification.SETOSA)
versicolor_clasification = LogisticRegression()
versicolor_y_train = convert_into_bynary_output(Y_train, IrisClasification.VERSICOLOR)
virginica_clasification = LogisticRegression()
virginica_y_train = convert_into_bynary_output(Y_train, IrisClasification.VIRGINICA)

setosa_clasification.train(X_train, setosa_y_train)
versicolor_clasification.train(X_train, versicolor_y_train)
virginica_clasification.train(X_train, virginica_y_train)

predictions_setosa: list[bool] = []
predictions_virginica: list[bool] = []
predictions_versicolor: list[bool] = []
for i in range(len(X_test)):
    sample_x = X_test[i]
    sample_y = Y_test[i]

    _, setosa_prediction = setosa_clasification.predict(sample_x)
    predictions_setosa.append(setosa_prediction)
for i in range(len(X_test)):
    sample_x = X_test[i]
    sample_y = Y_test[i]

    _, virginica_prediction = virginica_clasification.predict(sample_x)
    predictions_virginica.append(virginica_prediction)

for i in range(len(X_test)):
    sample_x = X_test[i]
    sample_y = Y_test[i]

    _, versicolor_prediction = versicolor_clasification.predict(sample_x)
    predictions_versicolor.append(versicolor_prediction)

setosa_y_test = convert_into_bynary_output(Y_test, IrisClasification.SETOSA)
versicolor_y_test = convert_into_bynary_output(Y_test, IrisClasification.VERSICOLOR)
virginica_y_test = convert_into_bynary_output(Y_test, IrisClasification.VIRGINICA)

setosa_accuracy, setosa_precision, setosa_recall = calculate_metrics(predictions_setosa, setosa_y_test)
versicolor_accuracy, versicolor_precision, versicolor_recall = calculate_metrics(predictions_versicolor, versicolor_y_test)
virginica_accuracy, virginica_precision, virginica_recall = calculate_metrics(predictions_virginica, virginica_y_test)

print("Setosa metrics:")
print(f"Setosa accuracy: {setosa_accuracy*100:.2f}%")
print(f"Setosa precision: {setosa_precision*100:.2f}%")
print(f"Setosa recall: {setosa_recall*100:.2f}%")

print("Versicolor metrics:")
print(f"Versicolor accuracy: {versicolor_accuracy*100:.2f}%")
print(f"Versicolor precision: {versicolor_precision*100:.2f}%")
print(f"Versicolor recall: {versicolor_recall*100:.2f}%")

print("Virginica metrics:")
print(f"Virginica accuracy: {virginica_accuracy*100:.2f}%")
print(f"Virginica precision: {virginica_precision*100:.2f}%")
print(f"Virginica recall: {virginica_recall*100:.2f}%")

for i in range(len(X_test)):
    sample_x = X_test[i]
    sample_y = Y_test[i]
    sample_iris_clasification: IrisClasification = IrisClasification(sample_y)
    _, setosa_prediction = setosa_clasification.predict(sample_x)
    _, virginica_prediction = virginica_clasification.predict(sample_x)
    _, versicolor_prediction = versicolor_clasification.predict(sample_x)

    if(setosa_prediction):
        print(f"Prediction {i}: Setosa\tActual: {sample_iris_clasification.name}")
    elif(virginica_prediction):
        print(f"Prediction {i}: Virginica\tActual: {sample_iris_clasification.name}")
    elif(versicolor_prediction):
        print(f"Prediction {i}: Versicolor\tActual: {sample_iris_clasification.name}")
    else:
        print(f"Prediction {i}: Unknown\tActual: {sample_iris_clasification.name}")