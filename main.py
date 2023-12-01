from utils import split_into_samples
from csv_reader import read_csv

iris_data = read_csv("./Iris.csv", True, ignore_column=[0])

train_percentage = 0.8
train_size = int(len(iris_data) * train_percentage)
test_size = len(iris_data) - train_size

X_train, Y_train = split_into_samples(iris_data, 0, train_size, features=4, label=4, randomize=True)
X_test, Y_test = split_into_samples(iris_data, train_size, len(iris_data), features=4, label=4, randomize=True)

