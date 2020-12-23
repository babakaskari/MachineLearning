import prepossessed_dataset


dataset = prepossessed_dataset.semi_super_no_date()
# x_unlabeled = dataset["x_unlabeled"]
x_train = dataset["x_train"]
y_train = dataset["y_train"]
x_test = dataset["x_test"]
y_test = dataset["y_test"]
x_unlabeled = dataset["x_unlabeled"]
print("x_train : ", x_train)
print("y_train : ", y_train)
print("x_test : ", x_test)
print("y_test : ", y_test)
print("x_train shape : ", x_train.shape)
print("y_train shape : ", y_train.shape)
print("x_test shape  : ", x_test.shape)
print("y_test shape : ", y_test.shape)
print("x_unlabeled shape : ", x_unlabeled.shape)

