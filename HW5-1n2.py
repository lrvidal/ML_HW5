import random
import sklearn.model_selection as sklMS
import sklearn.linear_model as sklLM
from sklearn.metrics import mean_squared_error

input_file = "./abalone.data"

mapping = {
    'M': [1,0,0],
    'F': [0,1,0],
    'I': [0,0,1]
}

with open(input_file, 'r') as file:
    lines = file.readlines()

processed_lines = []
for i, line in enumerate(lines):
    data = line.strip().split(',')
    gen = data[0]
    data.pop(0)
    if gen in mapping:
        data.insert(0, mapping[gen][0])
        data.insert(0, mapping[gen][1])
        data.insert(0, mapping[gen][2])

    floats = [float(item) for item in data[:-1]]
    floats.append(int(data[-1]))
    processed_lines.append(floats)

test_data, rest = sklMS.train_test_split(processed_lines, test_size=0.2, train_size=0.8)
validate_data, train_data = sklMS.train_test_split(rest, test_size=0.25, train_size=0.75)
lambdaOne = [0, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1]
lambdaTwo = [0, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1]

models = []
for lda1 in lambdaOne:
    for lda2 in lambdaTwo:
        alpha = lda1 + lda2
        l1_ratio = lda1 / (lda1+lda2) if lda1+lda2 != 0 else 0
        m = sklLM.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        m.fit(train_data, [item[-1] for item in train_data])
        models.append(m)


def MSE(model, dataset):
    y_pred = []
    y_true = []
    for item in dataset:
        y_pred.append(model.predict([item])[0])
        y_true.append(item[-1])
    return mean_squared_error(y_true=y_true, y_pred=y_pred)

m1 = models.copy()
m2 = models.copy()

trainSpecs = []
validateSpecs = []

print("Training Data:")
for lda1 in lambdaOne:
    for lda2 in lambdaTwo:
        m = m1.pop(0)
        mse = MSE(m, train_data)
        trainSpecs.append([lda1, lda2, mse])
        print("Lambda1: {}, Lambda2: {}, MSE: {}".format(lda1, lda2, mse))

print("Validating Data:")
for lda1 in lambdaOne:
    for lda2 in lambdaTwo:
        m = m2.pop(0)
        mse = MSE(m, validate_data)
        validateSpecs.append([lda1, lda2, mse])
        print("Lambda1: {}, Lambda2: {}, MSE: {}".format(lda1, lda2, mse))

print("Best Training Model:")
bestTrain = min(trainSpecs, key=lambda x: x[2])
print("Lambda1: {}, Lambda2: {}, MSE: {}".format(bestTrain[0], bestTrain[1], bestTrain[2]))

print("Best Validation Model:")
bestTest = min(validateSpecs, key=lambda x: x[2])
print("Lambda1: {}, Lambda2: {}, MSE: {}".format(bestTest[0], bestTest[1], bestTest[2]))

alpha = bestTest[0] + bestTest[1]
l1_ratio = bestTest[0] / (bestTest[0]+bestTest[1])
bestModel = sklLM.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
combinedData = train_data + validate_data
bestModel.fit(combinedData, [item[-1] for item in combinedData])
bestModelSpec = []

mseOfBestModel = MSE(bestModel, test_data)
print("MSE of model trained with optimal lambda values: {}".format(mseOfBestModel))