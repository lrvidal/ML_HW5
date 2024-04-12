import sklearn.model_selection as sklMS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import zero_one_loss
from sklearn.metrics import confusion_matrix

def zeroOneLoss(model, dataset):
    y_pred = []
    y_true = []
    for item in dataset:
        y_pred.append(model.predict([item])[0])
        y_true.append(item[-1])
    return zero_one_loss(y_true=y_true, y_pred=y_pred), confusion_matrix(y_true=y_true, y_pred=y_pred)

def printConfusionMatrix(data):
    print("       Negative   |   Positive")
    print("--------------------------------")
    print("True:   {}      |     {}".format(data[0][0], data[1][1]))
    print("False:  {}        |     {}".format(data[1][0], data[0][1]))

input_file = "./cammeo_osmancik.data"
mapping = {
    "Cammeo": 0,
    "Osmancik": 1
}

with open(input_file, 'r') as file:
    lines = file.readlines()

processed_lines = []
for _, line in enumerate(lines):
    data = line.strip().split(',')
    typ = data[-1]
    data.pop(-1)
    if typ in mapping:
        data.append(mapping[typ])
    
    floats = [float(item) for item in data[:-1]]
    floats.append(int(data[-1]))
    processed_lines.append(floats)

#Split data into test and rest
test_data, rest = sklMS.train_test_split(processed_lines, test_size=0.2, train_size=0.8)
#Split rest into validate and train
validate_data, train_data = sklMS.train_test_split(rest, test_size=0.25, train_size=0.75)

l2regModel = LogisticRegression(penalty='l2')
noneregModel = LogisticRegression(penalty=None)

l2regModel.fit(train_data, [item[-1] for item in train_data])
noneregModel.fit(train_data, [item[-1] for item in train_data])

l2regPredicts = []
noneregPredicts = []

for data in validate_data:
    l2regPredicts.append(l2regModel.predict([data])[0])
    noneregPredicts.append(noneregModel.predict([data])[0])

l2ZeroOne, l2Confusion = zeroOneLoss(l2regModel, validate_data)
noneZeroOne, noneConfusion = zeroOneLoss(noneregModel, validate_data)

print("L2 Regularization 0-1 Loss: {}".format(l2ZeroOne))
printConfusionMatrix(l2Confusion)
print("No Regularization 0-1 Loss: {}".format(noneZeroOne))
printConfusionMatrix(noneConfusion)

bestModel = LogisticRegression(penalty=None)
bestModel.fit(train_data + validate_data, [item[-1] for item in (train_data + validate_data)])

testZeroOne, testConfusion = zeroOneLoss(bestModel, test_data)
print("Test Data 0-1 Loss: {}".format(testZeroOne))
printConfusionMatrix(testConfusion)