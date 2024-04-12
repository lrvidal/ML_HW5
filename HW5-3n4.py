import sklearn.model_selection as sklMS
from sklearn.linear_model import LogisticRegression

def zeroOneLoss(model, dataset):
    data = [[0, 0, 0, 0], 0.0]
    for item in dataset:
        if model.predict([item])[0] == item[-1]:
            if item[-1] == 0:
                data[0][0] += 1
            else:
                data[0][1] += 1
            data[1] += 0
        else:
            if item[-1] == 1:
                data[0][2] += 1
            else:
                data[0][3] += 1
            data[1] += 1
    data[1] /= len(dataset)
    return data

def printConfusionMatrix(data):
    print("       Negative   |   Positive")
    print("--------------------------------")
    print("True:   {}      |     {}".format(data[0], data[1]))
    print("False:  {}        |     {}".format(data[2], data[3]))

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

l2data01, l2reg01 = zeroOneLoss(l2regModel, validate_data)
nonedata01, nonereg01 = zeroOneLoss(noneregModel, validate_data)

print("L2 Regularization 0-1 Loss: {}".format(l2reg01))
printConfusionMatrix(l2data01)
print("No Regularization 0-1 Loss: {}".format(nonereg01))
printConfusionMatrix(nonedata01)

bestModel = LogisticRegression(penalty=None)
bestModel.fit(train_data + validate_data, [item[-1] for item in (train_data + validate_data)])

testData01, testReg01 = zeroOneLoss(bestModel, test_data)
print("Test Data 0-1 Loss: {}".format(testReg01))
printConfusionMatrix(testData01)