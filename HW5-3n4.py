import sklearn.model_selection as sklMS
from sklearn.linear_model import LogisticRegression

def MSE(model, dataset):
    return sum([(model.predict([item])[0] - item[-1])**2 for item in dataset]) / len(dataset)


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

l2regMSE = MSE(l2regModel, validate_data)
noneregMSE = MSE(noneregModel, validate_data)

print("L2 Regularization MSE: {}".format(l2regMSE))
print("No Regularization MSE: {}".format(noneregMSE))