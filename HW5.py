import random
import sklearn.model_selection as sklMS
import sklearn.linear_model as sklLM

input_file = "./abalone.data"

mapping = {
    'M': '1,0,0',
    'F': '0,1,0',
    'I': '0,0,1'
}

with open(input_file, 'r') as file:
    lines = file.readlines()

num_entries = len(lines)
num_entries_to_remove = int(num_entries * 0.2)

indices_to_remove = random.sample(range(num_entries), num_entries_to_remove)

processed_lines = []
for i, line in enumerate(lines):
    if i not in indices_to_remove:
        data = line.strip().split(',')

        if data[0] in mapping:
            data[0] = mapping[data[0]]

        processed_line = ','.join(data)
        processed_lines.append(processed_line)

train_data, test_data = sklMS.train_test_split(processed_lines, test_size=0.25, train_size=0.75)

