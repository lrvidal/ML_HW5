import random

input_file = "./abalone.data"
training_file = "./abaloneTraining.data"
validation_file = "./abaloneValidation.data"

# Define a dictionary to map the values in the first column
mapping = {
    'M': '1,0,0',
    'F': '0,1,0',
    'I': '0,0,1'
}

# Read the input file and process the data
with open(input_file, 'r') as file:
    lines = file.readlines()

# Calculate the number of entries to remove
num_entries = len(lines)
num_entries_to_remove = int(num_entries * 0.2)

# Randomly select the indices of entries to remove
indices_to_remove = random.sample(range(num_entries), num_entries_to_remove)

processed_lines = []
for i, line in enumerate(lines):
    if i not in indices_to_remove:
        # Split the line by comma
        data = line.strip().split(',')

        # Swap the value in the first column using the mapping dictionary
        if data[0] in mapping:
            data[0] = mapping[data[0]]

        # Join the modified data back into a line
        processed_line = ','.join(data)
        processed_lines.append(processed_line)

# Calculate the number of entries for training and validation
num_remaining_entries = len(processed_lines)
num_training_entries = int(num_remaining_entries * 0.75)
num_validation_entries = num_remaining_entries - num_training_entries

# Randomly select the indices of entries for training and validation
indices_for_training = random.sample(range(num_remaining_entries), num_training_entries)
indices_for_validation = [i for i in range(num_remaining_entries) if i not in indices_for_training]

training_lines = [processed_lines[i] for i in indices_for_training]
validation_lines = [processed_lines[i] for i in indices_for_validation]

# Write the training data to the training file
with open(training_file, 'w') as file:
    file.write('\n'.join(training_lines))

# Write the validation data to the validation file
with open(validation_file, 'w') as file:
    file.write('\n'.join(validation_lines))