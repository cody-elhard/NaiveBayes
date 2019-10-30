import pandas
import numpy
import math

train_data = pandas.read_csv("buyTraining.txt", sep=" ", header=None)
train_data.columns = ["column1", "column2", "column3", "column4", "label"]

test_data = pandas.read_csv("buyTesting.txt", sep=" ", header=None)
test_data.columns = ["column1", "column2", "column3", "column4", "label"]

total_training_rows = train_data.shape[0]
total_test_rows = test_data.shape[0]
# Yes Count = train data where label == 1
# Where label is customly defined in the line by test_data.columns = [...columns, "label"]
yes_count = train_data[train_data['label'] == 1].shape[0]
# No Count = train data where label == -1
no_count = train_data[train_data['label'] == -1].shape[0]

# Remove the class label from the test data, so that is not used to adjust the model as a custom attribute
class_labels = test_data['label']
del test_data['label']

# Positively identified counts
true_positive_count = 0
true_negative_count = 0
# Falsely identified counts
false_positive_count = 0 
false_negative_count = 0

test_records = test_data.shape[0]
for i in range(test_records):
  test_row = test_data.iloc[i]

  # Set this to 1, which prevents from division by zero
  yes_prob = 0
  no_prob = 0

  for attribute, value in test_row.iteritems():
    # Training data where custom attribute is equal to value, given its a yes
    yes_count = train_data[(train_data[attribute] == value) & (train_data['label'] == 1)].shape[0]
    # Training data where custom attribute is equal to value, given its a no
    no_count = train_data[(train_data[attribute] == value) & (train_data['label'] == -1)].shape[0]

    yes_prob = yes_count / total_training_rows
    no_prob = no_count / total_training_rows

  predicted_class = 1 if yes_prob > no_prob else -1
  actual_class = class_labels.values[i]

  if (predicted_class == 1):
    if (actual_class == 1):
      true_positive_count += 1
    else:
      false_positive_count += 1
  elif predicted_class == -1:
    if (actual_class == -1):
      true_negative_count += 1
    else:
      false_negative_count += 1

print('--- ---')
print('counts')
# Positively identified counts
print('true_positive_count')
print(true_positive_count)
print('true_negative_count')
print(true_negative_count)
# Falsely identified counts
print('false_positive_count')
print(false_positive_count)
print('false_negative_count')
print(false_negative_count)
# Calculated metrics below
accuracy = (true_positive_count + true_negative_count) / total_test_rows
print('accuracy: ', accuracy)
# portion of testing data positives that were correcly identified
sensitivity = true_positive_count / (true_positive_count + false_negative_count)
print('sensitivity: ', sensitivity)
# portion of testing data negatives that were correctly identifified
specificity = true_negative_count / (false_positive_count + true_negative_count)
print('specificity: ', specificity)
precision = true_positive_count / (true_positive_count + false_positive_count)
print('precision: ', precision)
