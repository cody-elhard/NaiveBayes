import pandas
import numpy

debug = False

trainingFile = "irisTraining.txt"
train_data = pandas.read_csv(trainingFile, sep=" ", header=None)

testingFile = "irisTesting.txt"
test_data = pandas.read_csv(testingFile, sep=" ", header=None)

# Assign the columns arbritrary labels based on the length
columns_array = []
total_test_columns = test_data.shape[1]
for i in range(total_test_columns - 1):
  columns_array.append(i)
# Set the last item in the row to be the label
columns_array.append("label")

# Set the columns on the pandas sets
train_data.columns = columns_array
test_data.columns = columns_array

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

def normal_pdf(x, m, s):
  return (1/(2 * numpy.pi * s**2)**0.5) * numpy.exp(-1 * (x-m)**2 / (2 * s**2))

test_records = test_data.shape[0]
for i in range(test_records):
  test_row = test_data.iloc[i]
  # Set these to 1, which prevents from division by zero
  yes_prob = 1
  no_prob = 1

  for attribute, value in test_row.iteritems():
    # Training data where custom attribute is equal to value, given its a yes
    numpy_array_yes = numpy.array(train_data[train_data['label'] == 1][attribute])
    yes_mean = numpy.average(numpy_array_yes)
    yes_standard_deviation = numpy.std(numpy_array_yes)

    # Training data where custom attribute is equal to value, given its a no
    numpy_array_no = numpy.array(train_data[train_data['label'] == -1][attribute])
    no_mean = numpy.average(numpy_array_no)
    no_standard_deviation = numpy.std(numpy_array_no)

    yes_prob *= normal_pdf(value, yes_mean, yes_standard_deviation)
    no_prob *= normal_pdf(value, no_mean, no_standard_deviation)

  yes_prob = yes_prob * yes_count
  no_prob = no_prob * no_count

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
