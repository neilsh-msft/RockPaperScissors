from __future__ import print_function
from matplotlib import pyplot as plt
import os, sys
import numpy as np
import time
import math
import csv

from cntk import input_variable, Trainer
from cntk.initializer import glorot_uniform, he_normal
from cntk.layers import Recurrence, LSTM, Dropout, Dense
from cntk.learner import learning_rate_schedule, UnitType, momentum_as_time_constant_schedule, sgd, adam_sgd
from cntk.ops import *
from cntk.ops.sequence import last as Last
from cntk.utils import get_train_eval_criterion, get_train_loss

# Read in the RPS data
data = []
with open(os.path.join(script_directory, "rps.csv"), 'r') as csvfile:
  recordReader = csv.DictReader(csvfile, fieldnames=['HM','CM','WLD'])
  for row in recordReader:
    if row['WLD'] == "0": row['WLD'] = np.float32(0.0)
    if row['WLD'] == "1": row['WLD'] = np.float32(1.0)
    if row['WLD'] == "-1": row['WLD'] = np.float32(-1.0)
    data.append(row)

# We will play by looking back at the previous n moves
# The amount of training data necessary is proportional to the number of lookback moves
# Given that any one game has 9 possible combinations and all games are independent events,
# the number of possibilites is 9^(lookback_moves) = [ 1, 9, 81, 729, 6561, 59049, 531441, .... ]
# The best possible strategy will be to increase the amount of lookback as we get more training data.
lookbackMoves = 0
trainingFactor = 3
gameLength = 20
while len(data) > math.pow(9, lookbackMoves + 1) * trainingFactor and lookbackMoves < gameLength:
  lookbackMoves += 1
print("Number of lookback moves {0}".format(lookbackMoves))

# Transform the data into a training set
# The CSV file contains games in 20 packs, so we should build them in sequences of 20 where the first
# few pad the previous games with 0 data [CM, CM-1, CM-2, CM-3, CM-4]
def encodeLabel(move):
  rock = np.float32(1.0) if move == 'R' else np.float32(0.0)
  paper = np.float32(1.0) if move == 'P' else np.float32(0.0)
  scissors = np.float32(1.0) if move == 'S' else np.float32(0.0)
  return [rock, paper, scissors]
  
def encodeFeature(hm, cm, wld):
  return np.r_[encodeLabel(hm), encodeLabel(cm), wld].tolist()
  
def defaultMove():
  return [0, 0, 0]
  
def getMoveIndex(move):
  return np.argmax(move)
  
moveNumber = 0
previousMoves = np.array([encodeFeature('X', 'X', 0) for x in range(lookbackMoves)]).flatten().tolist()
training_features = []
training_labels = []
for row in data:
  training_features.append(previousMoves)
  training_labels.append(encodeLabel(row['HM']))
  moveNumber += 1
  if moveNumber % 20 == 0:
    moveNumber = 0
    previousMoves = np.array([encodeFeature('X', 'X', 0) for x in range(lookbackMoves)]).flatten().tolist()
  else:
    previousMoves = np.append(np.resize(np.roll(previousMoves, -7), (1, 7 * (lookbackMoves - 1))), encodeFeature(row['HM'], row['CM'], row['WLD'])).tolist()

training_features = np.array(training_features, dtype="float32")
training_labels = np.array(training_labels, dtype="float32")

# LSTM Network
input_dim = 7 * lookbackMoves
cell_dim = 64
hidden_dim = 128
num_output_classes = 3
minibatch_size = gameLength

# Defines the LSTM model for classifying sequences
def LSTM_sequence_classifer_net(input, output_classes):
  model = Recurrence(LSTM(lookbackMoves))(input)
  model = Last(model)
  model = Dropout(0.2)(model)
  model = Dense(output_classes)(model)
  return model

# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"
    if mb%frequency == 0:
        training_loss = get_train_loss(trainer)
        eval_error = get_train_eval_criterion(trainer)
        if verbose: 
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb + 1, training_loss, eval_error*100))
    return mb, training_loss, eval_error

inputs = input_variable(shape = input_dim, is_sparse = False)
label = input_variable(num_output_classes, dynamic_axes=[Axis.default_batch_axis()])

classifier_output = LSTM_sequence_classifer_net(inputs, num_output_classes)

loss = squared_error(classifier_output, label)
label_error = squared_error(classifier_output, label)
learner = adam_sgd(classifier_output.parameters, learning_rate_schedule(0.01, UnitType.minibatch), momentum_as_time_constant_schedule(minibatch_size / -math.log(0.9)))

trainer = Trainer(classifier_output, (loss, label_error), [learner])

# Initialize the parameters for the trainer, we will train, validate and test
# in minibatches corresponding to the 20 game definition
minibatch_size = gameLength
num_minibatches = len(training_features) // minibatch_size
validate_mb = int(num_minibatches * 0.1)
test_mb = int(num_minibatches * 0.1)
train_mb = num_minibatches - validate_mb - test_mb

#trainf, validatef, testf = training_features[:train_mb], training_features[train_mb:validate_mb], training_features[test_mb:]
#trainl, validatel, testl = training_labels[:train_mb], training_labels[train_mb:validate_mb], training_labels[test_mb:]
tf = np.split(training_features, num_minibatches)
tl = np.split(training_labels, num_minibatches)

print("Number of mini batches = train:{0} validate:{1} test:{2}".format(train_mb, validate_mb, test_mb))

# Run the trainer on and perform model training
training_progress_output_freq = 10
loss_summary = []
train_start = time.time()

for i in range(0, train_mb):
  #features = trainf[i*gameLength:(i+1)*gameLength]
  #labels = trainl[i*gameLength:(i+1)*gameLength]
  features = np.ascontiguousarray(tf[i%num_minibatches])
  labels = np.ascontiguousarray(tl[i%num_minibatches])

  # Specify the mapping of input variables in the model to actual minibatch data to be trained with
  trainer.train_minibatch({inputs : features, label : labels})
  loss_summary.append(trainer.previous_minibatch_loss_average)
  batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=1)

print ("Training took {:.1f} sec".format(time.time() - train_start))
plt.plot(loss_summary, label='training loss')
plt.show()

# Testing
avg_error = 0.0
for i in range(train_mb, train_mb + test_mb):
  features = np.ascontiguousarray(tf[i%num_minibatches])
  labels = np.ascontiguousarray(tl[i%num_minibatches])
  avg_error += trainer.test_minibatch({inputs : features, label : labels})

avg_error = avg_error / test_mb
print("Mean squared testing error: {0:2.2f}%".format(avg_error * 100))

classifier_output.save(os.path.join(script_directory, "rps.model"))

# Evaluation
out = softmax(classifier_output)
num_wins = 0.0
num_games = 0.0
for i in range(train_mb + test_mb, num_minibatches):
  features = np.ascontiguousarray(tf[i%num_minibatches])
  labels = np.ascontiguousarray(tl[i%num_minibatches])
  # Iterate through the games in the batch
  for j in range(0, gameLength):
    predicted_label_probs = np.array(out.eval({out.arguments[0]:[features[j]]})).flatten()
    if np.argmax(predicted_label_probs) == getMoveIndex(labels[j]) :
      num_wins += 1
    num_games += 1

print("Evaluation predictions: {0} / {1} ({2:2.2f}%)".format(int(num_wins), int(num_games), (num_wins/num_games) * 100))from __future__ import print_function
from matplotlib import pyplot as plt
import os, sys
import numpy as np
import time
import math
import csv

from cntk import input_variable, Trainer
from cntk.initializer import glorot_uniform, he_normal
from cntk.layers import Recurrence, LSTM, Dropout, Dense
from cntk.learner import learning_rate_schedule, UnitType, momentum_as_time_constant_schedule, sgd, adam_sgd
from cntk.ops import *
from cntk.ops.sequence import last as Last
from cntk.utils import get_train_eval_criterion, get_train_loss

# Read in the RPS data
data = []
with open(os.path.join(script_directory, "rps.csv"), 'r') as csvfile:
  recordReader = csv.DictReader(csvfile, fieldnames=['HM','CM','WLD'])
  for row in recordReader:
    if row['WLD'] == "0": row['WLD'] = np.float32(0.0)
    if row['WLD'] == "1": row['WLD'] = np.float32(1.0)
    if row['WLD'] == "-1": row['WLD'] = np.float32(-1.0)
    data.append(row)

# We will play by looking back at the previous n moves
# The amount of training data necessary is proportional to the number of lookback moves
# Given that any one game has 9 possible combinations and all games are independent events,
# the number of possibilites is 9^(lookback_moves) = [ 1, 9, 81, 729, 6561, 59049, 531441, .... ]
# The best possible strategy will be to increase the amount of lookback as we get more training data.
lookbackMoves = 0
trainingFactor = 3
gameLength = 20
while len(data) > math.pow(9, lookbackMoves + 1) * trainingFactor and lookbackMoves < gameLength:
  lookbackMoves += 1
print("Number of lookback moves {0}".format(lookbackMoves))

# Transform the data into a training set
# The CSV file contains games in 20 packs, so we should build them in sequences of 20 where the first
# few pad the previous games with 0 data [CM, CM-1, CM-2, CM-3, CM-4]
def encodeLabel(move):
  rock = np.float32(1.0) if move == 'R' else np.float32(0.0)
  paper = np.float32(1.0) if move == 'P' else np.float32(0.0)
  scissors = np.float32(1.0) if move == 'S' else np.float32(0.0)
  return [rock, paper, scissors]
  
def encodeFeature(hm, cm, wld):
  return np.r_[encodeLabel(hm), encodeLabel(cm), wld].tolist()
  
def defaultMove():
  return [0, 0, 0]
  
def getMoveIndex(move):
  return np.argmax(move)
  
moveNumber = 0
previousMoves = np.array([encodeFeature('X', 'X', 0) for x in range(lookbackMoves)]).flatten().tolist()
training_features = []
training_labels = []
for row in data:
  training_features.append(previousMoves)
  training_labels.append(encodeLabel(row['HM']))
  moveNumber += 1
  if moveNumber % 20 == 0:
    moveNumber = 0
    previousMoves = np.array([encodeFeature('X', 'X', 0) for x in range(lookbackMoves)]).flatten().tolist()
  else:
    previousMoves = np.append(np.resize(np.roll(previousMoves, -7), (1, 7 * (lookbackMoves - 1))), encodeFeature(row['HM'], row['CM'], row['WLD'])).tolist()

training_features = np.array(training_features, dtype="float32")
training_labels = np.array(training_labels, dtype="float32")

# LSTM Network
input_dim = 7 * lookbackMoves
cell_dim = 64
hidden_dim = 128
num_output_classes = 3
minibatch_size = gameLength

# Defines the LSTM model for classifying sequences
def LSTM_sequence_classifer_net(input, output_classes):
  model = Recurrence(LSTM(lookbackMoves))(input)
  model = Last(model)
  model = Dropout(0.2)(model)
  model = Dense(output_classes)(model)
  return model

# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"
    if mb%frequency == 0:
        training_loss = get_train_loss(trainer)
        eval_error = get_train_eval_criterion(trainer)
        if verbose: 
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb + 1, training_loss, eval_error*100))
    return mb, training_loss, eval_error

inputs = input_variable(shape = input_dim, is_sparse = False)
label = input_variable(num_output_classes, dynamic_axes=[Axis.default_batch_axis()])

classifier_output = LSTM_sequence_classifer_net(inputs, num_output_classes)

loss = squared_error(classifier_output, label)
label_error = squared_error(classifier_output, label)
learner = adam_sgd(classifier_output.parameters, learning_rate_schedule(0.01, UnitType.minibatch), momentum_as_time_constant_schedule(minibatch_size / -math.log(0.9)))

trainer = Trainer(classifier_output, (loss, label_error), [learner])

# Initialize the parameters for the trainer, we will train, validate and test
# in minibatches corresponding to the 20 game definition
minibatch_size = gameLength
num_minibatches = len(training_features) // minibatch_size
validate_mb = int(num_minibatches * 0.1)
test_mb = int(num_minibatches * 0.1)
train_mb = num_minibatches - validate_mb - test_mb

#trainf, validatef, testf = training_features[:train_mb], training_features[train_mb:validate_mb], training_features[test_mb:]
#trainl, validatel, testl = training_labels[:train_mb], training_labels[train_mb:validate_mb], training_labels[test_mb:]
tf = np.split(training_features, num_minibatches)
tl = np.split(training_labels, num_minibatches)

print("Number of mini batches = train:{0} validate:{1} test:{2}".format(train_mb, validate_mb, test_mb))

# Run the trainer on and perform model training
training_progress_output_freq = 10
loss_summary = []
train_start = time.time()

for i in range(0, train_mb):
  #features = trainf[i*gameLength:(i+1)*gameLength]
  #labels = trainl[i*gameLength:(i+1)*gameLength]
  features = np.ascontiguousarray(tf[i%num_minibatches])
  labels = np.ascontiguousarray(tl[i%num_minibatches])

  # Specify the mapping of input variables in the model to actual minibatch data to be trained with
  trainer.train_minibatch({inputs : features, label : labels})
  loss_summary.append(trainer.previous_minibatch_loss_average)
  batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=1)

print ("Training took {:.1f} sec".format(time.time() - train_start))
plt.plot(loss_summary, label='training loss')
plt.show()

# Testing
avg_error = 0.0
for i in range(train_mb, train_mb + test_mb):
  features = np.ascontiguousarray(tf[i%num_minibatches])
  labels = np.ascontiguousarray(tl[i%num_minibatches])
  avg_error += trainer.test_minibatch({inputs : features, label : labels})

avg_error = avg_error / test_mb
print("Mean squared testing error: {0:2.2f}%".format(avg_error * 100))

classifier_output.save(os.path.join(script_directory, "rps.model"))

# Evaluation
out = softmax(classifier_output)
num_wins = 0.0
num_games = 0.0
for i in range(train_mb + test_mb, num_minibatches):
  features = np.ascontiguousarray(tf[i%num_minibatches])
  labels = np.ascontiguousarray(tl[i%num_minibatches])
  # Iterate through the games in the batch
  for j in range(0, gameLength):
    predicted_label_probs = np.array(out.eval({out.arguments[0]:[features[j]]})).flatten()
    if np.argmax(predicted_label_probs) == getMoveIndex(labels[j]) :
      num_wins += 1
    num_games += 1

print("Evaluation predictions: {0} / {1} ({2:2.2f}%)".format(int(num_wins), int(num_games), (num_wins/num_games) * 100))