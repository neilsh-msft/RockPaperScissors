from __future__ import print_function
from matplotlib import pyplot as plt
import os, sys
import numpy as np
import time
import math
import csv
import copy

from cntk import input_variable, Trainer, squared_error, classification_error
from cntk.initializer import glorot_uniform, he_normal
from cntk.layers import Recurrence, LSTM, Dropout, Dense
from cntk.learners import learning_rate_schedule, UnitType, momentum_as_time_constant_schedule, sgd, fsadagrad
from cntk.ops import *
from cntk.ops.sequence import last as Last
from cntk.ops.sequence import input as SequenceInput

# Read in the RPS data
script_directory = os.path.dirname(sys.argv[0])
data = []
with open(os.path.join(script_directory, "rps.csv"), 'r') as csvfile:
  recordReader = csv.DictReader(csvfile, fieldnames=['HM','CM','WLD'])
  for row in recordReader:
    if row['WLD'] == "0": row['WLD'] = np.float32(1.0)
    if row['WLD'] == "1": row['WLD'] = np.float32(1.0)
    if row['WLD'] == "-1": row['WLD'] = np.float32(0.0)
    data.append(row)

# We will play by looking back at the previous n moves
# The amount of training data necessary is proportional to the number of lookback moves
# Given that any one game has 9 possible combinations and all games are independent events,
# the number of possibilites is 9^(lookback_moves) = [ 1, 9, 81, 729, 6561, 59049, 531441, .... ]
# The best possible strategy will be to increase the amount of lookback as we get more training data.
lookbackMoves = 0
trainingFactor = 3
numberOfFeatures = 7 # 7 if we add the win/loss feature, 6 otherwise
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
  return np.float32(np.r_[encodeLabel(hm), encodeLabel(cm), wld])
 
def defaultMove():
  return encodeFeature('X', 'X', np.float32(0.0))
  
def getMoveIndex(move):
  return np.argmax(move)

moveNumber = 0
previousMoves = []
previousMoves.append(defaultMove())
training_features = []
training_labels = []

for row in data:
  training_features.append(copy.deepcopy(previousMoves))
  training_labels.append(encodeLabel(row['HM']))
  moveNumber += 1
  if moveNumber % gameLength == 0:
    previousMoves = []
    previousMoves.append(defaultMove())
  else:
    if moveNumber % gameLength < lookbackMoves:
      previousMoves.append(encodeFeature(row['HM'], row['CM'], row['WLD']))
    else:
      previousMoves = np.float32(np.resize(np.roll(previousMoves, -1 * numberOfFeatures), (lookbackMoves - 1, numberOfFeatures)))
      previousMoves = np.resize(np.append(previousMoves, encodeFeature(row['HM'], row['CM'], row['WLD'])), (lookbackMoves, numberOfFeatures))

# LSTM Network
input_dim = numberOfFeatures * lookbackMoves
cell_dim = 64
hidden_dim = 128
num_output_classes = 3
minibatch_size = gameLength

# Defines the LSTM model for classifying sequences
def LSTM_sequence_classifer_net(input, output_classes):
  model = Recurrence(LSTM(input_dim, enable_self_stabilization=True))(input)
  model = Last(model) # Note this is semantically equivalent to "Fold" (take only the output of the last sequence)
  model = Dropout(0.2)(model)
  model = Dense(output_classes)(model)
  return model

# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"
    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose: 
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb + 1, training_loss, eval_error*100))
    return mb, training_loss, eval_error

inputs = SequenceInput(numberOfFeatures) #shape = input_dim, is_sparse = True)
classifier_output = LSTM_sequence_classifer_net(inputs, num_output_classes)
label = input_variable(shape = num_output_classes, is_sparse = False, dynamic_axes=classifier_output.dynamic_axes)

loss = squared_error(classifier_output, label)
label_error = classification_error(classifier_output, label)
learner = fsadagrad(classifier_output.parameters, learning_rate_schedule(0.01, UnitType.minibatch), momentum_as_time_constant_schedule(minibatch_size / -math.log(0.9)))

trainer = Trainer(classifier_output, (loss, label_error), [learner])

# Initialize the parameters for the trainer, we will train, validate and test
# in minibatches corresponding to the 20 game definition
num_minibatches = len(training_features) // minibatch_size
validate_mb = int(num_minibatches * 0.2)
test_mb = int(num_minibatches * 0.1)
train_mb = num_minibatches - validate_mb - test_mb

trainf, testf, validatef = training_features[:train_mb*gameLength], training_features[train_mb*gameLength:(train_mb + test_mb)*gameLength], training_features[validate_mb*gameLength:]
trainl, testl, validatel = training_labels[:train_mb*gameLength], training_labels[train_mb*gameLength:(train_mb + test_mb)*gameLength], training_labels[validate_mb*gameLength:]

print("Number of mini batches = train:{0} test:{1} validate:{2}".format(train_mb, test_mb, validate_mb))

# Run the trainer on and perform model training
training_progress_output_freq = 10
loss_summary = []
train_start = time.time()

for i in range(0, train_mb):
  features = trainf[i*gameLength:(i+1)*gameLength]
  labels = trainl[i*gameLength:(i+1)*gameLength]

  # Specify the mapping of input variables in the model to actual minibatch data to be trained with
  trainer.train_minibatch({inputs : features, label : labels})
  loss_summary.append(trainer.previous_minibatch_loss_average)
  batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=1)

print ("Training took {:.1f} sec".format(time.time() - train_start))
plt.plot(loss_summary, label='training loss')
plt.show()

# Testing
avg_error = 0.0
for i in range(0, test_mb):
  features = testf[i*gameLength:(i+1)*gameLength]
  labels = testl[i*gameLength:(i+1)*gameLength]
  avg_error += trainer.test_minibatch({inputs : features, label : labels})

avg_error = avg_error / test_mb
print("Mean squared testing error: {0:2.2f}%".format(avg_error * 100))

classifier_output.save(os.path.join(script_directory, "rps.model"))

# Evaluation
out = softmax(classifier_output)
num_wins = 0.0
num_games = 0.0
for i in range(0, validate_mb):
  features = validatef[i*gameLength:(i+1)*gameLength]
  labels = validatel[i*gameLength:(i+1)*gameLength]
  # Iterate through the games in the batch
  for j in range(0, gameLength):
    predicted_label_probs = np.array(out.eval({out.arguments[0]:[features[j]]})).flatten()
    if np.argmax(predicted_label_probs) == getMoveIndex(labels[j]) :
      num_wins += 1
    num_games += 1

print("Evaluation predictions: {0} / {1} ({2:2.2f}%)".format(int(num_wins), int(num_games), (num_wins/num_games) * 100))
