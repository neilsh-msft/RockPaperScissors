from __future__ import print_function
import os, sys
import numpy as np
import cntk
import cntk.ops as COps

from cntk.initializer import glorot_uniform
from cntk.layers import default_options, Input, Dense                    # Layers
from cntk.learner import sgd, learning_rate_schedule, UnitType
from cntk.utils import get_train_eval_criterion, get_train_loss

script_directory = os.path.dirname(sys.argv[0])

# We will play by looking back at the previous n moves
lookbackMoves = 5
gameLength = 20

# Read in the RPS data
import csv

data = []
with open(os.path.join(script_directory, "rps.csv"), 'r') as csvfile:
  recordReader = csv.DictReader(csvfile, fieldnames=['HM','CM','WLD'])
  for row in recordReader:
    if row['WLD'] == "0": row['WLD'] = 1
    if row['WLD'] == "1": row['WLD'] = 1
    if row['WLD'] == "-1": row['WLD'] = 0
    data.append(row)

# Transform the data into a training set
# The CSV file contains games in 20 packs, so we should build them in sequences of 20 where the first
# few pad the previous games with 0 data [CM, CM-1, CM-2, CM-3, CM-4]
def encodeLabel(move):
  rock = 1 if move == 'R' else 0
  paper = 1 if move == 'P' else 0
  scissors = 1 if move == 'S' else 0
  return [rock, paper, scissors]
  
def encodeFeature(hm, cm, wld):
  return np.r_[encodeLabel(hm), encodeLabel(cm), wld].tolist()
  
def defaultMove():
  return [0, 0, 0]
 
# Features - [n previous games]
# A) Previous Human Move (as R P S separately encoded 0 or 1)
# B) Previous Computer Move
# C) Result(Win/Draw = 1, Lose = 0)
# Labels - Next Human Move
moveNumber = 0
previousMoves = np.array([encodeFeature('X', 'X', 1) for x in range(lookbackMoves)]).flatten().tolist()
training_features = []
training_labels = []
for row in data:
  training_features.append(previousMoves)
  training_labels.append(encodeLabel(row['HM']))
  moveNumber += 1
  if moveNumber % 20 == 0:
    moveNumber = 0
    previousMoves = np.array([encodeFeature('X', 'X', 1) for x in range(lookbackMoves)]).flatten().tolist()
  else:
    previousMoves = np.insert(np.resize(previousMoves, (1, 7 * (lookbackMoves - 1))), 0, encodeFeature(row['HM'], row['CM'], row['WLD'])).tolist()

training_features = np.array(training_features, dtype="float32")
training_labels = np.array(training_labels, dtype="float32")
print(training_features.shape)
print(training_labels.shape)

# Network
inputs = 7 * lookbackMoves
outputs = 3 # Encode each move selection separately
hiddenLayers = 10

input = Input(inputs)
label = Input(outputs)

network = input
for i in range(0, hiddenLayers):
  network = Dense(hiddenLayers, init = glorot_uniform(), activation = COps.relu)(network)
network = Dense(outputs, init=glorot_uniform(), activation=None)(network)

loss = COps.cross_entropy_with_softmax(network, label)
label_error = COps.classification_error(network, label)
lr_per_minibatch = learning_rate_schedule(0.125, UnitType.minibatch)
trainer = cntk.Trainer(network, (loss, label_error), [sgd(network.parameters, lr=lr_per_minibatch)])

# Initialize the parameters for the trainer, we will train in minibatches corresponding to the 20 game definition
minibatch_size = gameLength
num_minibatches = len(training_features) // minibatch_size

print(num_minibatches)

# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"
    if mb%frequency == 0:
        training_loss = get_train_loss(trainer)
        eval_error = get_train_eval_criterion(trainer)
        if verbose: 
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))
    return mb, training_loss, eval_error

# Run the trainer on and perform model training
training_progress_output_freq = 1

# Visualize the loss over minibatch
plotdata = {"batchsize":[], "loss":[], "error":[]}

tf = np.split(training_features, num_minibatches)
tl = np.split(training_labels, num_minibatches)

print("Number of mini batches")
print(len(tf))

print("The shape of the training feature minibatch and label minibatch")
print(tf[0].shape)
print(tl[0].shape)

for i in range(0, int(num_minibatches)):
    features = np.ascontiguousarray(tf[i%num_minibatches])
    labels = np.ascontiguousarray(tl[i%num_minibatches])
    
    # Specify the mapping of input variables in the model to actual minibatch data to be trained with
    trainer.train_minibatch({input : features, label : labels})
    batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=1)

network.save(os.path.join(script_directory, "rps.model"))

# Testing
#test_minibatch_size = gameLength
#features, labels = generate_random_data_sample(test_minibatch_size, inputs, outputs)
#avg_error = trainer.test_minibatch({input : features, label : labels})
#print("Average error: {0:2.2f}%".format(avg_error * 100))

# Evaluation
# out = COps.softmax(network)
# predicted_label_probs = out.eval({input : features})
