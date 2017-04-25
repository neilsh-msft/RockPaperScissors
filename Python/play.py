import os, sys, csv
import numpy as np
from cntk.ops.functions import load_model

script_directory = os.path.dirname(sys.argv[0])
network = load_model(os.path.join(script_directory, "rps.model"))

# The model works by using lookback at the previous n moves - as determined by the size of the training set
# However, the LSTM model accepts sparse sequences, so it shows only one move as input
numberOfFeatures = 7
lookbackMoves = int(network.arguments[0].shape[0] / numberOfFeatures)
lookbackMoves = 3
gameLength = 20
moveNumber = 0

def encodeLabel(move):
  rock = np.float32(1.0) if move == 'R' else np.float32(0.0)
  paper = np.float32(1.0) if move == 'P' else np.float32(0.0)
  scissors = np.float32(1.0) if move == 'S' else np.float32(0.0)
  return [rock, paper, scissors]
  
def encodeFeature(hm, cm, wld): 
  return np.float32(np.r_[encodeLabel(hm), encodeLabel(cm), wld])
 
def defaultMove():
  return encodeFeature('X', 'X', np.float32(0.0))

previousMoves = []
counterMoves = ["P", "S", "R"]
moves = [
  ["R", "R", 0],
  ["R", "P", 1],
  ["R", "S", -1],
  ["P", "R", -1],
  ["P", "P", 0],
  ["P", "S", 1],
  ["S", "R", 1],
  ["S", "P", -1],
  ["S", "S", 0]];
winStates = ["WIN", "DRAW", "LOSE"]
  
def humanMove():
  move = ""
  while move not in counterMoves:
    move = input("Select (R)ock, (P)aper, or (S)cissors:")
  return move

def computerMove(moveNumber, previousHM, previousCM):
  global previousMoves
  if previousHM != "":
    wld = selectWinner(previousHM, previousCM)
    if wld == 1: wld = np.float32(1.0)
    if wld == 0: wld = np.float32(1.0)
    if wld == -1: wld = np.float32(0.0)

    if moveNumber % gameLength < lookbackMoves:
      previousMoves.append(encodeFeature(previousHM, previousCM, wld))
    else:
      previousMoves = np.float32(np.resize(np.roll(previousMoves, -1 * numberOfFeatures), (lookbackMoves - 1, numberOfFeatures)))
      previousMoves = np.resize(np.append(previousMoves, encodeFeature(previousHM, previousCM, wld)), (lookbackMoves, numberOfFeatures))
  else:
    previousMoves.append(defaultMove())

  eval_features = np.array(previousMoves, dtype="float32")
  result = np.array(network.eval({network.arguments[0]:[eval_features]})).flatten()
  return counterMoves[np.argmax(result)]
  
def selectWinner(human, computer):
  for move in moves:
    if move[0] == human and move[1] == computer: return move[2]

previousHumanMove = ""
previousComputerMove = ""
numWins = 0
with open(os.path.join(script_directory, "rps.csv"), 'a') as csvfile:
  recordWriter = csv.writer(csvfile, lineterminator='\n')
  while moveNumber < gameLength:
    hm = humanMove()
    cm = computerMove(moveNumber, previousHumanMove, previousComputerMove)
    winner = selectWinner(hm, cm)
    print ("You ", winStates[winner + 1], ".  The computer chose ", cm)
    if winner >= 0:
      numWins += 1

    # Record the result
    recordWriter.writerow([hm, cm, winner])
  
    previousHumanMove = hm
    previousComputerMove = cm
    moveNumber += 1

print("Computer wins: {0} / {1} ({2:2.2f}%)".format(int(numWins), int(gameLength), (numWins/gameLength) * 100))