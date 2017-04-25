import os, sys, csv
import numpy as np
from cntk.ops.functions import load_model

script_directory = os.path.dirname(sys.argv[0])
network = load_model(os.path.join(script_directory, "rps.model"))

# The model works by using lookback at the previous n moves - as determined by the size of the training set
lookbackMoves = int(network.arguments[0].shape[0] / 7)
gameLength = 20

def encodeLabel(move):
  rock = 1 if move == 'R' else 0
  paper = 1 if move == 'P' else 0
  scissors = 1 if move == 'S' else 0
  return [rock, paper, scissors]
  
def encodeFeature(hm, cm, wld):
  return np.r_[encodeLabel(hm), encodeLabel(cm), wld].tolist()
  
def defaultMove():
  return [0, 0, 0]

previousMoves = np.array([encodeFeature('X', 'X', 1) for x in range(lookbackMoves)]).flatten().tolist()

validMoves = ["R", "P", "S"]
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
  while move not in validMoves:
    move = input("Select (R)ock, (P)aper, or (S)cissors:")
  return move

def computerMove(moveNumber, previousHM, previousCM):
  global previousMoves
  if previousHM != "":
    wld = selectWinner(previousHM, previousCM)
    if wld == 0: wld = 1
    if wld == -1: wld = 0
    previousMoves = np.insert(np.resize(previousMoves, (1, 7 * (lookbackMoves - 1))), 0, encodeFeature(previousHM, previousCM, wld)).tolist()

  eval_features = np.array(previousMoves, dtype="float32")
  result = np.array(network.eval({network.arguments[0]:[eval_features]})).flatten()
  return counterMoves[np.argmax(result)]
  
def selectWinner(human, computer):
  for move in moves:
    if move[0] == human and move[1] == computer: return move[2]

moveNumber = 0
previousHumanMove = ""
previousComputerMove = ""
with open(os.path.join(script_directory, "rps.csv"), 'a') as csvfile:
  recordWriter = csv.writer(csvfile, lineterminator='\n')
  while moveNumber < gameLength:
    hm = humanMove()
    cm = computerMove(moveNumber, previousHumanMove, previousComputerMove)
    winner = selectWinner(hm, cm)
    print ("You ", winStates[winner + 1], ".  The computer chose ", cm)

    # Record the result
    recordWriter.writerow([hm, cm, winner])
  
    previousHumanMove = hm
    previousComputerMove = cm
    moveNumber += 1