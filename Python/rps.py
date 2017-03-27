import os, sys
import random
import csv

script_directory = os.path.dirname(sys.argv[0])

validMoves = ["R", "P", "S"]
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

def computerMove(strategy, moveNumber, previousHM, previousCM):
  if strategy == 0: return randomMove()
  # elif strategy == 1: return titForTat(moveNumber, previousHM)
  else: return winStayLoseShift(moveNumber, previousHM, previousCM)

def randomMove():
  return moves[random.randint(0,8)][0]

def titForTat(move, previousHM):
  if move == 0: return randomMove()
  else: return previousHM
  
def winStayLoseShift(move, previousHM, previousCM):
  if move == 0: return randomMove()
  if selectWinner(previousHM, previousCM) == -1:
    return randomMove()
  else:
    return previousCM
  
def selectWinner(human, computer):
  for move in moves:
    if move[0] == human and move[1] == computer: return move[2]

moveNumber = 0
previousHumanMove = ""
previousComputerMove = ""
strategy = random.randint(0, 1)
with open(os.path.join(script_directory, "rps.csv"), 'a') as csvfile:
  recordWriter = csv.writer(csvfile, lineterminator='\n')
  while moveNumber < 20:
    hm = humanMove()
    cm = computerMove(strategy, moveNumber, previousHumanMove, previousComputerMove)
    winner = selectWinner(hm, cm)
    print ("You ", winStates[winner + 1], ".  The computer chose ", cm)

    # Record the result
    recordWriter.writerow([hm, cm, winner])
  
    previousHumanMove = hm
    previousComputerMove = cm
    moveNumber += 1
