using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RockPaperScissors
{
    class GameEngine
    {
        public HandResult ComputerMove(int moveNumber, List<HandResult> previousHumanMoves, List<HandResult> previousComputerMoves)
        {
            // For now, always return paper
            return HandResult.Paper;
        }

        // Returns 1 if move1 beats move2; Return -1 if move1 is weaker than move2; Return 0 on draw
        public static int Compare(HandResult move1, HandResult move2)
        {
            if (move1 == move2) return 0;
            if (move1 == HandResult.Rock)  return move2 == HandResult.Paper ? -1 : 1;
            if (move1 == HandResult.Paper) return move2 == HandResult.Rock ? 1 : -1;
            System.Diagnostics.Debug.Assert(move1 == HandResult.Scissors);
            return move2 == HandResult.Paper ? 1 : -1;
        }
    }
}
