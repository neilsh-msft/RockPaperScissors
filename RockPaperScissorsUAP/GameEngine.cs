using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Windows.Storage.Streams;
using Windows.System;
using System.Runtime.InteropServices.WindowsRuntime;

namespace RockPaperScissors
{
    class GameEngine
    {
        string modelFilePath;
        public GameEngine(string modelFilePath)
        {
            this.modelFilePath = modelFilePath;
        }

        async Task<string> LaunchEvalCommand(byte[] buff)
        {
            var processLauncherOptions = new ProcessLauncherOptions();
            var standardOutput = new InMemoryRandomAccessStream();
            var standardInput = new InMemoryRandomAccessStream();

            IBuffer ibuff = buff.AsBuffer();
            await standardInput.WriteAsync(ibuff);

            processLauncherOptions.StandardOutput = standardOutput;
            processLauncherOptions.StandardError = null;
            processLauncherOptions.StandardInput = standardInput.GetInputStreamAt(0);

            standardInput.Dispose();

            var processLauncherResult = await ProcessLauncher.RunToCompletionAsync(@"EvalCS.exe", this.modelFilePath, processLauncherOptions);
            using (var outStreamRedirect = standardOutput.GetInputStreamAt(0))
            {
                var size = standardOutput.Size;
                using (var dataReader = new DataReader(outStreamRedirect))
                {
                    var bytesLoaded = await dataReader.LoadAsync((uint)size);
                    var stringRead = dataReader.ReadString(bytesLoaded);
                    var result = stringRead.Trim();
                    if (processLauncherResult.ExitCode == 0)
                    {
                        return result;
                    }
                    // For non-0 return, the string should be the exception message
                    throw new Exception(string.Format("Exception from EvalCS: {0}", result));
                }
            }
        }

        static readonly Dictionary<HandResult, string> HandResultToString = new Dictionary<HandResult, string>
        {
            [HandResult.Rock]     = "R",
            [HandResult.Paper]    = "P",
            [HandResult.Scissors] = "S",
        };

        static readonly Dictionary<string, HandResult> StringToHandResult= new Dictionary<string, HandResult>
        {
            ["R"] = HandResult.Rock,
            ["P"] = HandResult.Paper,
            ["S"] = HandResult.Scissors
        };

        public async Task<HandResult> ComputerMove(int moveNumber, List<HandResult> previousHumanMoves, List<HandResult> previousComputerMoves)
        {
            System.Diagnostics.Debug.Assert(previousHumanMoves.Count == previousComputerMoves.Count);

            if(previousHumanMoves.Count == 0)
            {
                // No prior history, make an arbitrary move
                return HandResult.Paper;
            }
#if true
            var str = previousHumanMoves
                        .Zip(previousComputerMoves, (h, c) => new[] { h, c })
                        .SelectMany(_ => _)
                        .Select(_ => HandResultToString[_])
                        .Aggregate((current, next) => current + " " + next);

            var buff = Encoding.ASCII.GetBytes(str);
            var move = await LaunchEvalCommand(buff);
            return StringToHandResult[move];
#else
            // For now, always return paper
            return HandResult.Paper;
#endif
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
