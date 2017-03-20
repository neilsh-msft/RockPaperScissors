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

            var processLauncherResult = await ProcessLauncher.RunToCompletionAsync(@"EvalCS.exe", "", processLauncherOptions);
            if (processLauncherResult.ExitCode == 0)
            {
                using (var outStreamRedirect = standardOutput.GetInputStreamAt(0))
                {
                    var size = standardOutput.Size;
                    using (var dataReader = new DataReader(outStreamRedirect))
                    {
                        var bytesLoaded = await dataReader.LoadAsync((uint)size);
                        var stringRead = dataReader.ReadString(bytesLoaded);
                        var result = stringRead.Trim();
                        return result;
                    }
                }
            }
            else
            {
                throw new Exception("Cannot start EvalCS");
            }
        }

        public async Task<HandResult> ComputerMove(int moveNumber, List<HandResult> previousHumanMoves, List<HandResult> previousComputerMoves)
        {
#if false
            // Some fake input
            var buff = new byte[] { (byte)'a', (byte)'b', (byte)'c' };
            var move = await LaunchEvalCommand(buff);
#endif
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
