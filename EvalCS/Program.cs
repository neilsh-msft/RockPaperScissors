using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using CNTK;
using System.IO;

namespace EvalCS
{
    class Program
    {
        static byte[] GetInputBytes()
        {
            List<byte> allbytes = new List<byte>();

            using (var s = Console.OpenStandardInput())
            {
                int offset = 0;
                while (true)
                {
                    byte[] buffer = new byte[bufferSize];
                    int bytesRead = s.Read(buffer, offset, bufferSize);
                    allbytes.AddRange(buffer.Take(bytesRead));
                    if (bytesRead < bufferSize) break;
                    offset += bytesRead;
                }
            }
            return allbytes.ToArray();
        }

        static string ProcessData(List<float> inputValues, string modelFile)
        {
            // Load the already trained model
            var model = Function.LoadModel(modelFile);

            // Setup the input values
            // The RPS game is currently a list of 35 floats [ HR0 HP0 HS0 CR0 CP0 CS0 WLD0 .... ]
            var inputs = new Dictionary<Variable, Value>();

            // Create the input variable
            var inputVariable = model.Arguments.Single();
            var values = Value.CreateSequence(inputVariable.Shape, inputValues, DeviceDescriptor.DefaultDevice());
            inputs.Add(inputVariable, values);

            // Setup the output variables
            var outputs = new Dictionary<Variable, Value>();
            outputs.Add(model.Output, null);

            try
            {
                model.Evaluate(inputs, outputs, DeviceDescriptor.DefaultDevice());

                // Get evaluate result as dense output
                var outputBuffer = new List<List<float>>();
                var outputVal = outputs[model.Output];
                outputVal.CopyVariableValueTo(model.Output, outputBuffer);

                var probabilities = outputBuffer[0];
                int indexOfMax = probabilities.IndexOf(probabilities.Max());

                var nextPredictedHumanMove = new[] { "R", "P", "S" }[indexOfMax];

                var computerMove = nextPredictedHumanMove == "R" ? "P" : nextPredictedHumanMove == "P" ? "S" : "R";

                return computerMove;

            }
            catch (Exception)
            {
                // TODO: handle error
                return "P";
            }
        }

        const int bufferSize = 1024;
        static int Main(string[] args)
        {
            if(args.Length != 1)
            {
                Console.WriteLine("Need parameter: path to model file");
                return -1;
            }

            var modelFile = args[0];

            if (!File.Exists(modelFile))
            {
                Console.WriteLine("Model file {0} not found", modelFile);
                return -1;
            }

#if true // real input
            var input = Encoding.Default.GetString(GetInputBytes()).Trim();
#else // useful for debugging
            var input = "R P R S R P S R S R";
#endif

            var stringArray = input.Split(' ');

            var humanMoves = stringArray.Where((m, i) => i % 2 == 0);
            var computerMoves = stringArray.Where((m, i) => i % 2 == 1);

            Func<string, float[]> unpack = (string s) =>
            {
                switch (s)
                {
                    case "R": return new float[] { 1, 0, 0 };
                    case "P": return new float[] { 0, 1, 0 };
                    case "S": return new float[] { 0, 0, 1 };
                    default:
                        Console.WriteLine("'{0}' unknown", s);
                        throw new NotSupportedException();
                }
            };

            Func<string, string, int> WinOrLoss = (string move1, string move2) =>
            {
                if (move1 == move2) return 1; // draw is considered 'win'
                if (move1 == "R") return move2 == "P" ? 0 : 1;
                if (move1 == "P") return move2 == "R" ? 1 : 0;
                System.Diagnostics.Debug.Assert(move1 == "S");
                return move2 == "P" ? 1 : 0;
            };

            var movesWithWinLoss = Enumerable.Zip(humanMoves, computerMoves, (h, m) =>
            {
                var list = Enumerable.Concat(unpack(h), unpack(m)).ToList();
                list.Add(WinOrLoss(h, m));
                return list;
            }).SelectMany(_ => _);


            // Produce output
            var output = ProcessData(movesWithWinLoss.ToList(), modelFile);

            Console.Write(output);

            return 0;
        }
    }
}
