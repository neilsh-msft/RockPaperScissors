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

        static readonly Dictionary<string, float[]> unpack = new Dictionary<string, float[]>
        {
            ["R"] = new float[] { 1, 0, 0 },
            ["P"] = new float[] { 0, 1, 0 },
            ["S"] = new float[] { 0, 0, 1 }
        };

        static readonly Dictionary<string, int> winLossStates = new Dictionary<string, int>
        {
            ["RR"] = 0,
            ["RP"] = 1,
            ["RS"] = -1,
            ["PR"] = -1,
            ["PP"] = 0,
            ["PS"] = 1,
            ["SR"] = 1,
            ["SP"] = -1,
            ["SS"] = 0
        };

        static IEnumerable<float> PrepareEvalData(string input)
        {
            IEnumerable<string> stringArray = input.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Take(10);
            var initialSize = stringArray.Count();

            while (stringArray.Any())
            {
                var movePair = stringArray.Take(2);
                var humanMove = movePair.First();
                var computerMove = movePair.Skip(1).First();

                foreach (float m in unpack[humanMove])
                    yield return m;
                foreach (float m in unpack[computerMove])
                    yield return m;

                int winLoss = winLossStates[humanMove + computerMove];
                if (winLoss == 0) winLoss = 1;
                if (winLoss == -1) winLoss = 0;
                yield return winLoss;

                stringArray = stringArray.Skip(2);
            }

            // Padding:
            for (int i = 0; i < 5 - initialSize/2; ++i)
            {
                for (int j = 0; j < 6; ++j) yield return 0;
                yield return 1;
            }
        }

        static string InvokeEval(List<float> inputValues, string modelFile)
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

            model.Evaluate(inputs, outputs, DeviceDescriptor.DefaultDevice());

            // Get evaluate result as dense output
            var outputBuffer = new List<List<float>>();
            var outputVal = outputs[model.Output];
            outputVal.CopyVariableValueTo(model.Output, outputBuffer);

            var probabilities = outputBuffer[0];
            int indexOfMax = probabilities.IndexOf(probabilities.Max());

            var computerMove = new[] { "R", "P", "S" }[indexOfMax];

            return computerMove;
        }

        const int bufferSize = 1024;
        static int Main(string[] args)
        {
            if(args.Length != 1)
            {
                Console.WriteLine("Need parameter: path to model file");
                return 1;
            }

            var modelFile = args[0];

            if (!File.Exists(modelFile))
            {
                Console.WriteLine("Model file {0} not found", modelFile);
                return 1;
            }

#if true// real input
            var input = Encoding.Default.GetString(GetInputBytes()).Trim();
#else // useful for debugging
            //var input = "R R R R R R R R R R";
            //var input = "S S S S S S S S S S";
            //var input = "P P P P P P P P P P";
            var input = "";
#endif
            try
            {
                var evalData = PrepareEvalData(input).ToList();
                System.Diagnostics.Debug.Assert(evalData.Count() == 35);
                var output = InvokeEval(evalData, modelFile);
                Console.Write(output);
                return 0;
            }
            catch(Exception ex)
            {
                Console.Write(ex.Message);
                return 1;
            }
        }
    }
}
