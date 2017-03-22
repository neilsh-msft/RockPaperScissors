using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using CNTK;

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

        static byte[] ProcessData(byte[] input)
        {
            // Load the already trained model
            var model = Function.LoadModel("C:\\temp\\brie\\Python\\rps.model");

            // Setup the input values
            // The RPS game is currently a list of 35 floats [ HR0 HP0 HS0 CR0 CP0 CS0 WLD0 .... ]
            var inputs = new Dictionary<Variable, Value>();
            var inputValues = new List<float>();

            // TODO: Add the input here
            for (int i = 0; i < 35; i++) inputValues.Add(0.0f);

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
            }
            catch (Exception exc)
            {
                return null;
            }

            // TODO: Parse the output here
            return input.Select(c => (byte)(Char.ToUpper((char)c))).ToArray();
        }

        const int bufferSize = 1024;
        static void Main(string[] args)
        {
            // Read buffer from stdin
            var input = GetInputBytes();

            // Produce output
            var output = ProcessData(input);

            // Send it to stdout
            using (var s = Console.OpenStandardOutput())
            {
                s.Write(output, 0, output.Length);
            }
        }
    }
}
