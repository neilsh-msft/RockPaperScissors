using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
