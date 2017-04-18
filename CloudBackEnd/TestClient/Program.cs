using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Net.Http;
using System.IO;
using System.Net;

namespace TestClient
{
    class Program
    {
        static async Task TestModelDownload(string webAPIUri, string outputFile)
        {
            Console.WriteLine("Starting download...", outputFile);
            var client = new HttpClient();
            var bytes = await client.GetByteArrayAsync(webAPIUri);
            File.WriteAllBytes(outputFile, bytes);
            Console.WriteLine("File {0} downloaded", outputFile);
        }

        static async Task TestTraining(string webAPIUri, string gameFile)
        {
            var message = new HttpRequestMessage();
            var content = new MultipartFormDataContent();
            var files = new List<string> { gameFile };

            foreach (var file in files)
            {
                var filestream = new FileStream(file, FileMode.Open);
                var fileName = System.IO.Path.GetFileName(file);
                content.Add(new StreamContent(filestream), "file", fileName);
            }

            message.Method = HttpMethod.Post;
            message.Content = content;
            message.RequestUri = new Uri(webAPIUri);

            var client = new HttpClient();
            var result = await client.SendAsync(message);

            Console.WriteLine("File {0} upload ended with result {1}", gameFile, result.StatusCode);
        }

        static void Main(string[] args)
        {
            if (args.Count() == 1)
            {
                string command = args[0];
                if (command == "download")
                {
                    //var webAPIUri = "http://localhost:3470/api/model/latest";
                    var webAPIUri = "http://fetaeval.azurewebsites.net/api/model/latest";
                    var saveModelToFile = "downloaded.model";
                    var task = TestModelDownload(webAPIUri, saveModelToFile);
                    task.Wait();
                }
                else if (command == "train")
                {
                    //var webAPIUri = "http://localhost:3470/api/training/game";
                    var webAPIUri = "http://fetaeval.azurewebsites.net/api/training/game";
                    var gameFile = "D:\\g\\brie\\GameFiles\\rps.csv";
                    var task = TestTraining(webAPIUri, gameFile);
                    task.Wait();
                }
            }
        }
    }
}
