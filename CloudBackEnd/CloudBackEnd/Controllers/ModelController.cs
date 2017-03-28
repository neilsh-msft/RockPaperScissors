using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Web.Http;

using System.Diagnostics;
using System.Threading.Tasks;
using System.IO;
using System.Runtime.InteropServices;

namespace CloudBackEnd.Controllers
{
    static class ModelFileLock
    {
        // Model file is locked while it is written to or read from
        public static object obj = new object();
    }

    [RoutePrefix("api/model")]
    public class ModelController : ApiController
    {
        [Route("latest")]
        public Task<HttpResponseMessage> GetLatestModel()
        {
            string modelFilePath = AppDomain.CurrentDomain.BaseDirectory + "CNTK\\Models\\rps.model";

            lock (ModelFileLock.obj)
            {
                var response = new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StreamContent(File.OpenRead(modelFilePath))
                };

                response.Content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/binary");

                return Task.FromResult(response);
            }
        }
    }

    [RoutePrefix("api/training")]
    public class GameUploadController : ApiController
    {
        [DllImport("ModelTrainerLib.dll")]
        static extern int TrainModel([MarshalAs(UnmanagedType.LPWStr)]String modelFileName, [MarshalAs(UnmanagedType.LPWStr)]String gameFileName);

        readonly string accumulatedGameHistoryFileName = "AllGames.csv";

        [Route("game")]
        [HttpPost]
        public async Task<HttpResponseMessage> UploadGameFile()
        {
            if (Request.Content.IsMimeMultipartContent())
            {
                var serverUploadFolder = AppDomain.CurrentDomain.BaseDirectory;
                var streamProvider = new MultipartFormDataStreamProvider(serverUploadFolder);
                await Request.Content.ReadAsMultipartAsync(streamProvider);

                foreach (MultipartFileData fileData in streamProvider.FileData)
                {
                    if (string.IsNullOrEmpty(fileData.Headers.ContentDisposition.FileName))
                    {
                        return Request.CreateResponse(HttpStatusCode.NotAcceptable);
                    }

                    string fileName = fileData.Headers.ContentDisposition.FileName;

                    string modelFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CNTK\\Models\\rps.model");
                    string gameFilePath = fileData.LocalFileName;

                    Debug.WriteLine(string.Format("Starting training. Model file: '{0}', Game file: '{1}'", modelFilePath, gameFilePath));

                    // Do we have the accumulatedGameHistoryFileName file? If not, create it from the received file
                    // If yet, concat the received file to it
                    var accumulatedGameHistoryFileNameFullPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, accumulatedGameHistoryFileName);
                    if(!File.Exists(accumulatedGameHistoryFileNameFullPath))
                    {
                        File.Copy(fileData.LocalFileName, accumulatedGameHistoryFileNameFullPath);
                    }
                    else
                    {
                        var lines = File.ReadAllLines(fileData.LocalFileName);
                        File.AppendAllLines(accumulatedGameHistoryFileNameFullPath, lines);
                    }

                    int trainingResult;

                    lock (ModelFileLock.obj)
                    {
                        trainingResult = TrainModel(modelFilePath, accumulatedGameHistoryFileNameFullPath);
                        Debug.WriteLine(string.Format("Trainer reports error code={0}", trainingResult));
                    }

                    // Delete the downloaded game file
                    if (File.Exists(gameFilePath))
                    {
                        File.Delete(gameFilePath);
                    }

                    if (trainingResult != 0)
                    {
                        return Request.CreateResponse(HttpStatusCode.InternalServerError);
                    }

                }

                return Request.CreateResponse(HttpStatusCode.OK);
            }
            else
            {
                return Request.CreateResponse(HttpStatusCode.NotAcceptable);
            }
        }
    }
}
