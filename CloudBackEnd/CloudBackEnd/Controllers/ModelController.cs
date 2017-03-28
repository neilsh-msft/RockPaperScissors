using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Web.Http;
using Swashbuckle.Swagger.Annotations;

using System.Threading.Tasks;
using System.IO;
using System.Drawing;


namespace CloudBackEnd.Controllers
{
    [RoutePrefix("api/model")]
    public class ModelController : ApiController
    {
        [Route("latest")]
        public async Task<HttpResponseMessage> GetLatestModel()
        {
            string modelFilePath = AppDomain.CurrentDomain.BaseDirectory + "CNTK\\Models\\rps.model";

            var response = new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StreamContent(File.OpenRead(modelFilePath))
            };

            response.Content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/binary");

            return await Task.FromResult(response);
        }
    }

    [RoutePrefix("api/training")]
    public class GameUploadController : ApiController
    {
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

                    // If the file already exists, delete it
                    // TODO: race conditions if the file is used in training
                    var copyToName = Path.Combine(serverUploadFolder, fileName);
                    if (File.Exists(copyToName))
                    {
                        File.Delete(copyToName);
                    }
                    File.Move(fileData.LocalFileName, copyToName);

                    // TODO: kick off training
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
