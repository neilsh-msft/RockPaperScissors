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
    public class ModelController : ApiController
    {
        // GET api/values
        [SwaggerOperation("GetAll")]
        public async Task<HttpResponseMessage> Get()
        {
            string modelFilePath = AppDomain.CurrentDomain.BaseDirectory + "CNTK\\Models\\rps.model";

            var response = new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StreamContent(File.OpenRead(modelFilePath))
            };

            response.Content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/binary");

            return await Task.FromResult(response);
        }

        // PUT api/values/[GAMEID]
        [SwaggerOperation("Update")]
        [SwaggerResponse(HttpStatusCode.OK)]
        [SwaggerResponse(HttpStatusCode.NotFound)]
        public void Put(int gameId, [FromBody]string value)
        {

        }
    }
}
