#define RUN_CONTINUOUS
#define RUN_EVALCS

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading;
// using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;

using Windows.Media.Capture;
using Windows.Storage;

using Windows.Storage.Streams;
using Windows.Graphics.Imaging;

using Windows.UI.Xaml.Media.Imaging;

using Windows.ApplicationModel;
using System.Threading.Tasks;
using Windows.System.Display;
using Windows.Graphics.Display;
using Windows.Media.MediaProperties;
using System.Diagnostics;
using OpenCvSharp;
using System.Net.Http;
using System.Text;

// The Blank Page item template is documented at http://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace RockPaperScissors
{
    class CloudConfig
    {
        public bool DownloadRemoteFile;
        public string WebAPIDownloadUri;
        public bool EnableCloudTraining;
        public string WebAPITrainUri;
    }

    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        private MediaCapture _mediaCapture;
        private int _slider = 50;
        private int _slider2 = 50;
        DispatcherTimer _dispatcherTimer;
        int _countDown = 0;
        GameEngine _gameEngine;
        CloudConfig _cloudConfig;

        // What was the latest hand gesture we saw?
        HandResult lastHandResult = HandResult.None;

        // How many times did we see it?
        int handResultCount = 0;

        // How many time do we need to see it before we recognize it?
        const int requiredCount = 2;

        int totalGamesPlayed = 0;
        int wins = 0;
        int losses = 0;
        int draws = 0;

        // List of all moves used for eval and training
        List<HandResult> humanMoves = new List<HandResult>();
        List<HandResult> computerMoves = new List<HandResult>();

        // Image assets
        static Dictionary<HandResult, BitmapImage> HandResultToImage = new Dictionary<HandResult, BitmapImage>();

        public MainPage()
        {
            this.InitializeComponent();

            humanMoves.Add(HandResult.Rock);
            humanMoves.Add(HandResult.Rock);
            humanMoves.Add(HandResult.Rock);

            computerMoves.Add(HandResult.Paper);
            computerMoves.Add(HandResult.Paper);
            computerMoves.Add(HandResult.Paper);

#if RUN_EVALCS
            _cloudConfig = new CloudConfig
            {
                // Change this if you want to download the model from the cloud vs. embedded resource:
                DownloadRemoteFile = true,
                // Change this if you want to hit the Azure-hosted Web API app vs. localhost:
                WebAPIDownloadUri = "http://fetaeval.azurewebsites.net/api/model/latest",
                // Submit game data to the cloud for training:
                EnableCloudTraining = true,
                // Training intergace Uri:
                WebAPITrainUri = "http://fetaeval.azurewebsites.net/api/training/game"
            };
#else
            _cloudConfig = new CloudConfig
            {
                // Change this if you want to download the model from the cloud vs. embedded resource:
                DownloadRemoteFile = false,
                // Change this if you want to hit the Azure-hosted Web API app vs. localhost:
                WebAPIDownloadUri = string.Empty,
                // Submit game data to the cloud for training:
                EnableCloudTraining = false,
                // Training intergace Uri:
                WebAPITrainUri = string.Empty
            };
#endif
            HandResultToImage[HandResult.Paper] = new BitmapImage(new Uri("ms-appx:///Assets/Paper.png"));
            HandResultToImage[HandResult.Rock] = new BitmapImage(new Uri("ms-appx:///Assets/Rock.png"));
            HandResultToImage[HandResult.Scissors] = new BitmapImage(new Uri("ms-appx:///Assets/Scissors.png"));

            var start = Start();
        }

        private async Task<string> DownloadModel()
        {
            try
            {
                byte[] bytes;
                if (_cloudConfig.DownloadRemoteFile)
                {
                    var client = new HttpClient();
                    bytes = await client.GetByteArrayAsync(_cloudConfig.WebAPIDownloadUri);
                }
                else
                {
                    bytes = File.ReadAllBytes("Assets\\rps.model");
                }
                var storageFolder = ApplicationData.Current.LocalFolder;
                var modelFile = await storageFolder.CreateFileAsync("rps.model", CreationCollisionOption.ReplaceExisting);
                await FileIO.WriteBufferAsync(modelFile, bytes.AsBuffer());
                return modelFile.Path;
            }
            catch (Exception)
            {
                Debug.WriteLine("Exception downloading model");
                return null;
            }
        }

        private async Task SubmitLatestGameToCloud(IEnumerable<HandResult> humanMoves, IEnumerable<HandResult> computerMoves)
        {
            if (_cloudConfig.EnableCloudTraining)
            {
                // Need to create a CSV file from humanMoves, computerMoves
                var str = Enumerable.Zip(humanMoves, computerMoves, (h, c) =>
                    String.Join(" ", GameEngine.HandResultToString[h], GameEngine.HandResultToString[c], GameEngine.Compare(h, c).ToString())).ToArray()
                    .Aggregate((a, b) => a + "\n" + b);

                var webAPIUri = _cloudConfig.WebAPITrainUri;

                var message = new HttpRequestMessage();
                var content = new MultipartFormDataContent();

                using (MemoryStream stream = new MemoryStream())
                {
                    StreamWriter streamWriter = new StreamWriter(stream);
                    streamWriter.Write(str);
                    streamWriter.Flush();
                    stream.Position = 0;

                    content.Add(new StreamContent(stream), "file", "game.csv");

                    message.Method = HttpMethod.Post;
                    message.Content = content;
                    message.RequestUri = new Uri(webAPIUri);

                    using (var client = new HttpClient())
                    {
                        var result = await client.SendAsync(message);
                        Debug.WriteLine(string.Format("Game uploaded, server response '{0}'", result.StatusCode));
                    }
                }
            }
        }

        private async Task Start()
        {
/* testing eval invocation
            var buff = Encoding.ASCII.GetBytes("aa");
            var ge = new GameEngine("");
            var move = await ge.LaunchEvalCommand(buff);
            return;
*/
            var modelFilePath = await DownloadModel();
            _gameEngine = new GameEngine(modelFilePath);
            _dispatcherTimer = new DispatcherTimer();
            _dispatcherTimer.Tick += dispatcherTimer_Tick;
            capture.Source = new SoftwareBitmapSource();
            mask1.Source = new SoftwareBitmapSource();
            mask2.Source = new SoftwareBitmapSource();
            mask3.Source = new SoftwareBitmapSource();
            mask4.Source = new SoftwareBitmapSource();
#if RUN_CONTINUOUS
            // Just always run for now.
            button_Click(null, null);
#endif
        }

        private void MediaCapture_Failed(MediaCapture sender, MediaCaptureFailedEventArgs errorEventArgs)
        {
            throw new NotImplementedException();
        }

        private async void dispatcherTimer_Tick(object sender, object e)
        {
            _countDown -= 200;
//            button.Content = Math.Ceiling(_countDown / 1000.0).ToString();

            if (_countDown <= -200)
            {
                _dispatcherTimer.Stop();

                // Prepare and capture photo
                if (!_mediaCapture.VideoDeviceController.Exposure.TrySetValue(
                    _mediaCapture.VideoDeviceController.Exposure.Capabilities.Min +
                    (_mediaCapture.VideoDeviceController.Exposure.Capabilities.Max -
                     _mediaCapture.VideoDeviceController.Exposure.Capabilities.Min) * _slider / 100))
                {
                    slider.IsEnabled = false;
                }
                else
                {
                    slider.IsEnabled = true;
                }

#if false
                if (_mediaCapture.VideoDeviceController.WhiteBalanceControl.Supported)
                {
                    var preset = _mediaCapture.VideoDeviceController.WhiteBalanceControl.Preset;
                    if (preset == Windows.Media.Devices.ColorTemperaturePreset.Auto)
                    {
                        await _mediaCapture.VideoDeviceController.WhiteBalanceControl.SetPresetAsync(
                            Windows.Media.Devices.ColorTemperaturePreset.Manual);
                    }

                    await _mediaCapture.VideoDeviceController.WhiteBalanceControl.SetValueAsync(
        _mediaCapture.VideoDeviceController.WhiteBalanceControl.Min +
        (_mediaCapture.VideoDeviceController.WhiteBalanceControl.Max -
         _mediaCapture.VideoDeviceController.WhiteBalanceControl.Min) * (uint)_slider2 / 100);
                }
                else
                {
                    if (!_mediaCapture.VideoDeviceController.WhiteBalance.TrySetValue(
        _mediaCapture.VideoDeviceController.WhiteBalance.Capabilities.Min +
        (_mediaCapture.VideoDeviceController.WhiteBalance.Capabilities.Max -
         _mediaCapture.VideoDeviceController.WhiteBalance.Capabilities.Min) * _slider2 / 100))
                    {
                        slider2.IsEnabled = false;
                    }
                    else
                    {
                        slider2.IsEnabled = true;
                    }
                }
#endif
                var lowLagCapture = await _mediaCapture.PrepareLowLagPhotoCaptureAsync(ImageEncodingProperties.CreateUncompressed(MediaPixelFormat.Bgra8));

                var capturedPhoto = await lowLagCapture.CaptureAsync();
                var softwareBitmap = capturedPhoto.Frame.SoftwareBitmap;
                await lowLagCapture.FinishAsync();

                SoftwareBitmap softwareBitmapBGRA8 = SoftwareBitmap.Convert(softwareBitmap,
                            BitmapPixelFormat.Bgra8,
                            BitmapAlphaMode.Premultiplied);

                Mat mat = SoftwareBitmapToMat(softwareBitmapBGRA8);

                HandDetect detector = new HandDetect(mat);
                detector.palmRatio = 1.6 + (1.2 * (_slider2 - 50) / 50);

                // detect the hand
                Scalar minYCrCb, maxYCrCb;
                Mat mask;
                bool handInRange = false;

                detector.SkinColorModel(mat, out maxYCrCb, out minYCrCb);
                mask = detector.HandDetection(mat, maxYCrCb, minYCrCb);

                detector.GetHull();
                int dfts = detector.GetPalmCenter();
                int tips = 0;
                if (dfts > 0)
                {
                    tips = detector.GetFingerTips();
                }

                // check if in range and determine hand being played
                handInRange = detector.OnHotspot(new Point(250, 240));
                var humanMove = detector.Detect(tips, dfts);

                fingerTips.Text = String.Format("Fingertips: {0}", tips);
                fingerDfcts.Text = String.Format("Defects: {0}", dfts);
                detectedPlay.Text = String.Format("Playing: {0}", humanMove);

                // Output openCV mats
                if (detector.myframe != null)
                {
                    await ((SoftwareBitmapSource)capture.Source).SetBitmapAsync(MatToSoftwareBitmap(detector.myframe));
                }

                if (detector.mask1 != null)
                {
                    await ((SoftwareBitmapSource)mask1.Source).SetBitmapAsync(MatToSoftwareBitmap(detector.mask1));
                }

                if (detector.mask2 != null)
                {
                    await ((SoftwareBitmapSource)mask2.Source).SetBitmapAsync(MatToSoftwareBitmap(detector.mask2));
                }

                if (detector.mask3 != null)
                {
                    await ((SoftwareBitmapSource)mask3.Source).SetBitmapAsync(MatToSoftwareBitmap(detector.mask3));
                }

                if (detector.mask4 != null)
                {
                    await ((SoftwareBitmapSource)mask4.Source).SetBitmapAsync(MatToSoftwareBitmap(detector.mask4));
                }

                Debug.WriteLine(string.Format("{0} : current: {1}, last: {2}", DateTime.Now.ToLocalTime(), humanMove, lastHandResult));

                // Determine winner of match
                if (humanMove != HandResult.None)
                {
                    if (humanMove == lastHandResult)
                    {
                        handResultCount++;
                        if (handResultCount >= requiredCount && handInRange)
                        {
                            // We've seen the same gesture requiredCount times. Accept it and kick off the eval
                            handResultCount = 0;
                            var computerMove = await _gameEngine.ComputerMove(totalGamesPlayed, humanMoves, computerMoves);
                            mask2.Source = HandResultToImage[humanMove];
                            mask3.Source = HandResultToImage[computerMove];
                            var winOrLose = GameEngine.Compare(humanMove, computerMove);

                            switch (winOrLose)
                            {
                                case 1:
                                    wins++;
                                    tbResult.Text = "You win";
                                    ((StackPanel)tbResult.Parent).Background = new SolidColorBrush(Windows.UI.Colors.LightGreen);
                                    tbHuman.Text = wins.ToString();
                                    break;

                                case 0:
                                    draws++;
                                    tbResult.Text = "You draw";
                                    ((StackPanel)tbResult.Parent).Background = null;
                                    tbDraws.Text = draws.ToString();
                                    break;

                                case -1:
                                    losses++;
                                    tbResult.Text = "You lose";
                                    ((StackPanel)tbResult.Parent).Background = new SolidColorBrush(Windows.UI.Colors.Red);
                                    tbComputer.Text = losses.ToString();
                                    break;
                            }

                            tbGamesPlayed.Text = (++totalGamesPlayed).ToString();

                            humanMoves.Add(humanMove);
                            computerMoves.Add(computerMove);

                            Debug.WriteLine(string.Format("--> {0} : {1}", humanMove, computerMove));

                            // Every 20 moves, submit the history to the cloud for training
                            if(humanMoves.Count() >= 20)
                            {
                                // Copy moves to temporary arrays and then clear the tracking history
                                var humanMovesCopy = new HandResult[humanMoves.Count()];
                                humanMoves.CopyTo(humanMovesCopy);

                                var computerMovesCopy = new HandResult[humanMoves.Count()];
                                humanMoves.CopyTo(computerMovesCopy);

                                var submission = this.SubmitLatestGameToCloud(computerMovesCopy, computerMovesCopy);

                                // Do not await submission, let it proceed asynchronously but clear the history
                                humanMoves.Clear();
                                computerMoves.Clear();
                            }
                        }
                    }
                    else
                    {
                        // Reset the counter. The gestures we've seen so far weren't consistent enough to accept
                        handResultCount = 0;
                    }

                    button.IsEnabled = true;
                    button.Content = "Play";
                }

                lastHandResult = humanMove;
#if RUN_CONTINUOUS
                _dispatcherTimer.Start();
#endif
            }
        }

        SoftwareBitmap MatToSoftwareBitmap(Mat image)
        {
            // Create the WriteableBitmap
            SoftwareBitmap result = new SoftwareBitmap(BitmapPixelFormat.Bgra8, image.Cols, image.Rows, BitmapAlphaMode.Premultiplied);
            byte[] bytes = new byte[image.Cols * image.Rows * 4];
            if (image.Type() == MatType.CV_8UC4)
            {
                // Already in BGRA format, copy bytes
                System.Runtime.InteropServices.Marshal.Copy(image.Data, bytes, 0, bytes.Length);
            }
            else if (image.Type() == MatType.CV_8UC3)
            {
                // in BGR format, add the Alpha plane
                Mat C255 = new Mat(image.Size(), MatType.CV_8UC1, new Scalar(255));
                Mat temp = new Mat(image.Size(), MatType.CV_8UC4);

                Mat[] channels = Cv2.Split(image);
                Cv2.Merge(new Mat[] { channels[0], channels[1], channels[2], C255 }, temp);

                System.Runtime.InteropServices.Marshal.Copy(temp.Data, bytes, 0, bytes.Length);
            }
            else if (image.Type() == MatType.CV_8UC1)
            {
                // single monochrome plane
                Mat C255 = new Mat(image.Size(), MatType.CV_8UC1, new Scalar(255));
                Mat temp = new Mat(image.Size(), MatType.CV_8UC4);

                Mat[] channels = { image, image, image, C255 };
                Cv2.Merge(channels, temp);

                System.Runtime.InteropServices.Marshal.Copy(temp.Data, bytes, 0, bytes.Length);
            }
            result.CopyFromBuffer(bytes.AsBuffer());

            return result;
        }

        Mat SoftwareBitmapToMat(SoftwareBitmap image)
        {
            byte[] bytes = new byte[image.PixelHeight * image.PixelWidth * 4];
            image.CopyToBuffer(bytes.AsBuffer());
            Mat result = new Mat(image.PixelHeight, image.PixelWidth, MatType.CV_8UC4, bytes);

            return result;
        }

        private async Task<bool> startCapture()
        {
            try
            {
                _mediaCapture = new MediaCapture();
                await _mediaCapture.InitializeAsync();

                _mediaCapture.VideoDeviceController.WhiteBalance.TrySetAuto(false);
                _mediaCapture.VideoDeviceController.Exposure.TrySetAuto(false);
                _mediaCapture.VideoDeviceController.Brightness.TrySetAuto(false);
                _mediaCapture.VideoDeviceController.Contrast.TrySetAuto(false);
                _mediaCapture.VideoDeviceController.BacklightCompensation.TrySetAuto(false);

                _mediaCapture.Failed += MediaCapture_Failed;

                PreviewControl.Source = _mediaCapture;
                await _mediaCapture.StartPreviewAsync();

                DisplayInformation.AutoRotationPreferences = DisplayOrientations.Landscape;

            }
            catch (UnauthorizedAccessException)
            {
                // This will be thrown if the user denied access to the camera in privacy settings
                System.Diagnostics.Debug.WriteLine("The app was denied access to the camera");
                return false;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine("MediaCapture initialization failed. {0}", ex.Message);
                return false;
            }
            return true;
        }

        private async void button_Click(object sender, RoutedEventArgs e)
        {
            if (await startCapture())
            {
                // start timer.  Period = 3 seconds, subtract 1 every seconds, capture at -200ms.
                _countDown = 3000;
                _dispatcherTimer.Interval = TimeSpan.FromMilliseconds(200);

                _dispatcherTimer.Start();
                button.IsEnabled = false;
            }
        }

        private void slider_ValueChanged(object sender, RangeBaseValueChangedEventArgs e)
        {
            _slider = (int)slider.Value;
        }
        private void slider2_ValueChanged(object sender, RangeBaseValueChangedEventArgs e)
        {
            _slider2 = (int)slider2.Value;
        }
    }
}
