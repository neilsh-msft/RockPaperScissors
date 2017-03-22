#define RUN_CONTINUOUS

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

// The Blank Page item template is documented at http://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace RockPaperScissors
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        private MediaCapture _mediaCapture;
        private bool _hsv = false;
        private int _slider = 50;
        private int _slider2 = 50;
        private bool _captureBackGround;
        private Mat _background;
        DispatcherTimer _dispatcherTimer;
        int _countDown;
        GameEngine _gameEngine;

        // What was the latest hand gesture we saw?
        HandResult lastHandResult = HandResult.None;

        // How many times did we see it?
        int handResultCount = 0;

        // How many time do we need to see it before we recognize it?
        const int requiredCount = 2;

        int totalGamesPlayed = 0;

        // List of all moves used for eval and training
        List<HandResult> humanMoves = new List<HandResult>();
        List<HandResult> computerMoves = new List<HandResult>();

        public MainPage()
        {
            this.InitializeComponent();
            _gameEngine = new GameEngine();
            _dispatcherTimer = new DispatcherTimer();
            _dispatcherTimer.Tick += dispatcherTimer_Tick;

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
            button.Content = Math.Ceiling(_countDown / 1000.0).ToString();

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

                if (_captureBackGround)
                {
                    _background = mat.Clone();
                    _captureBackGround = false;
                }

                HandDetect detector = new HandDetect(mat, _hsv, _background);
                detector.palmRatio = 1.6 + (1.2 * (_slider2 - 50) / 50);

                // detect the hand
                Scalar minYCrCb, maxYCrCb;
                Mat mask;

                detector.SkinColorModel(mat, out maxYCrCb, out minYCrCb);
                mask = detector.HandDetection(mat, maxYCrCb, minYCrCb);

                detector.GetHull();
                int dfts = detector.GetPalmCenter();
                int tips = 0;
                if (dfts > 0)
                {
                    tips = detector.GetFingerTips();
                }

                fingerTips.Text = String.Format("Fingertips: {0}", tips);
                fingerDfcts.Text = String.Format("Defects: {0}", dfts);

                SoftwareBitmap result = MatToSoftwareBitmap(detector.myframe);
                SoftwareBitmapSource bitmapSource = new SoftwareBitmapSource();
                await bitmapSource.SetBitmapAsync(result);
                capture.Source = bitmapSource;

                if (detector.mask1 != null)
                {
                    mask1.Source = new SoftwareBitmapSource();
                    await ((SoftwareBitmapSource)mask1.Source).SetBitmapAsync(MatToSoftwareBitmap(detector.mask1));
                }

                if (detector.mask2 != null)
                {
                    mask2.Source = new SoftwareBitmapSource();
                    await ((SoftwareBitmapSource)mask2.Source).SetBitmapAsync(MatToSoftwareBitmap(detector.mask2));
                }

                if (detector.mask3 != null)
                {
                    mask3.Source = new SoftwareBitmapSource();
                    await ((SoftwareBitmapSource)mask3.Source).SetBitmapAsync(MatToSoftwareBitmap(detector.mask3));
                }

                if (detector.mask4 != null)
                {
                    mask4.Source = new SoftwareBitmapSource();
                    await ((SoftwareBitmapSource)mask4.Source).SetBitmapAsync(MatToSoftwareBitmap(detector.mask4));
                }

                var humanMove = detector.Detect(tips, dfts);

                Debug.WriteLine(string.Format("{0} : current: {1}, last: {2}", DateTime.Now.ToLocalTime(), humanMove, lastHandResult));

                if (humanMove != HandResult.None)
                {
                    if (humanMove == lastHandResult)
                    {
                        handResultCount++;
                        if (handResultCount == requiredCount)
                        {
                            // We've seen the same gesture requiredCount times. Accept it and kick of the eval
                            tbHuman.Text = humanMove.ToString();
                            handResultCount = 0;
                            var computerMove = await _gameEngine.ComputerMove(totalGamesPlayed, humanMoves, computerMoves);
                            tbComputer.Text = computerMove.ToString();
                            var winOrLose = GameEngine.Compare(humanMove, computerMove);
                            tbResult.Text = winOrLose == 1 ? "You win" : winOrLose == -1 ? "You lose" : "Draw";
                            tbGamesPlayed.Text = (++totalGamesPlayed).ToString();

                            humanMoves.Add(humanMove);
                            computerMoves.Add(computerMove);

                            Debug.WriteLine(string.Format("--> {0} : {1}", humanMove, computerMove));
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
                System.Runtime.InteropServices.Marshal.Copy(image.Data, bytes, 0, bytes.Length);
            }
            else if (image.Type() == MatType.CV_8UC3)
            {
                Mat C255 = new Mat(image.Size(), MatType.CV_8UC1, new Scalar(255));
                Mat temp = new Mat(image.Size(), MatType.CV_8UC4);

                Mat[] channels = Cv2.Split(image);
                Cv2.Merge(new Mat[] { channels[0], channels[1], channels[2], C255 }, temp);

                System.Runtime.InteropServices.Marshal.Copy(temp.Data, bytes, 0, bytes.Length);
            }
            else if (image.Type() == MatType.CV_8UC1)
            {
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

        private async void button_Click(object sender, RoutedEventArgs e)
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

                // start timer.  Period = 3 seconds, subtract 1 every seconds, capture at -200ms.
                _countDown = 3000;
                _dispatcherTimer.Interval = TimeSpan.FromMilliseconds(200);

                _dispatcherTimer.Start();
                button.IsEnabled = false;
            }
            catch (UnauthorizedAccessException)
            {
                // This will be thrown if the user denied access to the camera in privacy settings
                System.Diagnostics.Debug.WriteLine("The app was denied access to the camera");
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine("MediaCapture initialization failed. {0}", ex.Message);
            }
        }

        private void background_Click(object sender, RoutedEventArgs e)
        {
            _captureBackGround = true;
        }

        private void hsvSwitch_Toggled(object sender, RoutedEventArgs e)
        {
            _hsv = hsvSwitch.IsOn;
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
