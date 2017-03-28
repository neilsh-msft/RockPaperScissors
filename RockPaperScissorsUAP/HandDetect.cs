// https://github.com/zeruniverse/Gesture_Recognition/blob/master/Gesture_Recognition/handdetect.cpp

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading;
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

using OpenCvSharp;

namespace RockPaperScissors
{
    public static class CvExtensions
    {
        public static int Area(this Rect rect)
        {
            return rect.Height * rect.Width;
        }
    }

    public enum HandResult
    {
        None,
        Rock,
        Paper,
        Scissors
    }

    class HandDetect
    {
        private Scalar cyan = new Scalar(255, 255, 0, 255); // cyan
        private Scalar white = new Scalar(255, 255, 255, 255); // white
        private Scalar magenta = new Scalar(255, 0, 255, 255); // magenta
        private Scalar black = new Scalar(0, 0, 0, 255); // black
        private Scalar yellow = new Scalar(0, 255, 255, 255); // yellow
        private Scalar green = new Scalar(0, 255, 0, 255); // green
        private Scalar blue = new Scalar(255, 0, 0, 255); // blue
        private Scalar red = new Scalar(0, 0, 255, 255); // red

        private Scalar palmClr;
        private Scalar armClr;
        private Scalar hullClr;
        private Scalar fingerTipClr;

        Point[] contour = null;
        Point[] fingers = null;
        Point[] fingerdft = null;
        Point[] palm = null;
        Vec4i[] defects = null;

        bool hsv = false;

        RotatedRect arm;
        Point armCenter;
        Point palmCenter;
        double aspectRatio;
        int handWidth;
        int handLength;
        int palmRadius;

        public Mat myframe;
        public Mat lumMap;
        public Mat mask1;
        public Mat mask2;
        public Mat mask3;
        public Mat mask4;

        public double palmRatio = 1.6;

        public HandDetect(Mat frame)
        {
            myframe = frame.Clone();

            palmClr = cyan;
            fingerTipClr = black;
            armClr = yellow;
            hullClr = magenta;
        }

        private HandDetect()
        {
        }

        public void SkinColorModel(Mat frame, out Scalar max, out Scalar min)
        {
            Mat roiMap = Mat.Zeros(frame.Size(), MatType.CV_8UC1);
            Mat bgrMap = frame.CvtColor(ColorConversionCodes.BGRA2BGR);
            lumMap = bgrMap.CvtColor(hsv ? ColorConversionCodes.BGR2HSV :
                ColorConversionCodes.BGR2YCrCb);

//            mask1 = lumMap.Clone();

#if false
            Mat[] channels;
            Mat temp = frame.Clone();

            channels = temp.Split();
            mask1 = channels[0];
            mask2 = channels[1];
            mask3 = channels[2];
#endif

            max = new Scalar(255, 0, 0);
            min = new Scalar(0, 255, 255);

            if (hsv)
            {
                max = new Scalar(20, 150, 255);
                min = new Scalar(0, 30, 80);
            }
            else // YCrCb
            {
                max = new Scalar(255, 173, 127);
                min = new Scalar(0, 133, 77);
            }
        }


        public Mat HandDetection(Mat frame, Scalar maxYCrCb, Scalar minYCrCb)
        {
            // create a single channel mask
            Mat mask = Mat.Zeros(frame.Size(), MatType.CV_8UC1);

            Cv2.InRange(lumMap, minYCrCb, maxYCrCb, mask);

#if false
            // Get rid of single pixel noise from the mask
            Cv2.Erode(mask, mask, null, null, 2);

            // enlarge the areas within the mask to attempt to join them
            Mat element = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(10,10));
            Cv2.Dilate(mask, mask, element, null, 1);

            /*
            *                    Mat.  cvSmooth(hsv_mask, hsv_mask, CV_MEDIAN, 27, 0, 0, 0);
                        Cv2.Canny(hsv_mask, hsv_edge, 1, 3, 5);
            */

#else
            Cv2.PyrUp(mask, mask);
            mask = mask.Resize(frame.Size());
#endif
            //            mask3 = mask.Clone();

            Point[][] contours;
            HierarchyIndex[] hierarchy;

            Cv2.FindContours(mask, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxNone);

            // find longest contour
            int max_contour_size = -1;
            foreach (Point[] points in contours)
            {
                if (points.Length >= 370 && points.Length > max_contour_size)
                {
                    contour = points;
                    max_contour_size = contour.Length;
                }
            }

            if (contour != null)
            {
                Cv2.DrawContours(mask, new Point[][] { contour }, 0, new Scalar(50), thickness: -1, maxLevel: 1, lineType: LineTypes.AntiAlias);
                Cv2.DrawContours(mask, new Point[][] { contour }, 0, new Scalar(100), thickness: 2, maxLevel: 1, lineType: LineTypes.AntiAlias);

                arm = Cv2.MinAreaRect(contour);
                if (Math.Abs(arm.Angle) < 45)
                {
                    aspectRatio = arm.Size.Width / arm.Size.Height;
                    handWidth = (int)arm.Size.Height;
                    handLength = (int)arm.Size.Width;
                }
                else
                {
                    aspectRatio = arm.Size.Height / arm.Size.Width;
                    handWidth = (int)arm.Size.Width;
                    handLength = (int)arm.Size.Height;
                }

                armCenter.X = (int)Math.Round(arm.Center.X);
                armCenter.Y = (int)Math.Round(arm.Center.Y);
                Point[] armPts = { arm.Points()[0], arm.Points()[1], arm.Points()[2], arm.Points()[3] };
                Cv2.Polylines(myframe, new Point[][] { armPts }, true, armClr, 4, LineTypes.Link8);
            }

            mask1 = mask.Clone();

            return mask;
        }

        public void GetHull()
        {
            if (contour == null)
            {
                return;
            }

            // Get the convex hull of the entire hand/arm
            int[] indices = Cv2.ConvexHullIndices(contour, clockwise: true);
            List<Point> pts = new List<Point>(indices.Length);
            foreach (int i in indices)
            {
                pts.Add(contour[i]);
            }

            fingers = pts.ToArray();
            Cv2.Polylines(myframe, new Point[][] { fingers }, true, hullClr, 4, LineTypes.Link8);

            // Find the convexity defects that indicate the edges of the palm
            defects = Cv2.ConvexityDefects(contour, indices);
            defects = SimplifyDefects(defects);

            pts = new List<Point>();

            // use 1/4th of the hand width as discriminator
            double discriminator = handWidth / 4.0;
            foreach (Vec4i d in defects)
            {
                // (a.k.a.cv::Vec4i): (start_index, end_index, farthest_pt_index, fixpt_depth), 
                double depth = d.Item3 / 256.0;
                if (depth > discriminator)
                {
                    Point p = contour[d.Item2];
                    pts.Add(p);
                }
            }

            palm = pts.ToArray();
        }

        public int GetPalmCenter()
        {
            palmCenter = armCenter;
            palmRadius = handWidth / 3;

            if (palm != null)
            {
                Cv2.Circle(myframe, palmCenter, 5, palmClr, -1, LineTypes.AntiAlias);
                Cv2.Ellipse(myframe, palmCenter, new Size(palmRadius, palmRadius), 0, 0, 360, palmClr, 4, LineTypes.Link8);
                Cv2.Line(myframe, palmCenter, armCenter, palmClr, 4, LineTypes.Link8);

#if false
                // bounding box -- upright hand
                Point lowerLeft = new Point(palmCenter.X - palmRadius * 1.9, palmCenter.Y - palmRadius * 1.5);
                Point upperRight = new Point(palmCenter.X + palmRadius * 1.9, palmCenter.Y + palmRadius * 1.5);
#else
                // bounding box -- horizontal hand
                Point lowerLeft = new Point(palmCenter.X - palmRadius * palmRatio, palmCenter.Y - palmRadius * 1.9);
                Point upperRight = new Point(palmCenter.X + palmRadius * palmRatio, palmCenter.Y + palmRadius * 1.9);
#endif

                Cv2.Rectangle(myframe, lowerLeft, upperRight, palmClr, 4, LineTypes.Link8);

                // All defects within the bounding box are assumed to be palm extents.
                List<Point> fingerDefectList = new List<Point>();
                foreach (Point p in palm)
                {
                    if ((lowerLeft.X < p.X) && (upperRight.X > p.X) &&
                        (lowerLeft.Y < p.Y) && (upperRight.Y > p.Y))
                    {
                        fingerDefectList.Add(p);
                        Cv2.Circle(myframe, p, 5, palmClr, -1, LineTypes.AntiAlias);
                    }
                    else
                    {
                        Cv2.Circle(myframe, p, 5, red, -1, LineTypes.AntiAlias);
                    }
                }

                fingerdft = fingerDefectList.ToArray();
                return fingerdft.Length;
            }
            else
            {
                return 0;
            }
        }

        /// <summary>
        /// Get the cosine value of (b-palm) and (c-palm)
        /// = a dot b / |a|*|b|
        /// </summary>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <returns>cosine value</returns>
        double Cosine(Point a, Point b, Point c)
        {
            return Point.DotProduct(b - a, c - a) / (Point.Distance(b, a) * Point.Distance(c, a));
        }

        public int GetFingerTips()
        {
            if (palmRadius == 0)
            {
                return 0;
            }

            Point tmp;
            int fingerCount = 0;
            List<Point> possibleTips = new List<Point>(fingers.Length);

            // only interested in fingers to the left of the palm center
            foreach (Point p in fingers)
            {
                if (p.X < palmCenter.X)
                {
                    possibleTips.Add(p);
                }
            }

            // sort possible finger tips by vertical Y
            possibleTips.Sort(delegate (Point a, Point b)
            {
                return (int)a.Y - b.Y;
            });

            // use palm center offset 45 degrees up to the right as starting point
            tmp = palmCenter - new Point(-100, 100);

            foreach (Point p in possibleTips)
            {
                // if angle between last tip and current tip and palm center is > ~10 degrees
                // and tip is to the left of the palm center + palmRatio times palm radius
                double cos = Cosine(palmCenter, tmp, p);
                if ((cos < 0.985) && p.X < (palmCenter.X - palmRatio * palmRadius))
                {
                    fingerCount++;
                    tmp = p;
                    Cv2.Circle(myframe, p, 5, fingerTipClr, -1, LineTypes.AntiAlias);
                }
            }

            return fingerCount;
        }

        Vec4i[] SimplifyDefects(Vec4i[] defects)
        {
            int tolerance = handWidth / 10;
            double angleTol = Math.Cos(95 * Math.PI / 180);
            List<Vec4i> newDefects = new List<Vec4i>(defects.Count());

            foreach (Vec4i d in defects)
            {
                Point start = contour[d.Item0];
                Point end = contour[d.Item1];
                Point defect = contour[d.Item2];
                double d1 = Point.Distance(start, defect);
                double d2 = Point.Distance(end, defect);
                double cos = Cosine(defect, start, end);
                if ((d1 > tolerance) &&
                    (d2 > tolerance) &&
                    (cos > angleTol) &&
                    (defect.X < armCenter.X))  // to the left of arm center.
                {
                    newDefects.Add(d);
                    Cv2.Circle(myframe, defect, 5, green, -1, LineTypes.AntiAlias);
                }
                else
                {
                    Cv2.Circle(myframe, defect, 5, red, -1, LineTypes.AntiAlias);
                }
            }

            return newDefects.ToArray();
        }

        public HandResult Detect(int tips, int defects)
        {
            HandResult result = HandResult.None;

            if (contour == null)
            {
                return result;
            }

            if (tips >= 3 && defects >= 2)
            {
                result = HandResult.Paper;
            }
            else if (tips == 0)
            {
                result = HandResult.Rock;
            }
            else if (defects >= 1 && tips >= 2)
            {
                result = HandResult.Scissors;
            }

            return result;
        }
    }
}
