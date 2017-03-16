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
        private Rect[] faces;
        
        private Scalar cyan = new Scalar(255, 255, 0, 255); // cyan
        private Scalar white = new Scalar(255, 255, 255, 255); // white
        private Scalar magenta = new Scalar(255, 0, 255, 255); // magenta
        private Scalar black = new Scalar(0, 0, 0, 255); // black
        private Scalar yellow = new Scalar(0, 255, 255, 255); // yellow
        private Scalar green = new Scalar(0, 255, 0, 255); // green

        private Scalar palmClr;
        private Scalar faceClr;
        private Scalar armClr;
        private Scalar hullClr;
        private Scalar fingerTipClr;

        Point[] contour = null;
        Point[] fingers = null;
        Point[] fingerdft = null;
        Point[] palm = null;
        Vec4i[] defects = null;

        RotatedRect arm;
        Point armCenter;
        Point palmCenter;
        double aspectRatio;
        int palmRadius;
        
        public Mat myframe;
        public Mat mask1;
        public Mat mask2;
        public Mat mask3;
        public Mat mask4;
        public Mat mybackground = null;

        public HandDetect(Mat frame)
        {
            myframe = frame.Clone();

            palmClr = cyan;
            faceClr = magenta;
            fingerTipClr = black;
            armClr = yellow;
            hullClr = magenta;
        }

        private HandDetect()
        {
        }

        public Rect FaceDetect(Mat frame, CascadeClassifier faceClassifier)
        {
            Mat grayscale;
            Rect p = new Rect();

            grayscale = frame.CvtColor(ColorConversionCodes.BGRA2GRAY).EqualizeHist();

            faces = faceClassifier.DetectMultiScale(grayscale, 1.1, 2, HaarDetectionType.ScaleImage, new Size(30, 30));
            foreach (Rect face in faces)
            {
                // highlight each face
                Point center = new Point(face.X + face.Width * 0.5, face.Y + face.Height * 0.5);
                myframe.Ellipse(center, new Size(face.Width * 0.5, face.Height * 0.5), 0, 0, 360, faceClr, 4, LineTypes.Link8, 0);

                // choose the largest
                if (face.Area() > p.Area())
                {
                    p = face;
                }
            }

            return p;
        }

        public void SkinColorModel(Mat frame, Rect faceregion, out Vec3b maxYCrCb, out Vec3b minYCrCb)
        {
            Mat roiMap = Mat.Zeros(frame.Size(), MatType.CV_8UC1);
            Mat bgrMap = frame.CvtColor(ColorConversionCodes.BGRA2BGR);
            Mat lumMap = bgrMap.CvtColor(ColorConversionCodes.BGR2YCrCb);

            maxYCrCb = new Vec3b(255, 0, 0);
            minYCrCb = new Vec3b(0, 255, 255);

            if (faceregion.Area() > 5)
            {
                // foreach pixel, if the gray scale value is between 40 and 200
                // and blue > green and red > blue, find the max and min of the chroma channels.
                for (int i = faceregion.X; (i < (faceregion.X + faceregion.Width)) && (i < frame.Cols); i++)
                {
                    for (int j = faceregion.Y; (j < (faceregion.Y + faceregion.Height)) && (j < frame.Rows); j++)
                    {
                        int r, b, g;
                        int y, cb, cr;
                        Vec3b bgr = bgrMap.At<Vec3b>(j, i);
                        Vec3b yCrCb = lumMap.At<Vec3b>(j, i);

                        r = bgr.Item2;
                        b = bgr.Item0;
                        g = bgr.Item1;

                        y = yCrCb.Item0;
                        cr = yCrCb.Item1;
                        cb = yCrCb.Item2;

                        int gray = (int)(0.299 * r + 0.587 * g + 0.114 * b);

#if true
                        if ((gray < 200 /* 200 */) && (gray > 40 /* 40 */)
                            && (b > g - 10) && (r > b)) // (b > g) && (r > b))
                        {
#else
                        if ((r > 95) && (g > 40) && (b > 20) &&
                            ((Math.Max(Math.Max(r, g), b) - Math.Min(Math.Min(r, g), b)) > 15) &&
                            (Math.Abs(r - g) > 15) && (r > g) && (r > b))
                        { 
#endif
                        roiMap.Set<byte>(j, i, 255);

                            maxYCrCb.Item0 = Math.Max(maxYCrCb.Item0, yCrCb.Item0);
                            maxYCrCb.Item1 = Math.Max(maxYCrCb.Item1, yCrCb.Item1);
                            maxYCrCb.Item2 = Math.Max(maxYCrCb.Item2, yCrCb.Item2);

                            minYCrCb.Item0 = Math.Min(minYCrCb.Item0, yCrCb.Item0);
                            minYCrCb.Item1 = Math.Min(minYCrCb.Item1, yCrCb.Item1);
                            minYCrCb.Item2 = Math.Min(minYCrCb.Item2, yCrCb.Item2);
                        }
                    }
                }
            }
            else
            {
                maxYCrCb = new Vec3b(255, 173, 127);
                minYCrCb = new Vec3b(0, 133, 77);
            }

            //Cv2.InRange(lumMap, (Scalar)minYCrCb, (Scalar)maxYCrCb, roiMap);
            mask1 = roiMap.Clone();
        }

        public Mat HandDetection(Mat frame, Rect faceRegion, Vec3b maxYCrCb, Vec3b minYCrCb)
        {
            // create a single channel mask
            Mat mask = Mat.Zeros(frame.Size(), MatType.CV_8UC1);

            if (faceRegion.Area() > 5)
            {
                Cv2.Rectangle(myframe, faceRegion, faceClr, 4, LineTypes.Link8);

                // Drop lower edge of face region by 25% or to lower edge of frame
                if (faceRegion.Y > faceRegion.Height / 4)
                {
                    faceRegion.Y -= faceRegion.Height / 4;
                    faceRegion.Height += faceRegion.Height / 4;
                }
                else
                {
                    faceRegion.Height += faceRegion.Y;
                    faceRegion.Y = 0;
                }
                // avoid noise for T-shirt
                faceRegion.Height += faceRegion.Height / 2;

                Cv2.Rectangle(myframe, faceRegion, faceClr, 4, LineTypes.Link8);
            }

            // Convert to YCrCb
            int y, cr, cb;
            Mat p, b;

            p = frame.CvtColor(ColorConversionCodes.BGR2YCrCb);
//            mask1 = p.Clone();

            for (int i = 0; i < frame.Cols; i++)
            {
                for (int j = 0; j < frame.Rows; j++)
                {
                    // foreach pixel, if the chroma channels are within the min and max, set the mask value to 255
                    y = p.At<Vec4b>(j, i)[0];
                    cr = p.At<Vec4b>(j, i)[1];
                    cb = p.At<Vec4b>(j, i)[2];
                    if (y > minYCrCb.Item0 && y < maxYCrCb.Item0 &&
                        cr > minYCrCb.Item1 && cr < maxYCrCb.Item1 &&
                        cb > minYCrCb.Item2 && cb < maxYCrCb.Item2)
                    {
                        mask.Set<byte>(j, i, 255);
                    }

                    // subtract known background elements if we have them.
                    if (mybackground != null)
                    {
                        b = mybackground;
                        if (Math.Abs((int)frame.At<Vec3b>(j, i)[0] - (int)b.At<Vec3b>(j, i)[0]) < 10 &&
                            Math.Abs((int)frame.At<Vec3b>(j, i)[1] - (int)b.At<Vec3b>(j, i)[1]) < 10 &&
                            Math.Abs((int)frame.At<Vec3b>(j, i)[2] - (int)b.At<Vec3b>(j, i)[2]) < 10)
                        {
                            mask.Set<byte>(j, i, 0);
                        }
                    }
                }
            }

            mask2 = mask.Clone();

            // subtract the suspected face regions from the mask
            foreach (Rect face in faces)
            {
                Cv2.Rectangle(mask, face, new Scalar(0), -1); // filled rectangle
            }

            // subtrace the enlarged face region
            Cv2.Rectangle(mask, faceRegion, new Scalar(0), -1); // filled rectangle

            // Get rid of single pixel noise from the mask
            Cv2.Erode(mask, mask, null, null, 2);

            // enlarge the areas within the mask to attempt to join them
            Mat element = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(10,10));
            Cv2.Dilate(mask, mask, element, null, 1);

            mask3 = mask.Clone();

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
                }
                else
                {
                    aspectRatio = arm.Size.Height / arm.Size.Width;
                }

                armCenter.X = (int)Math.Round(arm.Center.X);
                armCenter.Y = (int)Math.Round(arm.Center.Y);
                Point[] armPts = { arm.Points()[0], arm.Points()[1], arm.Points()[2], arm.Points()[3] };
                Cv2.Polylines(myframe, new Point[][] { armPts }, true, armClr, 4, LineTypes.Link8);

                // find palm center
/*            
                double dist,maxdist=-1; 
                Point center; 
                vector<Point2f> cont_seq; 
                for (int i = 0; i < max_contour_size; i++){ 
                    int* point = (int*)cvGetSeqElem(maxrecord, i); 
                    cont_seq.insert(cont_seq.end(), Point2f(point[0], point[1])); 
                } 
                for (int i = 0; i< frame.cols; i++) 
                { 
                    for (int j = 0; j< frame.rows; j++) 
                    { 
                        dist = pointPolygonTest(cont_seq, cv::Point2f(i, j), true); 
                        if (dist > maxdist) 
                        { 
                            maxdist = dist; 
                            center = cv::Point(i, j); 
                        } 
                    } 
                } 
                cvCircle(&myframe_ipl, center, 10, CV_RGB(255, 0,0), -1, 8, 0);
*/
            }
            mask4 = mask.Clone();

            return mask;
        }

        public void GetHull()
        {
            if (contour == null)
            {
                return;
            }

            // Would like to use Cv2.ConvexHull() that returns an array of Points, but ConvexityDefects
            // only takes a list of indices.
//            fingers = Cv2.ConvexHull(contour, clockwise: true);

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
            pts = new List<Point>();

            // use 1/15th of average of width and height of hand/wrist as discriminator
            double discriminator = Math.Max(10.0, (arm.Size.Width + arm.Size.Height) / 30.0);
            foreach (Vec4i d in defects)
            {
                // (a.k.a.cv::Vec4i): (start_index, end_index, farthest_pt_index, fixpt_depth), 
                double depth = d.Item3 / 256.0;
                if (depth > discriminator)
                {
                    Point p = contour[d.Item2];
                    Cv2.Circle(myframe, p, 5, palmClr, -1, LineTypes.AntiAlias);
                    pts.Add(p);
                }
            }
            palm = pts.ToArray();
            Cv2.Polylines(myframe, new Point[][] { palm }, true, hullClr, 4, LineTypes.Link8);
        }

        public int GetPalmCenter()
        {
            palmCenter = armCenter;

            if (palm != null && palm.Length > 0)
            {
                // find the average of the palm points to find palm center
                palmCenter.X = 0;
                palmCenter.Y = 0;
                foreach (Point p in palm)
                {
                    palmCenter += p;
                }
                palmCenter *= 1.0 / palm.Length;

                palmRadius = 0;

                // find the average radius of the palm from the center
                foreach (Point p in palm)
                {
                    palmRadius += (int)Point.Distance(p, palmCenter);
                }
                palmRadius /= palm.Length;

                // if the palm is higher than the arm center use the arm center
/*                if (palmCenter.Y > armCenter.Y)
                {
                    palmCenter = armCenter;
                }
*/

                if (palm.Length < 3)
                {
                    palmRadius = 0;
                }

                Cv2.Circle(myframe, palmCenter, 5, palmClr, -1, LineTypes.AntiAlias);
                Cv2.Ellipse(myframe, palmCenter, new Size(palmRadius, palmRadius), 0, 0, 360, palmClr, 4, LineTypes.Link8);
                Cv2.Line(myframe, palmCenter, armCenter, palmClr, 4, LineTypes.Link8);

#if false
                // bounding box -- upright hand
                Point lowerLeft = new Point(palmCenter.X - palmRadius * 1.9, palmCenter.Y - palmRadius * 1.5);
                Point upperRight = new Point(palmCenter.X + palmRadius * 1.9, palmCenter.Y + palmRadius * 1.5);
#else
                // bounding box -- upright hand
                Point lowerLeft = new Point(palmCenter.X - palmRadius * 1.5, palmCenter.Y - palmRadius * 1.9);
                Point upperRight = new Point(palmCenter.X + palmRadius * 1.5, palmCenter.Y + palmRadius * 1.9);
#endif

                Cv2.Rectangle(myframe, lowerLeft, upperRight, palmClr, 4, LineTypes.Link8);

                List<Point>fingerList = new List<Point>();
                foreach (Point p in palm)
                {
                    if ((lowerLeft.X < p.X) && (upperRight.X > p.X) &&
                        (lowerLeft.Y < p.Y) && (upperRight.Y > p.Y))
                    {
                        fingerList.Add(p);
                        Cv2.Circle(myframe, p, 5, fingerTipClr, -1, LineTypes.AntiAlias);
                    }
                }

                fingerdft = fingerList.ToArray();
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
            List<Point> gaps = new List<Point>(fingerdft.Length + 3);
            List<Point> possibleTips = new List<Point>(fingers.Length + 3);
            List<Point> checkedTips = new List<Point>();

            Point tmp;
            int fingerCount = 0;

            if (palmRadius == 0)
            {
                return 0;
            }

#if false

            gaps.AddRange(fingerdft);
            gaps.Add(new Point(-1, 0)); // lower bound
            gaps.Add(new Point(30000, 0));
            gaps.Add(new Point(30001, 0)); // upper bound

            gaps.Sort(delegate (Point a, Point b)
            {
                return (int)a.X - b.X;
            });

            possibleTips.AddRange(fingers);
            possibleTips.Sort(delegate (Point a, Point b)
            {
                return (int)a.X - b.X;
            });

            tmp = palmCenter;
            tmp.Y = 999;

            foreach (Point p in possibleTips)
            {
                if ((Cosine(tmp, p) < 0.98) && 
                    (p.Y < palmCenter.Y + 0.8 * palmRadius) &&
                    (Point.Distance(palmCenter, p) > palmRadius * 1.87))
                {
                    fingerCount ++;
                    tmp = p;
                    Cv2.Circle(myframe, p, 5, fingerTipClr, -1, LineTypes.AntiAlias);
                }
            }
#else
            // Right hand sideways
            gaps.AddRange(fingerdft);
            gaps.Add(new Point(0, -1)); // lower bound
            gaps.Add(new Point(0, 30000));
            gaps.Add(new Point(0, 30001)); // upper bound

            gaps.Sort(delegate (Point a, Point b)
            {
                return (int)a.Y - b.Y;
            });

            possibleTips.AddRange(fingers);
            possibleTips.Sort(delegate (Point a, Point b)
            {
                return (int)a.Y - b.Y;
            });

            tmp = palmCenter;
            tmp.X = -999;

            foreach (Point p in possibleTips)
            {
                double cos = Cosine(palmCenter, tmp, p);
                if ((cos < 0.98) &&
                    (p.X < palmCenter.X + 0.8 * palmRadius) &&
                    (Point.Distance(palmCenter, p) > palmRadius * 1.87))
                {
                    fingerCount++;
                    tmp = p;
                    Cv2.Circle(myframe, p, 5, fingerTipClr, -1, LineTypes.AntiAlias);
                }
            }
#endif

            /*
413 	char tmp[30]; 
414 	_itoa_s(cnt_finger, tmp, 10); 
415 	CvFont Font1 = cvFont(3, 3); 
416 	cvPutText(&frame, tmp, cvPoint(10, 50), &Font1, CV_RGB(255, 0, 0)); 
*/
            return fingerCount;
        }

        public HandResult Detect(int tips, int defects)
        {
            HandResult result = HandResult.None;

            if (contour == null)
            {
                return result;
            }

            if (tips >= 4 && defects >= 3 && tips <= 6 && defects < 5)
            {
                result = HandResult.Paper;
            }
            else if (tips == 0 && defects >= 2 && defects <= 5 && aspectRatio > 1.7)
            {
                result = HandResult.Paper;  // open palm with all fingers together
            }
            else if (tips == 0 && aspectRatio > 1.7)
            {
                result = HandResult.Paper;
            }
            else if (tips == 0)
            {
                result = HandResult.Rock;
            }
            else if (tips >= 1 && tips <= 2 && defects >= 2 && defects <= 4)
            {
                result = HandResult.Scissors;
            }
            else if (tips == 3 && defects >= 2 && defects <= 3)
            {
                result = HandResult.Scissors;
            }
            else if (tips == 3 && defects >= 4 && defects <= 5)
            {
                result = HandResult.Paper;
            }
            return result;
        }
    }
}
    