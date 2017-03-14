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

    class HandDetect
    {
        private Rect[] faces;
        private Scalar palmClr = new Scalar(255, 255, 255, 255);
        private Scalar faceClr = new Scalar(255, 0, 255, 255);
        private Scalar fingerClr = new Scalar(0, 0, 0, 255);
        private Scalar hullClr = new Scalar(0, 255, 0, 255);

        Point[] contour = null;
        Point[] fingers = null;
        Point[] palm = null;
        Vec4i[] defects = null;

        public Mat myframe;
        public Mat mask1;
        public Mat mask2;
        public Mat mask3;
        public Mat mask4;
        public Mat mybackground = null;

        public HandDetect(Mat frame)
        {
            myframe = frame;
        }

        private HandDetect()
        {
        }

        public Rect? FaceDetect(Mat frame, CascadeClassifier faceClassifier)
        {
            Mat frame_gray;
            Rect? p = null;

            frame_gray = frame.CvtColor(ColorConversionCodes.BGRA2GRAY).EqualizeHist();

            faces = faceClassifier.DetectMultiScale(frame_gray, 1.1, 2, HaarDetectionType.ScaleImage, new Size(30, 30));
            foreach (Rect face in faces)
            {
                Point center = new Point(face.X + face.Width * 0.5, face.Y + face.Height * 0.5);

                if (!p.HasValue || (face.Area() > p.Value.Area()))
                {
                    p = face;
                }

                myframe.Ellipse(center, new Size(face.Width * 0.5, face.Height * 0.5), 0, 0, 360, faceClr, 4, LineTypes.Link8, 0);
            }
            return p;
        }

        public void SkinColorModel(Mat frame, Rect faceregion, out Vec3i maxYCrCb, out Vec3i minYCrCb)
        {
            Mat p = frame.CvtColor(ColorConversionCodes.BGR2YCrCb);

            maxYCrCb = new Vec3i(295, -1, -1);
            minYCrCb = new Vec3i(-1, 295, 295);

            if (faceregion.Area() > 5)
            {
                for (int i = faceregion.X; (i < (faceregion.X + faceregion.Width)) && (i < frame.Cols); i++)
                {
                    for (int j = faceregion.Y; (j < (faceregion.Y + faceregion.Height)) && (j < frame.Rows); j++)
                    {
                        int r, b, g;
                        int y, cb, cr;
                        Vec4b bgr = frame.At<Vec4b>(j, i);
                        Vec4b yCrCb = p.At<Vec4b>(j, i);

                        r = bgr.Item2;
                        b = bgr.Item0;
                        g = bgr.Item1;

                        y = yCrCb.Item0;
                        cr = yCrCb.Item1;
                        cb = yCrCb.Item2;

                        int gray = (int)(0.2989 * r + 0.5870 * g + 0.1140 * b);
                        if (gray < 200 && gray > 40 && b > g && r > b)
                        {
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
                maxYCrCb = new Vec3i(255, 173, 127);
                minYCrCb = new Vec3i(0, 133, 77);
            }
        }

        public Mat HandDetection(Mat frame, Rect faceRegion, Vec3i maxYCrCb, Vec3i minYCrCb)
        {
            Size size = frame.Size();

            Mat mask = Mat.Zeros(size, MatType.CV_8UC1);

            if (faceRegion.Area() > 5)
            {
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
            }

            // Turn to YCrCb
            int y, cr, cb;
            Mat p, b;

            p = frame.CvtColor(ColorConversionCodes.BGR2YCrCb);
            mask1 = p.Clone();

            for (int i = 0; i < frame.Cols; i++)
            {
                for (int j = 0; j < frame.Rows; j++)
                {
                    y = p.At<Vec4b>(j, i)[0];
                    cr = p.At<Vec4b>(j, i)[1];
                    cb = p.At<Vec4b>(j, i)[2];
                    if (y > minYCrCb.Item0 && y < maxYCrCb.Item0 &&
                        cr > minYCrCb.Item1 && cr < maxYCrCb.Item1 &&
                        cb > minYCrCb.Item2 && cb < maxYCrCb.Item2)
                    {
                        mask.Set<byte>(j, i, 255);
                    }

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

            //mask1 = mask.Clone();

            foreach (Rect face in faces)
            {
                Cv2.Rectangle(mask, face, new Scalar(0), -1);    // filled rectangle
            }

            Cv2.Erode(mask, mask, null, null, 2);

            mask2 = mask.Clone();

            Cv2.Dilate(mask, mask, null, null, 1);

            mask3 = mask.Clone();

            Point[][] contours;
            HierarchyIndex[] hierarchy;

            Cv2.FindContours(mask, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxNone);

            int max_contour_size = -1;

            // find longest contour
            foreach (Point[] points in contours)
            {
                if (points.Length >= 370 && points.Length > max_contour_size)
                {
                    contour = points;
                    max_contour_size = contour.Length;
                }
            }

            Point armcenter, palm_center;
            if (contour != null)
            {
                Cv2.DrawContours(mask, new Point[][] { contour }, 0, new Scalar(100), thickness: 4, maxLevel: 2, lineType: LineTypes.AntiAlias);

                RotatedRect center = Cv2.MinAreaRect(contour);
                armcenter.X = (int)Math.Round(center.Center.X);
                armcenter.Y = (int)Math.Round(center.Center.Y);
                Cv2.Circle(myframe, armcenter, 10, palmClr, -1, LineTypes.Link8, 0);

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

                Get_hull();
            }
            mask4 = mask.Clone();

            return mask;
        }

        public void Get_hull()
        {
            // Would like to use Cv2.ConvexHull() that returns an array of Points, but ConvexityDefects
            // only takes a list of indices.
//            fingers = Cv2.ConvexHull(contour, clockwise: true);

            int[] indices = Cv2.ConvexHullIndices(contour, clockwise: true);
            List<Point> pts = new List<Point>(indices.Length);
            foreach (int i in indices)
            {
                pts.Add(contour[i]);
            }

            fingers = pts.ToArray();
            Cv2.Polylines(myframe, new Point[][] { fingers }, true, hullClr, 4, LineTypes.Link8);

            defects = Cv2.ConvexityDefects(contour, indices);
            pts = new List<Point>();

            foreach (Vec4i d in defects)
            {
                // (a.k.a.cv::Vec4i): (start_index, end_index, farthest_pt_index, fixpt_depth), 
                if ((d.Item3 / 256) > 10) // depth
                {
                    Point p = contour[d.Item2];
                    Cv2.Circle(myframe, p, 5, palmClr, -1, LineTypes.AntiAlias);
                    pts.Add(p);
                }
            }
            palm = pts.ToArray();
        }

        public void GetPalmCenter()
        {

        }

        public void GetFingerTips()
        {

        }




    }
}
    