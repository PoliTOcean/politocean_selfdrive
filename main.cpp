#include "include/AutoDriveLib.h"
#include <opencv2/highgui/highgui.hpp>
int fps=100;


//For a video
int main()
{
    VideoCapture capture;
    AutoDrive_Data autodrive_data;
    AutoDriveLib auto_drive;

    //opening video
    autodrive_data.frame= capture.open("../VID_20200302_235949.mp4");
    

    //storing vedio
    Size size = Size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT));
    VideoWriter writer;
    writer.open("../result.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 1000/fps, size, true);



    while (capture.read(autodrive_data.frame))//can consider that in realtime everytime a new frame updated do the calculation once
    {

        auto_drive.renew_time(autodrive_data);//to get the real time gap of two frames
        auto_drive.Optical_Flow_Detection(autodrive_data);//use optical flow method to find and trace the feature points
        auto_drive.cal_pixel_speed(autodrive_data);//calculate the "pixel speed"
        //auto_drive.drive(middle,blue);//Control part, now assume the input is the x axis value of the middle point and blue borders
        
        imshow("output", autodrive_data.frame);
        waitKey(1000/fps);
        writer.write(autodrive_data.frame);
    }
    capture.release();
    return 0;
}

