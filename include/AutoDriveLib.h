/**************************************************************************

Author: Rex_YYJ

Date:2020-03-03

Description:Provide the data struct and functions to realize a auto-drive
            algorithm using OpenCV.

**************************************************************************/
#ifndef _AUTODRIVELIB_H
#define _AUTODRIVELIB_H

#define Resolution_x_Max 1920
#define Resolution_x_Min 0


#include<iostream>
#include<opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>   
#include <opencv2/xfeatures2d.hpp>
#include<opencv2/face.hpp>
#include<iostream>
#include<math.h>
#include <string> 
#include<fstream> 
#include <sys/time.h> 
#include <string>
#include <vector>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


struct AutoDrive_Data
{
    Mat frame, gray;            //current frame
    Mat prev_gray;              //previous frame
    vector<Point2f> features;   //the feature points form shi-tomasi angular point detection
    vector<Point2f>fpts[2];     //fpts[0]is the coordinates of feature points in current frame,fpts[1] for the previous frame  
    vector<Point2f> iniPoints;  //the tracking feature points,for drawing the moving tracks
    vector<uchar>status;        //the flag indicates that the feature points is correctly traced or not,特征点跟踪成功标志位
    vector<float>errors;        //the error sum in the region when tracking the feature points,跟踪时候区域误差和
    
    int tracking_status=0;      //a flag identifying there is a freature point detection in current frame or not
    long sys_time[2]={0,0};     //sys_time[0]is the arrive time of previous frame, sys_time[1] is for current frame
    vector<int> x_diff,y_diff;  //the movement of each tracked feature point
    float x_speed,y_speed;      //the "pixel speed", the unit is "unit/s"

    
    float x_power,y_power;      //final output
    
};



class AutoDriveLib
{

private:
    double maxCorners = 5000;
    double qualitylevel = 0.1;
    double minDistance = 10;
    double blockSize = 3;
    double k = 0.04;    //feature detection variables
    int min_points=20;//the minimum tracking points, if the tracking points less than this number, redetect
    
    int Filter_size=3;
    vector<float>filter_window_x={0,0,0};
    vector<float>filter_window_y={0,0,0};//Here and Filter_size should change at the same time, I don't know why I can't use it as a parameter

    int middle_coordinate=((Resolution_x_Max-Resolution_x_Min)/2);
    int blue_border[2]={200,1720};//blue_border[0]for left border, blue_border[1] for right border
    int red_border[2]={0,1920};//same as above
    float max_a=1;              //max acceleration in pixel domain,need more functions to determin

public:
    /*For finding the feature points in frame*/
    void detectFeatures(AutoDrive_Data &data)
    {
        goodFeaturesToTrack(data.gray, data.features, maxCorners, qualitylevel, minDistance, Mat(), blockSize, false, k);
        //The algorithm is fast enough to use in a realtime system
        cout << "detect features : " << data.features.size() << endl;
    }
    
    /*For drawing the feature points*/
    void drawFeature(AutoDrive_Data &data)
    {   
        for (size_t t = 0;t< data.fpts[0].size(); t++) {
            circle(data.frame, data.fpts[0][t], 2, Scalar(0, 0, 255), 2);
        }
    }  

    /*For drawing the track of the feature points*/
    void drawTrackLines(AutoDrive_Data &data)
    {
        for (size_t t = 0; t<data.fpts[1].size(); t++)
        {
            line(data.frame, data.iniPoints[t], data.fpts[1][t], Scalar(0, 255, 0), 1, 8, 0); // 绘制线段
            circle(data.frame, data.fpts[1][t], 2, Scalar(0, 0, 255), 2, 8, 0);
        }
    }
    /*Kanade-Lucas-Tomasi(KLT) Feature Tracker, for tracking the feature points*/
    void KLTrackFeature(AutoDrive_Data &data)
    {
        calcOpticalFlowPyrLK(data.prev_gray, data.gray, data.fpts[0], data.fpts[1], data.status, data.errors);
        int k = 0;//the number of the traced feature points
        data.x_diff.clear();
        data.y_diff.clear();
        for (int i = 0; i < data.fpts[1].size(); i++) {
            double dist = abs(data.fpts[0][i].x - data.fpts[1][i].x) + abs(data.fpts[0][i].y - data.fpts[1][i].y);
            if (data.status[i] && dist > 1)//Make sure the point is traced and its movement is larger than one coordinate
            {
                data.x_diff.push_back(data.fpts[0][i].x - data.fpts[1][i].x);
                data.y_diff.push_back(data.fpts[0][i].y - data.fpts[1][i].y);//save the pixel movement in x and y
                //将跟踪到的移动了的特征点在vector中连续起来，剔掉损失的和禁止不动的特征点（这些跟踪点在前面帧中）
                //Connect the traced moving feature points in feature, discad the lost or the motionless feature points
                //(these feature points are in the previous frame)
                data.iniPoints[k] = data.iniPoints[i];
                //同上（只是这些跟踪点在当前帧中）
                //Same as above, but the feature points are in the current frame
                data.fpts[1][k++] = data.fpts[1][i];
            }
        }
        //保存特征点并绘制跟踪轨迹
        //Save the feature points
        data.iniPoints.resize(k);
        data.fpts[1].resize(k);
        //交换，将此帧跟踪到特征点作为下一帧的待跟踪点
        //swap the feature points in the current frame with the ones in the previous frame
        //making the feature points in current frame be the fracking point in the next frame
        std::swap(data.fpts[1], data.fpts[0]);
    }

    /*The top level of the Optical flow deteciton, call this function when use*/
    void Optical_Flow_Detection(AutoDrive_Data &data)
    {
        cvtColor(data.frame, data.gray, COLOR_BGR2GRAY);
        if (data.fpts[0].size() < min_points) //checking the tracking feature points is more than the minimum number or not
        {
            AutoDriveLib::detectFeatures(data);
            //adding the tracking feature points
            data.fpts[0].insert(data.fpts[0].end(), data.features.begin(), data.features.end());//追加带跟踪的特征点
            data.iniPoints.insert(data.iniPoints.end(), data.features.begin(), data.features.end());
            data.tracking_status=0;
        }
        else
        {
            data.tracking_status=1;
            cout << "tracjing........" << endl;
        }
        if (data.prev_gray.empty())
            data.gray.copyTo(data.prev_gray);//save the first frame
        AutoDriveLib::KLTrackFeature(data);//KLT
        AutoDriveLib::drawFeature(data);//draw the feature points
        //renew the previous frame
        data.gray.copyTo(data.prev_gray);
        printf("finish opticlal flow\n");
    }

    /* Function to get the system time, return type is long, unit is ms*/
    long getCurrentTime()  
    {  
        struct timeval tv;  
        gettimeofday(&tv,NULL);  
        return tv.tv_sec * 1000 + tv.tv_usec / 1000;  
    } 
    
    /*Function for recording the arrive time for current and previous frame*/
    void renew_time(AutoDrive_Data &data)
    {
        if (data.sys_time[0]==0)
        {
            data.sys_time[0]=getCurrentTime();
            return ;
        }
        else if(data.sys_time[1]==0)
        {
            data.sys_time[1]=getCurrentTime();
            return ;
        }
        else
        {
            data.sys_time[0]=data.sys_time[1];
            data.sys_time[1]=getCurrentTime();
            printf("time_diff=%ld",(data.sys_time[1]-data.sys_time[0]));
        }
    }

    /*Function to fine the major element(mode) in the vector.*/
    template<typename T> T majorElemCandidate( vector<T> A ){
        T maj;
        int count=0;
        
        for(int i=0;i<A.size();++i){
            if(count == 0){
                maj = A[i];
                count++;
            } else {
                maj == A[i]? count++ : count--;
            }
        }
        
        return maj; //假如众数存在,则为maj(但也有可能不存在)
    }

    /*Draw the speed in frame.*/
    void output_data_in_frame(AutoDrive_Data &data)
    {
        string x_out;
        string y_out;
        x_out="x_speed= ";
        x_out+=std::to_string(data.x_speed);
        x_out+=" units/s";
        y_out="y_speed= ";
        y_out+=std::to_string(data.y_speed);
        y_out+=" uints/s";
        cv::putText(data.frame,x_out,Point(50,60),FONT_HERSHEY_SIMPLEX,1,Scalar(255,23,0),2,4);
        cv::putText(data.frame,y_out,Point(50,80),FONT_HERSHEY_SIMPLEX,1,Scalar(0,23,255),2,4);
    }
    /*Median filter to make the speed smoother*/
    float median_filter_x(float new_data)
    {
        
        for (int i=0;i<Filter_size-1;i++)
        {
            filter_window_x[i]=filter_window_x[i+1];
        }
        filter_window_x[Filter_size-1]=new_data;

        vector<float>temp(filter_window_x);
        sort(temp.begin(),temp.end());
        return temp[(int)(Filter_size/2)];
        
    }
    float median_filter_y(float new_data)
    {
        
        for (int i=0;i<Filter_size-1;i++)
        {
            filter_window_y[i]=filter_window_y[i+1];
        }
        filter_window_y[Filter_size-1]=new_data;

        vector<float>temp(filter_window_y);
        sort(temp.begin(),temp.end());
        return temp[(int)(Filter_size/2)];
        
    }
    /*Calculate the pixel speed and output in frame.
      Only when there is ONLY tracking but NO deteciton do the calculation,
      avoiding a vibration due to the new points*/
    void cal_pixel_speed(AutoDrive_Data &data)
    {
        if (data.tracking_status==1&&(data.sys_time[0]!=0&&data.sys_time[1]!=0))
        {

            int x_diff=(int)majorElemCandidate(data.x_diff);
            int y_diff=(int)majorElemCandidate(data.y_diff);
            long time_diff=data.sys_time[1]-data.sys_time[0];

            data.x_speed=median_filter_x(x_diff*1000/((int)time_diff));
            data.y_speed=median_filter_y(y_diff*1000/((int)time_diff));

            output_data_in_frame(data);
        }
    }
    /*Nenew the coordinates of borders*/
    void renew_coordinates(int middle,int blue[])
    {
        middle_coordinate=middle;
        blue_border[0]=blue[0];
        blue_border[1]=blue[1];
        red_border[0]=2*blue_border[0]-middle_coordinate;
        red_border[1]=2*blue_border[1]-middle_coordinate;
    }


    //Acutal control part, Require more variables

    // /*Predict can the ROV stop before reaching the borders under current x speed, and return different flag for differnt actions*/
    // int safe_zone_detect(AutoDrive_Data &data)
    // {
    //     float speed_cp=fabs(data.x_speed);
    //     float t_stop=speed_cp/max_a;
    //     float s_stop=speed_cp*t_stop+0.5*max_a*t_stop*t_stop;
        
    //     float s_trigger=1.2*s_stop;

    //     if (data.x_speed<0)
    //     {
    //         if ((Resolution_x_Min-red_border[0]<s_trigger)||(Resolution_x_Max-blue_border[1]<s_trigger))
    //         {
    //             printf("Danger of Red_left border or Blue_right border,full power to right");
    //             return 1;
    //         }
    //         else 
    //         {
    //             return 0;
    //         }
    //     }
    //     else if(data.x_speed>0)
    //     {
    //         if((red_border[1]-Resolution_x_Max<s_trigger)||(blue_border[0]-Resolution_x_Min<s_trigger))
    //         {
    //             printf("Danger of Blue_left border or Red_right border,full power to left");
    //             return -1;
    //         }
    //         else
    //         {
    //             return 0;
    //         }
    //     }
    //     else
    //     {
    //         return 0;
    //     }
        
    // }
    // /*Calculate the power to sent to the ROV*/
    // void drive(AutoDrive_Data &data,int middle,int blue[])
    // {
    //     renew_coordinates(middle,blue);
    //     int mode_flat=safe_zone_detect(data);

    //     if(middle>Resolution_x_Min&&middle<Resolution_x_MAX)
    //     {
    //         data.y_power=pid_controller(40,data);//make ROV moving forward in a certain speed
    //     }

    //     switch (mode_flag)//according to the flag value changing the driving mode
    //     {
    //     case 0:
    //         target_speed=pid_controller(middle,data);
    //         data.x_power=pid_controller(traget_speed,data);
    //         break;
    //     case 1:
    //         data.x_power=100;//act as an emergency breaker
    //         break;
    //     case -1:
    //         data.x_power=-100;//same as above
    //         break;
    //     default:
    //         break;
    //     }

    // }
 


};






#endif