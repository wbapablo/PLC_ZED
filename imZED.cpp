//============================================================================
// Name        : imZED.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <ctime>
#include <chrono>
#include <cmath>

#include <thread>
#include <mutex>

#include <opencv2/opencv.hpp>

#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <vector>
#include <iomanip>
#include <stddef.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>




#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>


#include <sstream>

#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common_headers.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/cloud_viewer.h>

//New includes



using namespace std;
using namespace cv;
using namespace sl::zed;

void planextraction(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,pcl::PointCloud<pcl::PointXYZRGB>::Ptr& result){
	//create cloud data

	  pcl::PointIndices::Ptr inliers_ransac_b (new pcl::PointIndices);
	  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZRGB>);

	  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
	  // Optional

	  seg.setOptimizeCoefficients (true);
	  // Mandatory
	  seg.setModelType (pcl::SACMODEL_PLANE);
	  seg.setMethodType (pcl::SAC_RANSAC);
	  seg.setMaxIterations(10);
	  seg.setDistanceThreshold (0.3);//0.06 //3.58

	  // Create the filtering object
	  pcl::ExtractIndices<pcl::PointXYZRGB> extract;

	  seg.setInputCloud (cloud);
	  seg.segment (*inliers_ransac_b, *coefficients);

	  extract.setInputCloud (cloud);
	  extract.setIndices (inliers_ransac_b);
	  extract.setNegative (false);
	  extract.filter (*result);

	  extract.setNegative (true);
	  extract.filter (*cloud_f);
	  cloud.swap (cloud_f);
}


void out_data(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){
	int aux=0;
	int sz=cloud->points.size();
	for(int i=0;i<cloud->points.size();i++){
		aux+=cloud->points[i].z;
		}
	cout<<"depth mean= "<<aux/sz<<endl;

	cv::Mat data_pts = cv::Mat(sz,2,CV_64FC1);

	for (int j=0;j<data_pts.rows;j++){
		data_pts.at<double>(j,0)=cloud->points[j].x;
		data_pts.at<double>(j,1)=cloud->points[j].y;
		}
	PCA pca_analysis(data_pts,cv::Mat(),CV_PCA_DATA_AS_ROW);
	Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0,0)),
					   static_cast<int>(pca_analysis.mean.at<double>(0,1)));

	vector<Point2d> eigen_vecs(2);
	vector<double> eigen_val(2);

	for(int k=0;k<2;k++){
		eigen_vecs[k]=Point2d(pca_analysis.eigenvectors.at<double>(k,0),
							  pca_analysis.eigenvectors.at<double>(k,1));

		eigen_val[k]= pca_analysis.eigenvalues.at<double>(0,k);

		}

	cout<<"eigenvectors= "<<eigen_vecs<<endl;
	printf("eigenvalues= %f,%f \n",eigen_val[0],eigen_val[1]);


	eigen_vecs.clear();
	eigen_val.clear();
	}


typedef struct image_bufferStruct {
    float* data_cloud;
    std::mutex mutex_input;
    int width, height;
} image_buffer;

Camera* zed;
image_buffer* buffer;
SENSING_MODE dm_type = STANDARD;
bool stop_signal;

// Grab called in a thread to parallelize the rendering and the computation

void grab_run() {
    float* p_cloud;

    while (!stop_signal) {
        if (!zed->grab(dm_type)) {
            p_cloud = (float*) zed->retrieveMeasure(MEASURE::XYZRGBA).data; // Get the pointer
            // Fill the buffer
            buffer->mutex_input.lock(); // To prevent from data corruption
            memcpy(buffer->data_cloud, p_cloud, buffer->width * buffer->height * sizeof (float) * 4);
            buffer->mutex_input.unlock();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

std::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud) {
    // Open 3D viewer and add point cloud
    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("PCL ZED 3D Viewer"));
    viewer->setBackgroundColor(0.12, 0.12, 0.12);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    return (viewer);
}

int main(int argc, char **argv) {
	stop_signal = false;

    if (argc > 2) {
        std::cout << "Only the path of a SVO can be passed in arg" << std::endl;
        return -1;
    }
    if (argc == 1) // Live Mode
        zed = new Camera(VGA);
    else // SVO playback mode
        zed = new Camera(argv[1]);
    sl::zed::InitParams params;
    params.mode = PERFORMANCE;
    params.unit = METER; // Scale to fit OpenGL world
    params.coordinate = RIGHT_HANDED; // OpenGL compatible
    params.verbose = true;

    char key = ' ';
    char ind = '1';



    cv::namedWindow("VIEW", cv::WINDOW_AUTOSIZE);

    ERRCODE err = zed->init(params);
    cout << errcode2str(err) << endl;
    if (err != SUCCESS) {
        delete zed;
        return 1;
    }

    int width = zed->getImageSize().width;
    int height = zed->getImageSize().height;
	    // Allocate data
    buffer = new image_buffer();
    buffer->height = height;
    buffer->width = width;
    buffer->data_cloud = new float[buffer->height * buffer->width * 4];

    int size = height*width;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer = rgbVis(point_cloud_ptr);





    float* data_cloud;
	    // Run thread
    std::thread grab_thread(grab_run);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    float color;
    int index4 = 0;

    point_cloud_ptr->points.resize(size);

    while (!viewer->wasStopped()) {

    	pcl::PointCloud<pcl::PointXYZRGB>::Ptr result (new pcl::PointCloud<pcl::PointXYZRGB>);
    	pcl::PointCloud<pcl::PointXYZRGB>::Ptr result1 (new pcl::PointCloud<pcl::PointXYZRGB>);

    	pcl::PointCloud<pcl::PointXYZRGB>::Ptr nube (new pcl::PointCloud<pcl::PointXYZRGB>);

    	if (buffer->mutex_input.try_lock()) {

            data_cloud = buffer->data_cloud;
	        index4 = 0;

            for (auto &it : point_cloud_ptr->points) {
                float X = data_cloud[index4 * 4];
                if (!isValidMeasure(X)) // Checking if it's a valid point
                    it.x = it.y = it.z = it.rgb = 0;
                else {
                    it.x = X;
                    it.y = data_cloud[index4 * 4 + 1];
                    it.z = data_cloud[index4 * 4 + 2];
                    color = data_cloud[index4 * 4 + 3];
                    // Color conversion (RGBA as float32 -> RGB as uint32)
                    uint32_t color_uint = *(uint32_t*) & color;
                    unsigned char* color_uchar = (unsigned char*) &color_uint;
                    color_uint = ((uint32_t) color_uchar[0] << 16 | (uint32_t) color_uchar[1] << 8 | (uint32_t) color_uchar[2]);
                    it.rgb = *reinterpret_cast<float*> (&color_uint);
                }
                index4++;
            }
            buffer->mutex_input.unlock();
            //AQUÍ EXTRCCIÓN DE PLANOS ETC.
            nube=point_cloud_ptr;




            std::cout<<"plane 1: "<< nube->points.size() <<endl;
            if(nube->points.size()>500){
            	planextraction(nube,result);
            	//out_data(result);
            	}else{
            		cout<<"plane can't be detected"<<endl;
            	}
            std::cout<<"plane 2: "<< nube->points.size() <<endl;
            if(nube->points.size()>1000){
            	planextraction(nube,result1);
            	//out_data(result1);
            	}else{
            		cout<<"plane can't be detected"<<endl;
            	}

            if(key == '1'){
            	ind = 1;
            	cout<<"KEY :"<< key << endl;
            }else if(key == '2'){
            	ind = 2;
            }


            switch(ind){
            case 1:
            	viewer->updatePointCloud(result);
            	break;
            case 2:
            	viewer->updatePointCloud(result1);
            	break;
            default:
            	viewer->updatePointCloud(result);
            	break;

            }

            key = waitKey(5);

            //viewer->updatePointCloud(result);


            viewer->spinOnce(15);
    	}



    	std::this_thread::sleep_for(std::chrono::milliseconds(1));

    }

	    // Stop the grabbing thread
    stop_signal = true;
    grab_thread.join();

    delete[] buffer->data_cloud;
    delete buffer;
    delete zed;
    return 0;
}
