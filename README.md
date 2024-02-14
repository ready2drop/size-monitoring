# Size Monitoring by YOLOv8 :heavy_check_mark:
The algorithm is elaborated on our paper [Deep Learning-Based Pancreas Detection and Size Monitoring with Data Augmentation for Medical Imaging Analysis(KSC 2023)](https://github.com/ready2drop/size-monitoring/blob/main/paper.pdf)
# Steps to run Code
 
 ### If you are using google colab then you will first need to mount the drive with mentioned command first, (Windows or Linux users) both can skip this step.
 ``` 
 from google.colab import drive
 drive.mount("/content/drive")
 ```
 ### Install requirements with mentioned command below.
 ```
 pip install -r requirements.txt
 ```

 - Download yolov8 detection weights from [link](https://github.com/ultralytics/ultralytics) and move them to the working directory {yolov8x,pt}

 ### Run the code with mentioned command below.

 ```
 # baseline
 python size_monitoring.py

 # Change your own data directory
 data_path = "preprocessing/nii2video/*"

 ```
 

# Reference
- https://github.com/ultralytics/ultralytics.git
