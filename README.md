# Size Monitoring by detection algorithm  



# Features
 - Support for Comparision of (FPS & Time) Graph
 - How to run Code in Google Colab
 - Code can run on Both (CPU & GPU)
 - Video/WebCam/External Camera/IP Stream Supported


# Steps to run Code
 
 ### If you are using google colab then you will first need to mount the drive with mentioned command first, (Windows or Linux users) both can skip this step.
 ``` 
 from google.colab import drive
 drive.mount("/content/drive")
 ```
 ### Clone the repository.
 git clone https://github.com/ready2drop/yolov7_pose.git

 ### Go to the cloned folder.
 ```
 cd yolov7_pose
 ```
 ### Create a virtual envirnoment (Recommended, If you dont want to disturb python packages)
 ```
 # For Linux Users
 python3 -m venv psestenv
 source psestenv/bin/activate

 # For Window Users
 python3 -m venv psestenv
 cd psestenv
 cd Scripts
 activate
 cd ..
 cd ..
 ```

 ### Upgrade pip with mentioned command below.
 ```
 pip install --upgrade pip
 ```

 ### Install requirements with mentioned command below.
 ```
 pip install -r requirements.txt
 ```

 - Download yolov7 pose estimation weights from [link](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt) and move them to the working directory {yolov7-w6-pose,pt}

 ### Run the code with mentioned command below.

 ```
 #baseline
 python pose-estimate.py

 #if you want to change source file
 python pose-estimate.py --source "your custom video.mp4"

 #For CPU (defualt 'cpu')
 python pose-estimate.py --source "your custom video.mp4" --device cpu

 #For GPU (0,1,2,3 - device arguments of gpus)
 python pose-estimate.py --source "your custom video.mp4" --device 0

 #For LiveStream (Ip Stream URL Format i.e "rtsp://username:pass@ipaddress:portno/video/video.amp")
 python pose-estimate.py --source "your IP Camera Stream URL" --device 0

 #For WebCam
 python pose-estimate.py --source 0

 #For External Camera
 python pose-estimate.py --source 1
 ```
 
- Output file will be created in the working directory with name ["your-file-name-without-extension"+"_keypoint.mp4"] 

# Result   :mega:
|Football|Dance|Snowboard|
|---|---|---|
|![football](https://github.com/ready2drop/yolov7_pose/assets/89971553/6969ec65-2288-4c01-8799-3cb462169e06)|![dance](https://github.com/ready2drop/yolov7_pose/assets/89971553/bc3a8d65-112d-4223-8668-65e076e5acc6)|![snowboard](https://github.com/ready2drop/yolov7_pose/assets/89971553/d9fd1552-9ac2-4738-bff8-4373c9245a16) |


# Reference
- https://github.com/WongKinYiu/yolov7
- https://github.com/WongKinYiu/yolov7/releases
- https://learnopencv.com/yolov7-object-detection-paper-explanation-and-inference/
