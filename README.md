# Drone_Object_detection_tinyYOLOv3_and_YOLOv3

## Developed by _Ashish Shah, Meghana Murali, Gagan Agarwal, Bharaneeshwar Bala, Minnu Elsa Thomas_.

### Goal - Develop a project to detect four objects (Person, Bike, Car, Building) in a video captured by drone. 
 
#### Approach -
- Developing a YOLOv3 and tinyYOLOv3 model
- Creation of custom dataset
- Training the model on a custom dataset
- Implementation of the model to gain output
- Developing a GUI for front end
 
 
Download the cfg/yolov3-tiny-custom.cfg, cfg/yolov3-tiny-custom_last.weights and obj.data
Compile the code using the video
```
!./darknet detector train "/obj.data" "darknet/cfg/yolov3-tiny-custom.cfg" "darknet/backup/yolov3-tiny-custom_last.weights" -dont_show
```
 
 
#### 1.) Developing a YOLOv3 model
- Download the dataset from the img/ folder and save it in a new folder img/ in your repository.
- Download the cfg/yolov3_custom.cfg for YOLOv3 and cfg/yolo-tony in your repository
- Download the pre-trained weights file to fit your custom data in your repository
- For YOLOv3 - darknet53.conv.74 (154 MB)
- For tinyYOLOv3 - yolov3-tiny.conv.11 (6 MB)
- Make a train.txt and test.txt file in your repository which contains:
- ‘Path of your repository’/img/name_of_imagefile.jpg
 

 
- Create a file obj.names in your repository containing the names of your object to be identified. In this case, obj.names contains 
    - Person
    - Bike
    - Car
    - Building 
 

 
- Create a new folder in your repository named as ‘backup’. Here, all your updated trained weights will be saved.
- Make an obj.data file in your repository with data:
    - classes = ‘no. of classes or no. of object you want to detect’
    - train  = ‘path to your repository’/train.txt
    - valid  = ‘path to your repository’/test.txt
    - names =‘path to your repository’/obj.names
    - backup = ‘path to your repository’/backup
 

 
- Now open the .cfg file and make these changes according to you.
    - change line batch to batch=64
    - change line subdivisions to subdivisions=64
    - change line max_batches to (classes*2000 but not less than the number of training images, but not less than the number of training images and not less than 6000), f.e. max_batches=10000 if you train for 4 classes
    - change line steps to 80% and 90% of max_batches, f.e. steps=8000,9000
    - set network size width=416 height=416 or any value multiple of 32
    - change line classes=4 to your number of objects in each of 2 [yolo]-layers
    - change [filters=27] to filters=(classes + 5)x3 in the 3 [convolutional] before each [yolo] layer, keep in mind that it only has to be the last [convolutional] before each of the [yolo] layers. 
    - So if classes=1 then should be filters=18. If classes=2 then write filters=21.

		
 
#### 2.) Creation of custom dataset:
- The aim of this project is detection of a person, motorcycle, car or building from a drone footage taken at a height of 70 meters. This leads to the necessity of creating a custom dataset. There are many tools available online to create annotations for custom dataset. 
- There are different ways to show the detected object like highlighting the outline of the object or drawing bounding boxes around it. Depending on this the dataset created has the labels. Here bounding boxes have been used. The features it considers are: class(i.e. which object it is, height, width, x and y coordinates.
 
#### 3.) Training the model on a custom dataset
- In this case Google Colab was used for the project, any other notebook can also be used. Do use GPU from runtime in Colab.
Only for Colab Notebook
```# This cell imports the drive library and mounts your Google #Drive as a VM local drive. You can access your Drive files 
# using this path "/content/gdrive/My Drive/"
 
from google.colab import drive
drive.mount('/content/gdrive')
```
- Save all the above downloaded and edited files in a folder named /darknet and see the content of it
```# List the content of your local computer folder 
!ls -la "Path_of_darknet_repository/darknet"
```

- Now, you need to download cuDNN from the Nvidia web site. You'll need to sign up on the site. Download cuDNN from Nvidia website. Right now, because we have CUDA 10.0 preinstalled in Colab runtime, you need download cuDNN v7.5.0.56 for CUDA v10.0 - the file is cudnn-10.0-linux-x64-v7.5.0.56.tgz
On your local computer, create a folder named cuDNN in your local folder darknet. Copy the tgz file there
 
```# We're unzipping the cuDNN files from your Drive folder directly to the VM CUDA folders
!tar -xzvf gdrive/My\ Drive/darknet/cuDNN/cudnn-10.0-linux-x64-v7.5.0.56.tgz -C /usr/local/
!chmod a+r /usr/local/cuda/include/cudnn.h
 ```
 Now we check the version we already installed. Can comment this line on future runs
```
!cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

- Cloning and compiling Darknet. ONLY NEEDS TO BE RUN ON THE FIRST EXECUTION!!
In this step, we'll clone the darknet repo and compile it.
    - Clone Repo
    - Compile Darknet
    - Copy compiled version to Drive
 
```# Leave this code uncommented on the very first run of your notebook or if you ever need to recompile darknet again.
 #Comment this code on the future runs.
!git clone https://github.com/kriyeng/darknet/
%cd darknet
 
 #Check the folder
!ls
 
!git checkout feature/google-colab
 
#Compile Darknet
!make
 
#Copies the Darknet compiled version to Google drive
!cp ./darknet /content/gdrive/My\ Drive/darknet/bin/darknet
```

 
- Copy the darknet compiled version from drive to the VM.
- Make the local darknet folder
- Copy the darknet file
- Set execution permissions
 
``` 
# Copy the Darkent compiled version to the VM local drive
!cp /content/gdrive/My\ Drive/darknet/bin/darknet ./darknet
 
# Set execution permissions to Darknet
!chmod +x ./darknet
```

 
##### To start training use the command
```
#use “path_of_your_darknet’’/files_as_used_below
!./darknet detector train "/content/gdrive/My Drive/darknet/obj.data" "/content/gdrive/My Drive/darknet/yolov3-tiny-custom.cfg" "/content/gdrive/My Drive/darknet/yolov3-tiny.conv.15" -dont_show 
```

 
#### 4. Implementation of the model to gain output. 
Make a python code that take input video.(yolo.py)
- For live feed:
```
def live_feed(obj_dec):
    # Load the weights and configutation to form the pretrained tinyYOLOv3 model
    net = cv.dnn.readNetFromDarknet('darknet/cfg/yolov3-tiny.cfg', 'darknet/backup/yolov3-tiny.weights')
    count=0
    vid = cv.VideoCapture(0)
    while True:
        _, frame = vid.read()
        height, width = frame.shape[:2]

        if count == 0:
            frame, boxes, confidences, classids, idxs = infer_image(net,  obj_dec, layer_names, \
                                height, width, frame, colors, labels, FLAGS)
            count += 1
        else:
            frame, boxes, confidences, classids, idxs = infer_image(net,  obj_dec, layer_names, \
                                height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
            count = (count + 1) % 6

        cv.imshow('webcam', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
```

			
- For video input
``` 
def vid_det_tinyYOLO(vid_path,obj_dec):
    # Load the weights and configutation to form the pretrained tinyYOLOv3 model
    net = cv.dnn.readNetFromDarknet('darknet/cfg/yolov3-tiny-custom.cfg', 'darknet/backup/yolov3-tiny-custom_last.weights')

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    try:
        vid = cv.VideoCapture(vid_path)
        height, width = None, None
        writer = None
    except:
        raise 'Video cannot be loaded!\n\
                            Please check the path provided!'

    finally:
        while True:
            grabbed, frame = vid.read()
            # Checking if the complete video is read
            if not grabbed:
                break

            if width is None or height is None:
                height, width = frame.shape[:2]

            frame, _, _, _, _ = infer_image(net, obj_dec, layer_names, height, width, frame, colors, labels, FLAGS)

            if writer is None:
                # Initialize the video writer
                fourcc = cv.VideoWriter_fourcc(*"MJPG")
                writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30, 
                                (frame.shape[1], frame.shape[0]), True)


            writer.write(frame)
            cv.imshow('frame',frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
 ```
The other yolo_util python file will draw the boundary boxes at the selected object.
- Selection of layer
```def infer_image(net, obj_dec, layer_names, height, width, img, colors, labels, FLAGS, 
            boxes=None, confidences=None, classids=None, idxs=None, infer=True):
    
    if infer:
        # Contructing a blob from the input image
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), 
                        swapRB=True, crop=False)
 
        # Perform a forward pass of the YOLO object detector
        net.setInput(blob)
 
        # Getting the outputs from the output layers
        start = time.time()
        outs = net.forward(layer_names)
        end = time.time()
 
        if FLAGS.show_time:
            print ("[INFO] YOLOv3 took {:6f} seconds".format(end - start))
 
        
        # Generate the boxes, confidences, and classIDs
        boxes, confidences, classids = generate_boxes_confidences_classids(outs, obj_dec, height, width, FLAGS.confidence)
 ```
- Generate the confidence of each box.
```def generate_boxes_confidences_classids(outs, obj_dec, height, width, tconf):
    boxes = []
    confidences = []
    classids = []
    for out in outs:
        for detection in out:
            #print (detection)
            #a = input('GO!')`
            
 `           # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
 ```       	
 
- Draw the labels and boxes of selective object
```def draw_labels_and_boxes(img, obj_dec, boxes, confidences, classids, idxs, colors, labels):
     If there are any detections
    if len(idxs) > 0:
        for i in idxs.flatten():
          if(labels[classids[i]] in obj_dec):
              # Get the bounding box coordinates
              x, y = boxes[i][0], boxes[i][1]
              w, h = boxes[i][2], boxes[i][3]
              # Get the unique color for this class
              color = [int(c) for c in colors[classids[i]]]
            # print(classids[i])
              # Draw the bounding box rectangle and label on the image
              cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
              text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
              cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
 ```
5. Developing a GUI:
 
- For ease of use of the project, a GUI was created such that any person could operate it and detect the objects he/she wished to. The GUI was created using the Tkinter library in Python. 
- The project consists of three parts, detecting objects in a video using tinyYOLO, detecting objects in a video using YOLO and detecting objects on a live feed using tinyYOLO. 
- The following flowchart summarises the operation of the GUI:
- There are four pages in the GUI, all the pages are linked with each other. The user at any point of time can choose what input to give and which objects to detect.
