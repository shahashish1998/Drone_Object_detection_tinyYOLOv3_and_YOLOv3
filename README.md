# Drone_Object_detection_tinyYOLOv3_and_YOLOv3

Drone_Object_detection_tinyYOLO_and_YOLO
Developer- Ashish Shah, Meghana Murali, Gagan Agarwal, Bharaneeshwar Bala, Minnu Elsa Thomas.
Goal - Develop a project to detect four objects (Person, Bike, Car, Building) in a video captured by drone.
Approach - Developing a YOLOv3 and tinyYOLOv3 model Creation of custom dataset Training the model on a custom dataset Implementation of the model to gain output Developing a GUI for front end.
For the training part
First step is Developing a YOLOv3 model Download the dataset from the darknet/data/dataset folder and save it in a new folder . In the darknet/cfg/yolov3_custom.cfg is for YOLOv3 and darknet/cfg/yolo-tony is for the tiny-yolov3.

The pre-trained weights file to fit your custom data For YOLOv3 - darknet53.conv.74 (154 MB) For tinyYOLOv3 - yolov3-tiny.conv.11 (6 MB).

now firstly run the train_test_spl.py python file that split the whole data into train and test.

The train.txt or the test.txt contain's the path of the file like:

data/dataset/video_MP4_184.jpg
data/dataset/video_MP4_203.jpg
data/dataset/video_MP4_237.jpg
data/dataset/video_MP4_257.jpg
data/dataset/video_MP4_283.jpg
data/dataset/video_MP4_306.jpg
......... 
The file obj.names in your repository containing the names of your object to be identified. In this case, obj.names contains Person Bike Car Building.

person
car
bike
building
And obj.data file in repository with data:

classes =4
train = data/train.txt
valid = data/test.txt
names = data/obj.names
backup = backup
Training the model on a custom dataset In this case Google Colab was used for the project, any other notebook can also be used. Do use GPU from runtime in Colab. Only for Colab Notebook by runing the following command make sure that u clone the repository and mount your google drive

!./darknet detector train "/obj.data" "/cfg/yolov3-tiny-custom.cfg" "/cfg/yolov3-tiny-custom_last.weights" -dont_show 
COMPARESION
after training model we get below comarison:

Metrics	TinyYOLOv3	YOLOv3
TruePositives(TP)	793	1920
FalsePositives(FP)	1896	1460
FalseNegatives(FN)	2891	1764
mAP(MeanAverage Precision)after 3000 iterations	33.31%	60.75%
Sensitivity/Recall	22%	52.11%
Avg.IOU	18.75%	37.94%
Avg.Loss	18.7-19.2%	16.4-16.9%
 Graph plotted using mAP values

IMPLEMENTATION
The last stem is to implement by using open cv so for this yolo.py. For image input if FLAGS.image_path is None and FLAGS.video_path is None: print ('Neither path to an image or path to video provided') print ('Starting Inference on Webcam')

For the image input the code in yolo.py is:

try:
    img = cv.imread(img_path)
    height, width = img.shape[:2]
except:
    raise 'Image cannot be loaded!\n\
            Please check the path provided!'
finally:
img, _, _, _, _ = infer_image(net, obj_dec, layer_names, height, width, img, colors, labels, FLAGS)
cv.imwrite('image_out.jpg',img)
For video input:

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
        if not grabbed:
                break
        if width is None or height is None:
            height, width = frame.shape[:2]

        frame, _, _, _, _ = infer_image(net, obj_dec, layer_names, height, width, frame, colors, labels, FLAGS)
        if writer is None:
            fourcc = cv.VideoWriter_fourcc(*"MJPG")
            writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30, 
            (frame.shape[1], frame.shape[0]), True)
            writer.write(frame)
    print ("[INFO] Cleaning up...")
    writer.release()
    vid.release()
    cap = cv.VideoCapture('output.mp4')
and for last live feed:

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
vid.release()
cv.destroyAllWindows()
deploying in the GUI
At the last Developing a GUI:

The trial GUI2(1) contain the code of GUI using the Tkinter library in Python. some screenshots are:

screen shot

!screen shot
