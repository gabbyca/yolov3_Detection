import cv2
import numpy as np
import time


#initialize camera
camera = 'tcp://10.161.141.50:5000'
#stream = cv2.VideoCapture(camera)
stream = cv2.VideoCapture(0)


# def findFace(frame):
#     faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(gray, 1.3, 6)
#     for(x,y,w,h) in faces:
#             frame = cv2.rectangle(frame, (x,y), (x+w, y+h), color =(0,255,0), thickness=5)
#     return frame 


#vis rec YOLOv3
def loadYolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()] 
    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

def load_image(img_path):
    # image loading
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shapelo
    return img, height, width, channels

# detect object in image using blob
def detect_objects(img, net, outputLayers):         
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

#set dimensions of object and coordinates of image within frame
#only set coordinates after testing confidence of the object in the image 
def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids

#configure label to show around images if detected
def draw_labels(boxes, confs, colors, class_ids, classes, img): 
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    cv2.imshow("Image", img)

#combine all methods to use for real time detection
#will compare each frame of video using functions above to test confidence of 
#the object in the image
def webcam_detect():
    model, classes, colors, output_layers = loadYolo()
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()

#set standards for writing and frames per second
fps = int(stream.get(cv2.CAP_PROP_FPS))
t= time.localtime()
current_time = time.strftime("%H:%M:%S", t)


output = cv2.VideoWriter('videoStorageOpencv/' + current_time + '.avi',cv2.VideoWriter_fourcc('M','J','P','G'),fps,(70,70))

while(True):
    #read frames of stream
    ret, frame = stream.read()  
    imgOriginal = frame.copy()

    #yolo
    loadYolo()  
    webcam_detect()
    
    #show video
    cv2.imshow('Video', frame)
    output.write(frame)

    #break code if 'q' pressed
    if cv2.waitKey(1) == ord('q'):
        break

#release objs
output.release()
stream.release()
cv2.destroyAllWindows()

