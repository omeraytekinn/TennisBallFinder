import cv2
import numpy as np
import os
import imutils
from skimage.measure import compare_ssim
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
        

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]


def showImage(img):
    cv2.imshow("Preview",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def findObjects(filePath):
    print("processing: "+filePath)
    img = cv2.imread(filePath)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(
        img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(outputlayers)
    boxes = []
    cnt = 0
    indexes = []
    for out in outs:
        for detection in out:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            x = int(center_x - w/2)
            y = int(center_y - h/2)

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.8:
                indexes.append(cnt)
                boxes.append([x, y, w, h])
                cnt += 1
    objects = []
    org_objects= []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            obj = img[y:y+h, x:x+w]
            a,b,c = obj.shape
            if a==0 or b==0 or c==0:
                continue
            obj = cv2.resize(obj,(30,30))
            objects.append(obj)
            org_obj = img[y:y+h, x:x+w]
            org_objects.append(org_obj)
    return objects, org_objects

def createDatasetFromFolder(folderName, datasetPath="dataset3"):
    files = os.listdir(folderName)
    names = []
    dataset = []
    if not os.path.exists(datasetPath):
        os.makedirs(datasetPath)
    for fileName in files:
        objects, org_objects= findObjects(folderName+"/"+fileName)
        for i,obj in enumerate(objects):
            cv2.imwrite(datasetPath+"/"+fileName.split('.')[0]+"_"+str(i)+".jpg", obj)


def findImage(imgPath, datasetPath="dataset3", resultPath="result"):
    objects, org_objects = findObjects(imgPath)
    flag = False
    images = os.listdir(datasetPath)
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    for i,obj in enumerate(objects):
        for imgName in images:
            image = cv2.imread(datasetPath+'/'+imgName)
            (score, diff) = compare_ssim(obj, image, full=True, multichannel=True)
            diff = (diff * 255).astype("uint8")
            if score > 0.3:
                flag = True
                print("Tennis Ball Found!")
                cv2.imwrite(resultPath+'/'+imgPath.split('/')[1].split('.')[0]+'_tennis.jpg',org_objects[i])
                break
        if flag == True:
            break
        
def testImages(testPath="test"):
    images = os.listdir(testPath)
    for image in images:
        if image != '.DS_Store':
            findImage(testPath+"/"+image)

createDatasetFromFolder("tennis")
testImages()