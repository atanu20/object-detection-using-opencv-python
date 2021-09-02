import cv2

# img=cv2.imread('lena.png')
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)


className=[]
classFile="coco.names"
with open (classFile, 'rt') as f:
    className=f.read().rstrip('\n').split('\n')

configpath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightpath='frozen_inference_graph.pb'

net= cv2.dnn_DetectionModel(weightpath,configpath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True:
    success,img=cap.read()
    print(img.shape)

    classId, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classId,bbox)
    if(len(classId)!=0):
        for id, confi, box in zip(classId.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, className[id - 1].upper(), (box[0] + 10, box[1] + 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)

    cv2.imshow('Output', img)
    if(cv2.waitKey(1)==13):
        break