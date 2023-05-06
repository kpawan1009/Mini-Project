from ultralytics import YOLO
import cv2
import cvzone
import math
import winsound

# cap=cv2.VideoCapture('../videos/garbagekaran.mp4')
cap=cv2.VideoCapture('../videos/garbagevid3.mp4')

model = YOLO("../bestgarbage.pt")

classNames2 = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

model2=YOLO('../yolo-weights/bestn.pt')

classNames=['bottle', 'can']
count=0
myColor=(0,255,0)
#bgr
while True:
    success,img=cap.read()
    results=model(img,stream=True)
    resulthuman=model(img,stream=True)
    for rhuman in resulthuman:
        for r in results:
            istrue = False
            boxes=r.boxes
            for box in boxes:
                myColor = (0,255, 0)
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                w,h=x2-x1,y2-y1
                # boxeshuman = rhuman.boxes
                # for boxhuman in boxeshuman:
                #     clshuman = int(boxhuman.cls[0])
                #     print(clshuman)
                #     print(clshuman)
                #     print(clshuman)
                #     print(clshuman)
                #     print(clshuman)
                #     print(clshuman)
                #     print(clshuman)
                #     print(clshuman)
                #     print(clshuman)
                #     if classNames2[clshuman] == "person":
                #         print("Human")
                #         print("Human")
                #         print("Human")
                #         print("Human")
                #         print("Human")
                #         x1h, y1h, x2h, y2h = box.xyxy[0]
                #         x1h, y1h, x2h, y2h = int(x1h), int(y1h), int(x2h), int(y2h)
                #         wh, hh = x2h - x1h, y2h - y1h
                #         if (x1+x2)/2 < x2h and (x1+x2)/2 > x1h and (y1+y2)/2 < y2h and (y1+y2)/2 > y1h :
                #             istrue=True
                conf=math.ceil(box.conf[0]*100)/100
                print(conf)
                cls=int(box.cls[0])
                i=0
                while i<20:
                    i+=1
                    print(istrue)
                if(classNames[cls]=='can' or classNames[cls] == 'bottle' and istrue==False and conf>0.7):
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
                    myColor=(0,0,255)
                    cvzone.putTextRect(img, f'{classNames[cls]}', (max(0, x1), max(35, y1)), scale=1, thickness=1,
                                       colorB=myColor, colorT=(255, 255, 255), colorR=myColor)
                    filename="Image"+str(count)+".jpg"
                    count+=1
                    cv2.imwrite(filename,img)
                    winsound.Beep(1445, 100)

    cv2.imshow("Image",img)
    cv2.waitKey(1)