import cv2
import numpy as np
import time
# from flask import Flask
# app = Flask(__name__)
def detectFaces():
    #Load YOLO
    net = cv2.dnn.readNetFromDarknet("Yolo/yolo_models/yolov3-face.cfg", "Yolo/yolo_weights/yolov3-face.weights")

    classes = []
    with open("Yolo/yolo_labels/yolo-face-labels","r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    colors= np.random.uniform(0,255,size=(len(classes),3))

    #loading image
    cap=cv2.VideoCapture(0) #0 for 1st webcam
    font = cv2.FONT_HERSHEY_PLAIN
    starting_time= time.time()
    frame_id = 0

    while True:
        _,frame= cap.read() 
        frame=cv2.flip(frame,1)
        frame_id+=1
        
        height,width,channels = frame.shape
        #detecting objects
        blob = cv2.dnn.blobFromImage(frame,0.003125,(320,320),(0,0,0),True,crop=False) #reduce 416 to 320    

            
        net.setInput(blob)
        outs = net.forward(outputlayers)
        # print(outs[1])


        #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
        class_ids=[]
        confidences=[]
        boxes=[]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    #object detected
                    center_x= int(detection[0]*width)
                    center_y= int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    cv2.circle(frame,(center_x,center_y),10,(0,255,0),2)
                    #rectangle co-ordinaters
                    x=int(center_x - w/2)
                    y=int(center_y - h/2)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                    boxes.append([x,y,w,h]) #put all rectangle areas
                    confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                    class_ids.append(class_id) #name of the object tha was detected

        indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)

        cv2.putText(frame,"Faces Detected:"+str(len(indexes)),(10,50),font,2,(255,255,255),1)
        filetowrite = open("countFile.txt", "w")
        filetowrite.write(str(len(indexes)))
        filetowrite.close()
        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence= confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(0,255,0),2)
                # cv2.putText(frame,"FID: "+str(i+1),(x+5,y+20),font,1,(0,255,0),2)
                

        elapsed_time = time.time() - starting_time
        fps=frame_id/elapsed_time
        cv2.putText(frame,"FPS:"+str(round(fps,2)),(30,450),font,2,(0,255,0),1)
        
        cv2.imshow("Image",frame)
        key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
        
        if key == 27 or key =='q': #esc or q key stops the process
            break

    cap.release()    
    cv2.destroyAllWindows()
    return str(len(indexes))

# @app.route('/count')
# def new():
#     num = detectFaces()
    # f= open("countFile.txt","r")
    # content = f.read()
    # f.close()
    # return num

# @app.route('/')
# def hello():
#     return 'Welcome to MIRAI VIZION'

if __name__ == "__main__":
    detectFaces()
