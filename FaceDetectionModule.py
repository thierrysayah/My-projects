import cv2
import mediapipe as mp
import time



class FaceDetector():
    def __init__(self, minDetCon = 0.5):
        self.minDetCon = minDetCon
        # setup face detection
        self.mpFaceDet = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDet = self.mpFaceDet.FaceDetection(self.minDetCon)


    def findFaces(self,img, draw = True):
        #convert from BGR color space to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #process image for face detection
        self.results = self.faceDet.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id,bbox,detection.score])
                if draw:
                    img = self.fancyDraw(img,bbox)
                    cv2.rectangle(img,bbox,(255,0,255),2)
                    cv2.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
        return img, bboxs

    def fancyDraw(self,img, bbox,l =30, t =5, rt =1):
        x,y,w,h = bbox
        x1,y1=x+w, y+h

        # cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # top left markers
        cv2.line(img,(x,y),(x+l,y),(255,0,255),t)
        cv2.line(img,(x, y), (x , y+l), (255, 0, 255), t)
        return img

def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    detector = FaceDetector()
    while True:
        # read frame
        success, img = cap.read()
        img, bboxs= detector.findFaces(img)
        # calculate and show fps on image
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, f'FPS :{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        # show final image
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()