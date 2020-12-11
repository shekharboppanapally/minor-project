import numpy as np
import tensorflow as tf
import cv2
import numpy as np
import tensorflow as tf
import cv2
model = tf.keras.models.load_model('minorproject.h5')

a=np.zeros([300,300],dtype='uint8')
cv2.rectangle(a,(50,50),(250,250),(102,178,255),-5)
print("press P for prediction")
print("press C for clear")
print("press ESC for quit")
wname="Digits"
stat=False
cv2.namedWindow(wname)

def digits(event,x,y,flags,param):
    global stat
    if event == cv2.EVENT_LBUTTONDOWN:
        stat = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if stat==True:
            cv2.circle(a,(x,y),5,(255,255,255),-3)
    elif event == cv2.EVENT_LBUTTONUP:
        stat = False
cv2.setMouseCallback(wname,digits)

while True:
    cv2.imshow(wname,a)
    key= cv2.waitKey(1)
    if key == 27:
        break
    elif key ==ord('p'):
        digit = a[50:250,50:250]
        digit = cv2.resize(digit,(28,28)).reshape(1,28,28)
        print(np.argmax(model.predict(digit)))
    elif key == ord('c'):
        a[:,:] = 0
        cv2.rectangle(a,(50,50),(300,300)(102,178,255),-5)
cv2.destroyAllWindows()    
        
