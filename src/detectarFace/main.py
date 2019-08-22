import cv2

image_path = '../../dados/images/img.jpg'

img = cv2.imread(image_path)

clf = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = clf.detectMultiScale(gray,1.3,10)

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),2)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)