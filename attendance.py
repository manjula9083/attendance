import cv2
import face_recognition


img1 = face_recognition.load_image_file('images/kalpana.jpg')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('images/gandhi.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(img1)[0]  #\*it contains top,right,bottom,left*/
encodeElon = face_recognition.face_encodings(img1)[0]
cv2.rectangle(img1,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2) #first for color of rectangle and next is 2 for thickness

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)#this is for test image recognition

results = face_recognition.compare_faces([encodeElon],encodeTest)#for comparing img is same person or not
faceDis = face_recognition.face_distance([encodeElon],encodeTest)#comparing distance btw the faces
print(results,faceDis)
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
# we can put text in img        (to put distance,roundoff),origin,fontstyle,1=scale,color,width
cv2.imshow('kalpana',img1)
cv2.imshow('gandhi',imgTest)
cv2.waitKey(0)