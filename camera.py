import cv2
import face_recognition
import os
from datetime import datetime
import numpy as np

class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret,frame=self.video.read()
        img1 = face_recognition.load_image_file("images/gandhi.jpg")
        img1_face_encoding = face_recognition.face_encodings(img1)[0]
        img2 = face_recognition.load_image_file("images/kalpana.jpg")
        img2_face_encoding = face_recognition.face_encodings(img2)[0]
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        known_face_encodings = [
            img1_face_encoding,
            img2_face_encoding
        ]
        known_face_names = [
            "gandhi",
            "kalpana"
        ]
        #for marking attendance in excel
        def markAttendance(name):
            with open('Attendance.csv', 'r+') as f:
                myDataList = f.readlines()
                nameList = []
                for line in myDataList:
                    entry = line.split(',')  # for new line
                    nameList.append(entry[0])  # appends attendy name
                if name not in nameList:  # checks the name in list
                    now = datetime.now()
                    dtString = now.strftime('%H:%M:%S')
                    f.writelines(f'\n{name},{dtString}')

        process_this_frame = True
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            markAttendance(name)

        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()
