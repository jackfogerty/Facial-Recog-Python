import cv2
import cvlib as cv
import joblib
from PIL import Image
import time

#this function uses openCV to capture default camera, and then estimates emotion using the model
def estimate_emotion(path):
    loaded_model = joblib.load(path)

    #selects defauly camera
    cap = cv2.VideoCapture(0)
    emotion = None
    last_prediction_time = 0
    while True:
        ret, frame = cap.read()

        faces, confidences = cv.detect_face(frame) #detecting faces in the frame

        current_time = time.time()

        #crops each face in the frame, and predicts what emotion they are feeling, then prints that prediction
        counter = 0
        for face, confidences in zip(faces, confidences):
            (start_x, start_y, end_x, end_y) = face

            cropped_face = frame[start_y:end_y, start_x:end_x]

            face_resize = cv2.resize(cropped_face, (48, 48))

            gray_face = cv2.cvtColor(face_resize, cv2.COLOR_BGR2GRAY)

            flat_face = gray_face.flatten()

            if current_time - last_prediction_time >= 3:
                emotion = loaded_model.predict([flat_face])[0]
                last_prediction_time = current_time

            label  = f'Emotion: {emotion}'
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 2)
            counter += 1

        cv2.imshow('Emotion Detection', frame)

        #press q to close this program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    estimate_emotion("emotion_detector.pkl")