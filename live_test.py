import cv2
import cvlib as cv
import joblib
from PIL import Image

#this function uses openCV to capture default camera, and then estimates emotion using the model
def estimate_emotion(path):
    loaded_model = joblib.load(path)

    #selects defauly camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        faces, confidences = cv.detect_face(frame) #detecting faces in the frame

        #crops each face in the frame, and predicts what emotion they are feeling, then prints that prediction
        for face, confidences in zip(faces, confidences):
            (start_x, start_y, end_x, end_y) = face

            cropped_face = frame[start_y:end_y, start_x:end_x]

            face_resize = cv2.resize(cropped_face, (48, 48)).convert('L')

            flat_face = face_resize.flatten()

            emotion = loaded_model.predict([flat_face])[0]

            label  = f'Emotion: {emotion}'
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 2)

        cv2.imshow('Emotion Detection', frame)

        #press q to close this program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    estimate_emotion("Facial Recog Python/emotion_detector.pkl")