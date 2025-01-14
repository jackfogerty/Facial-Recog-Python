import cv2
import cvlib as cv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
import numpy as np
import os
from PIL import Image

#this function converts the collection of images and their corresponding emotions to a
#dictionary holding both, then to a pandas dataframe which can then be used to train the model
def generate_data(num_samples = 1000):
    data = {'image':[], 'label':[]}
    for image_file in os.listdir("Facial Recog Python/dataset/train/angry"):
        path = os.path.join("Facial Recog Python/dataset/train/angry", image_file)
        with Image.open(path) as image:
            resized_image = image.resize((48, 48)).convert('L')
            flat_image = np.array(resized_image).flatten()
            data['image'].append(flat_image)
            data['label'].append('angry')
    
    for image_file in os.listdir("Facial Recog Python/dataset/train/disgusted"):
        path = os.path.join("Facial Recog Python/dataset/train/disgusted", image_file)
        with Image.open(path) as image:
            resized_image = image.resize((48, 48)).convert('L')
            flat_image = np.array(resized_image).flatten()
            data['image'].append(flat_image)
            data['label'].append('disgusted')

    for image_file in os.listdir("Facial Recog Python/dataset/train/fearful"):
        path = os.path.join("Facial Recog Python/dataset/train/fearful", image_file)
        with Image.open(path) as image:
            resized_image = image.resize((48, 48)).convert('L')
            flat_image = np.array(resized_image).flatten()
            data['image'].append(flat_image)
            data['label'].append('fearful')

    for image_file in os.listdir("Facial Recog Python/dataset/train/happy"):
        path = os.path.join("Facial Recog Python/dataset/train/happy", image_file)
        with Image.open(path) as image:
            resized_image = image.resize((48, 48)).convert('L')
            flat_image = np.array(resized_image).flatten()
            data['image'].append(flat_image)
            data['label'].append('happy')

    for image_file in os.listdir("Facial Recog Python/dataset/train/neutral"):
        path = os.path.join("Facial Recog Python/dataset/train/neutral", image_file)
        with Image.open(path) as image:
            resized_image = image.resize((48, 48)).convert('L')
            flat_image = np.array(resized_image).flatten()
            data['image'].append(flat_image)
            data['label'].append('neutral')

    for image_file in os.listdir("Facial Recog Python/dataset/train/sad"):
        path = os.path.join("Facial Recog Python/dataset/train/sad", image_file)
        with Image.open(path) as image:
            resized_image = image.resize((48, 48)).convert('L')
            flat_image = np.array(resized_image).flatten()
            data['image'].append(flat_image)
            data['label'].append('sad')

    for image_file in os.listdir("Facial Recog Python/dataset/train/surprised"):
        path = os.path.join("Facial Recog Python/dataset/train/surprised", image_file)
        with Image.open(path) as image:
            resized_image = image.resize((48, 48)).convert('L')
            flat_image = np.array(resized_image).flatten()
            data['image'].append(flat_image)
            data['label'].append('surprised')

    df = pd.DataFrame(data)
    print("done creating dataframe")
    return df

def create_model():
    #splits the data into 4 parts; a set containing training images, a set containing training corresponding labels
    #and a set containing the above for testing. The testing set is 20% of the total data set.
    training_data = generate_data()
    img_train, img_test, label_train, label_test = train_test_split(training_data['image'],
                                                                    training_data['label'], 
                                                                    test_size = 0.00000001, 
                                                                    random_state = 4)
    
    #creates the RandomForest algorithm object
    model = Pipeline([('classifier', RandomForestClassifier(n_estimators=100))])

    img_train_flat = [np.array(img).ravel() for img in img_train]
    
    #training the model
    model.fit(img_train_flat, label_train)

    #saving model to file
    model_filename = 'emotion_detector.pkl'
    joblib.dump(model, model_filename)
    print("Trained model saved to " + model_filename)

    #load model from file
    done_model = joblib.load(model_filename)

#this function uses openCV to capture default camera, and then estimates emotion using the model
def estimate_emotion(loaded_model):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        faces, confidences = cv.detect_face(frame)

        for face, confidences in zip(faces, confidences):
            (start_x, start_y, end_x, end_y) = face

            cropped_face = frame[start_y:end_y, start_x:end_x]

            face_resize = cv2.resize(cropped_face, (40, 40))

            flat_face = face_resize.flatten()

            emotion = loaded_model.predict([flat_face])[0]

            label  = f'Emotion: {emotion}'
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 2)

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    create_model()


    
