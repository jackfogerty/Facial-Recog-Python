import numpy as np
from PIL import Image
import os
import joblib

def test_dictionary():
    data = {'image':[], 'label':[]}
    num_correct = 0
    num_photos = 0
    num_happy = 0
    num_sad = 0
    num_angry = 0
    num_disgusted = 0
    num_surprised = 0
    num_scared = 0
    num_neutral = 0
    num_type_photos = []
    i = 0
    for image_file in os.listdir("Facial Recog Python/dataset/test/angry"):
        i += 1
        path = os.path.join("Facial Recog Python/dataset/test/angry", image_file)
        with Image.open(path) as image:
            flat_image = np.array(image).flatten()
            data['image'].append(flat_image)
            data['label'].append('angry')
            num_type_photos.append(i)
    
    i = 0
    for image_file in os.listdir("Facial Recog Python/dataset/test/disgusted"):
        i += 1
        path = os.path.join("Facial Recog Python/dataset/test/disgusted", image_file)
        with Image.open(path) as image:
            flat_image = np.array(image).flatten()
            data['image'].append(flat_image)
            data['label'].append('disgusted')
            num_type_photos.append(i)

    i = 0
    for image_file in os.listdir("Facial Recog Python/dataset/test/fearful"):
        i += 1
        path = os.path.join("Facial Recog Python/dataset/test/fearful", image_file)
        with Image.open(path) as image:
            flat_image = np.array(image).flatten()
            data['image'].append(flat_image)
            data['label'].append('fearful')
            num_type_photos.append(i)

    i = 0
    for image_file in os.listdir("Facial Recog Python/dataset/test/happy"):
        i += 1
        path = os.path.join("Facial Recog Python/dataset/test/happy", image_file)
        with Image.open(path) as image:
            flat_image = np.array(image).flatten()
            data['image'].append(flat_image)
            data['label'].append('happy')
            num_type_photos.append(i)

    i = 0
    for image_file in os.listdir("Facial Recog Python/dataset/test/neutral"):
        i += 1
        path = os.path.join("Facial Recog Python/dataset/test/neutral", image_file)
        with Image.open(path) as image:
            flat_image = np.array(image).flatten()
            data['image'].append(flat_image)
            data['label'].append('neutral')
            num_type_photos.append(i)

    i = 0
    for image_file in os.listdir("Facial Recog Python/dataset/test/sad"):
        i += 1
        path = os.path.join("Facial Recog Python/dataset/test/sad", image_file)
        with Image.open(path) as image:
            flat_image = np.array(image).flatten()
            data['image'].append(flat_image)
            data['label'].append('sad')
            num_type_photos.append(i)

    i = 0
    for image_file in os.listdir("Facial Recog Python/dataset/test/surprised"):
        i += 1
        path = os.path.join("Facial Recog Python/dataset/test/surprised", image_file)
        with Image.open(path) as image:
            flat_image = np.array(image).flatten()
            data['image'].append(flat_image)
            data['label'].append('surprised')
            num_type_photos.append(i)

    model = joblib.load("emotion_detector.pkl")

    for i in range(len(data['image'])):
        img = data['image'][i]
        reshaped_img = np.array(img.reshape(1, -1))
        prediction = model.predict(reshaped_img)
        if prediction == data['label'][i]:
            num_correct += 1
        if prediction == "angry":
            num_angry += 1
        if prediction == "happy":
            num_happy += 1
        if prediction == "sad":
            num_sad += 1
        if prediction == "surprised":
            num_surprised += 1
        if prediction == "neutral":
            num_neutral += 1
        if prediction == "fearful":
            num_scared += 1
        if prediction == "disgusted":
            num_disgusted += 1
        num_photos += 1

    percent_accuracy = num_correct/num_photos * 100
    print(f'The model is {percent_accuracy}% accurate overall.')
    print(f'{num_angry/num_photos * 100}% of the guesses were angry, and {num_type_photos[0]/num_photos}% of the photos are angry')
    print(f'{num_sad/num_photos * 100}% of the guesses were sad, and {num_type_photos[5]/num_photos}% of the photos are sad')
    print(f'{num_happy/num_photos * 100}% of the guesses were happy, and {num_type_photos[3]/num_photos}% of the photos are happy')
    print(f'{num_surprised/num_photos * 100}% of the guesses were surprised, and {num_type_photos[6]/num_photos}% of the photos are surprised')
    print(f'{num_neutral/num_photos * 100}% of the guesses were neutral, and {num_type_photos[4]/num_photos}% of the photos are neutral')
    print(f'{num_disgusted/num_photos * 100}% of the guesses were disgusted, and {num_type_photos[1]/num_photos}% of the photos are disgusted')
    print(f'{num_scared/num_photos * 100}% of the guesses were scared, and {num_type_photos[2]/num_photos}% of the photos are scared')

if __name__ == "__main__":

    test_dictionary()
    

    