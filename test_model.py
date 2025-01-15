import numpy as np
from PIL import Image
import os
import joblib

def test_dictionary():
    data = {'image':[], 'label':[]}
    num_correct, num_photos, num_happy, num_sad, num_angry, num_disgusted, num_surprised, num_scared, num_neutral = 0, 0, 0, 0, 0, 0, 0, 0, 0
    num_correct_happy, num_correct_sad, num_correct_angry, num_correct_disgusted, num_correct_surprised, num_correct_scared, num_correct_neutral = 0, 0, 0, 0, 0, 0, 0

    current = os.path.dirname(os.path.abspath(__file__))
    relative_training_path = os.path.join(current, "dataset", "test")
    for image_file in os.listdir(os.path.join(relative_training_path, "angry")):
        path = os.path.join(relative_training_path, "angry", image_file)
        with Image.open(path) as image:
            flat_image = np.array(image).flatten()
            data['image'].append(flat_image)
            data['label'].append('angry')
        num_angry += 1
    
    for image_file in os.listdir(os.path.join(relative_training_path, "disgusted")):
        path = os.path.join(relative_training_path, "disgusted", image_file)
        with Image.open(path) as image:
            flat_image = np.array(image).flatten()
            data['image'].append(flat_image)
            data['label'].append('disgusted')
        num_disgusted += 1

    for image_file in os.listdir(os.path.join(relative_training_path, "fearful")):
        path = os.path.join(relative_training_path, "fearful", image_file)
        with Image.open(path) as image:
            flat_image = np.array(image).flatten()
            data['image'].append(flat_image)
            data['label'].append('fearful')
        num_scared += 1

    for image_file in os.listdir(os.path.join(relative_training_path, "happy")):
        path = os.path.join(relative_training_path, "happy", image_file)
        with Image.open(path) as image:
            flat_image = np.array(image).flatten()
            data['image'].append(flat_image)
            data['label'].append('happy')
        num_happy += 1

    for image_file in os.listdir(os.path.join(relative_training_path, "neutral")):
        path = os.path.join(relative_training_path, "neutral", image_file)
        with Image.open(path) as image:
            flat_image = np.array(image).flatten()
            data['image'].append(flat_image)
            data['label'].append('neutral')
        num_neutral += 1

    for image_file in os.listdir(os.path.join(relative_training_path, "sad")):
        path = os.path.join(relative_training_path, "sad", image_file)
        with Image.open(path) as image:
            flat_image = np.array(image).flatten()
            data['image'].append(flat_image)
            data['label'].append('sad')
        num_sad += 1

    for image_file in os.listdir(os.path.join(relative_training_path, "surprised")):
        path = os.path.join(relative_training_path, "surprised", image_file)
        with Image.open(path) as image:
            flat_image = np.array(image).flatten()
            data['image'].append(flat_image)
            data['label'].append('surprised')
        num_surprised += 1

    model = joblib.load("emotion_detector.pkl")

    for i in range(len(data['image'])):
        img = data['image'][i]
        reshaped_img = np.array(img.reshape(1, -1))
        prediction = model.predict(reshaped_img)
        if prediction == data['label'][i]:
            num_correct += 1
            if prediction == "angry":
                num_correct_angry += 1
            if prediction == "happy":
                num_correct_happy += 1
            if prediction == "sad":
                num_correct_sad += 1
            if prediction == "surprised":
                num_correct_surprised += 1
            if prediction == "neutral":
                num_correct_neutral += 1
            if prediction == "fearful":
                num_correct_scared += 1
            if prediction == "disgusted":
                num_correct_disgusted += 1
        num_photos += 1

    percent_accuracy = num_correct/num_photos * 100
    print(f'The model is {percent_accuracy}% accurate overall.')
    print(f'{num_correct_angry/num_angry * 100}% accurate for angry, getting {num_correct_angry} correct out of {num_angry}')
    print(f'{num_correct_sad/num_sad * 100}% accurate for sad, getting {num_correct_sad} correct out of {num_sad}')
    print(f'{num_correct_happy/num_happy * 100}% accurate for happy, getting {num_correct_happy} correct out of {num_happy}')
    print(f'{num_correct_surprised/num_surprised * 100}% accurate for surprised, getting {num_correct_surprised} correct out of {num_surprised}')
    print(f'{num_correct_neutral/num_neutral * 100}% accurate for neutral, getting {num_correct_neutral} correct out of {num_neutral}')
    print(f'{num_correct_disgusted/num_disgusted * 100}% accurate for disgusted, getting {num_correct_disgusted} correct out of {num_disgusted}')
    print(f'{num_correct_scared/num_scared * 100}% accurate for scared, getting {num_correct_scared} correct out of {num_scared}')

if __name__ == "__main__":
    test_dictionary()
    

    