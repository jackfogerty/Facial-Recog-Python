from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import joblib
import pandas as pd
import numpy as np
import os
from PIL import Image

#this function converts the collection of images and their corresponding emotions to a
#dictionary holding both, then to a pandas dataframe which can then be used to train the model
def generate_data(num_samples = 1000):
    data = {'image':[], 'label':[]}
    current = os.path.dirname(os.path.abspath(__file__))
    relative_training_path = os.path.join(current, "dataset", "train")
    for image_file in os.listdir(os.path.join(relative_training_path, "angry")):
        path = os.path.join(relative_training_path, "angry", image_file)
        with Image.open(path) as image:
            resized_image = image.resize((48, 48)).convert('L')
            flat_image = np.array(resized_image).flatten()
            data['image'].append(flat_image)
            data['label'].append('angry')
    
    for image_file in os.listdir(os.path.join(relative_training_path, "disgusted")):
        path = os.path.join(relative_training_path, "disgusted", image_file)
        with Image.open(path) as image:
            resized_image = image.resize((48, 48)).convert('L')
            flat_image = np.array(resized_image).flatten()
            data['image'].append(flat_image)
            data['label'].append('disgusted')

    for image_file in os.listdir(os.path.join(relative_training_path, "fearful")):
        path = os.path.join(relative_training_path, "fearful", image_file)
        with Image.open(path) as image:
            resized_image = image.resize((48, 48)).convert('L')
            flat_image = np.array(resized_image).flatten()
            data['image'].append(flat_image)
            data['label'].append('fearful')

    for image_file in os.listdir(os.path.join(relative_training_path, "happy")):
        path = os.path.join(relative_training_path, "happy", image_file)
        with Image.open(path) as image:
            resized_image = image.resize((48, 48)).convert('L')
            flat_image = np.array(resized_image).flatten()
            data['image'].append(flat_image)
            data['label'].append('happy')

    for image_file in os.listdir(os.path.join(relative_training_path, "neutral")):
        path = os.path.join(relative_training_path, "neutral", image_file)
        with Image.open(path) as image:
            resized_image = image.resize((48, 48)).convert('L')
            flat_image = np.array(resized_image).flatten()
            data['image'].append(flat_image)
            data['label'].append('neutral')

    for image_file in os.listdir(os.path.join(relative_training_path, "sad")):
        path = os.path.join(relative_training_path, "sad", image_file)
        with Image.open(path) as image:
            resized_image = image.resize((48, 48)).convert('L')
            flat_image = np.array(resized_image).flatten()
            data['image'].append(flat_image)
            data['label'].append('sad')

    for image_file in os.listdir(os.path.join(relative_training_path, "surprised")):
        path = os.path.join(relative_training_path, "surprised", image_file)
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
    #and a set containing the above for testing.
    training_data = generate_data()
    img_train, img_test, label_train, label_test = train_test_split(training_data['image'],
                                                                    training_data['label'], 
                                                                    test_size = 0.15, 
                                                                    random_state = 4)
    
    #creates the RandomForest algorithm object
    #model = Pipeline([('classifier, RandomForestObject(n_estimators=300))])
    model = Pipeline([('classifier', MLPClassifier(solver= 'adam', hidden_layer_sizes=(512, 256, 128, 64, 32, 16), max_iter=1000, random_state=3))])
    img_train_flat = [np.array(img).ravel() for img in img_train]
    img_test_flat = [np.array(img).ravel() for img in img_test]
    
    #training the model
    model.fit(img_train_flat, label_train)

    #saving model to file
    model_filename = 'emotion_detector.pkl'
    joblib.dump(model, model_filename)
    print("Trained model saved to " + model_filename)

    #load model from file
    model = joblib.load(model_filename)

    #print accuracy of model
    accuracy = model.score(img_test_flat, label_test)
    print(f'{accuracy} % accurate')

    

if __name__ == "__main__":
    create_model()


    
