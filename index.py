from flask import Flask,render_template,request, url_for, redirect, send_from_directory
import static.py.functions as f
from werkzeug.utils import secure_filename
#########MODELO FINAL########################
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from os import listdir
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB


import pickle
import cv2
from skimage.feature import hog
from skimage import io, color, measure
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def extract_hog(image):
    features, hog_image = hog(image, visualize=True)
    return features

def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors


def extract_hu_moments(image):
    moments = measure.moments_hu(image)
    return moments

def load_and_preprocess_images(image, target_size):
        image = cv2.imread(image)
        image=  cv2.resize(image, target_size, interpolation = cv2.INTER_AREA)
        image=  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image= cv2.convertScaleAbs(image, alpha=1.5, beta=10)
        image= cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return image


def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors


num_clusters=100

def assign_to_visual_words(sift_features, kmeans_model):
    visual_words = kmeans_model.predict(sift_features)
    hist, _ = np.histogram(visual_words, bins=np.arange(num_clusters + 1))
    return hist



target_size = (128, 128)


modelo_svm= pickle.load(open('static/bin/modelo_svm_hog.pkl', 'rb'))
modelo_tree= pickle.load(open("static/bin/modelo_tree_hu.pkl", 'rb'))
modelo_random= pickle.load(open("static/bin/modelo_random_sift.pkl", 'rb'))
bow= pickle.load(open("static/bin/sift_bow.pkl", 'rb'))

# Config
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/files"


# Routes
@app.route('/', methods=['GET','POST'])
def index():    
    return render_template('layout.html')

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        file = request.files['uploadFile']
        filename = secure_filename(file.filename)
        if file and f.allowed_file(filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file=os.path.join(app.config["UPLOAD_FOLDER"],filename)

            imagen=load_and_preprocess_images(file, target_size)
            imagen2=load_img(file, target_size=target_size)

            img_array = img_to_array(imagen2)
            img_array = np.expand_dims(img_array, axis=0)

            batch_size = 263
            img_array_batch = np.repeat(img_array, batch_size, axis=0)

            hu_momento=extract_hu_moments(imagen)
            hog=extract_hog(imagen)
            sift_caracteristicas=assign_to_visual_words(extract_sift_features(imagen),bow)

            model_path = 'static/bin/model_file.h5'
            model_keras = load_model(model_path)
            predictions = model_keras.predict(img_array_batch)

            predicted_class_index = np.argmax(predictions[0])

            # You may have a list of class labels to map the index to a human-readable class name
            class_labels = ['class1', 'class2', 'class3']  # Replace with your actual class labels

            # Get the predicted class label
            predicted_class_label = class_labels[predicted_class_index]

            #Predicciones
            print("Modelo Keras")
            print("Predicted class:", predicted_class_label)
            print("Modelo SVM")
            print(modelo_svm.predict([hog]))
            print("Modelo Tree")
            print(modelo_tree.predict([hu_momento]))
            print("Modelo Random Forest")
            print(modelo_random.predict([sift_caracteristicas]))          
            os.remove(file) 
            #resp_analisis = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
            #return send_from_directory(directory=resp_analisis, filename='analisis.xlsx')
            return render_template('upload.html', show_predictions_modal=predicted_class_label)

        else:
            print("Tipo de archivo no permitido!")
            return redirect(url_for('upload'))
    else:
        return render_template('upload.html')

@app.route("/about", methods=['GET','POST'])
def about():
    if request.method == 'POST':
        return render_template('about.html')
    else:
        return render_template('about.html')
    
if __name__ == '__main__':
    app.run(debug=False)
