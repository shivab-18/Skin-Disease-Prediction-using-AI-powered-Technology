from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import tensorflow as tf
import cv2
import os
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model and label encoder
model = tf.keras.models.load_model('final_cnn_ga_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Prescription info
prescriptions = {
    'BA-cellulitis': {
        'medicine': 'Amoxicillin or Cephalexin',
        'soap': 'Antibacterial soap',
        'usage': 'Take twice daily for 7 days',
    },
    'BA-impetigo': {
        'medicine': 'Mupirocin ointment',
        'soap': 'Mild antiseptic soap',
        'usage': 'Apply 3 times daily for 5 days',
    },
    'FU-athlete-foot': {
        'medicine': 'Clotrimazole cream',
        'soap': 'Antifungal foot wash',
        'usage': 'Apply after washing feet, twice daily',
    },
    'FU-nail-fungus': {
        'medicine': 'Terbinafine tablets',
        'soap': 'Antifungal soap',
        'usage': 'Once daily for 6 weeks',
    },
    'FU-ringworm': {
        'medicine': 'Ketoconazole cream',
        'soap': 'Antifungal soap',
        'usage': 'Apply 2 times a day for 2 weeks',
    },
    'PA-cutaneous-larva-migrans': {
        'medicine': 'Albendazole',
        'soap': 'Normal soap',
        'usage': 'Daily dose for 5 days',
    },
    'VI-chickenpox': {
        'medicine': 'Acyclovir',
        'soap': 'Oatmeal bath soap',
        'usage': '5 times daily for 5 days',
    },
    'VI-shingles': {
        'medicine': 'Valacyclovir',
        'soap': 'Mild antibacterial soap',
        'usage': '3 times daily for 7 days',
    }
}

# Preprocess input image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (100, 100))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    prescription = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            image_path = save_path

            img = preprocess_image(save_path)
            pred = model.predict(img)
            class_index = np.argmax(pred, axis=1)[0]
            class_label = label_encoder.inverse_transform([class_index])[0]
            prediction = class_label
            prescription = prescriptions.get(class_label)

    return render_template('index.html', prediction=prediction, image=image_path, prescription=prescription)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        session['profile'] = {
            'name': username,
            'email': f'{username}@example.com',
            'phone': 'Not Provided',
            'age': 'Not Provided',
            'gender': 'Not Provided'
        }
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        session['profile'] = {
            'name': request.form['name'],
            'email': request.form['email'],
            'phone': request.form['phone'],
            'age': request.form['age'],
            'gender': request.form['gender']
        }
        return redirect(url_for('index'))
    return render_template('signup.html')

@app.context_processor
def inject_profile():
    return dict(profile=session.get('profile'))

if __name__ == '__main__':
    app.run(debug=True)
