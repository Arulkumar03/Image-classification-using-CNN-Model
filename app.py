from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

model = load_model('model.h5')

def predict_label(image_path):
    img = image.load_img(image_path, target_size=(128, 128, 3))  # Provide your target size here
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the input shape expected by the model
    # Predict the class of the image
    prediction = model.predict(img_array)
    predicted_class = "fraudulent" if prediction > 0.5 else "non-fraudulent"  #binary classification with a threshold of 0.5
    return predicted_class
    
# routes
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template("home.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_hours():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)
        return render_template("home.html", prediction=p, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
