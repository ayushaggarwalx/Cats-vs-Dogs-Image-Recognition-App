import gradio as gr
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Loading saved model
model = tf.keras.models.load_model('cats_vs_dogs.h5', custom_objects={'KerasLayer': hub.KerasLayer})

def predict(input_image):
    try:
        # Convert PIL Image to Numpy array
        input_image = img_to_array(input_image)
        # Resize the Numpy array
        input_image = np.resize(input_image, (224, 224, 3))
        input_image = np.array(input_image).astype(np.float32) / 255.0
        input_image = np.expand_dims(input_image, axis=0) 


        # Making prediction
        prediction = model.predict(input_image)

        # Postprocess prediction
        labels = ['Cat', 'Dog']
        threshold = 0.5  # threshold for classifying as 'Dog'
        predicted_class = 'Dog' if prediction[0] > threshold else 'Cat'
        prediction_probability = prediction[0] if predicted_class == 'Dog' else 1 - prediction[0]

        cat_emoji = "\U0001F408"  # Cat emoji
        dog_emoji = "\U0001F415"  # Dog emoji

        selected_emoji = dog_emoji if predicted_class == 'Dog' else cat_emoji

        # Combine the predicted class and the probability into a single string
        output = f"{selected_emoji} {predicted_class}"

        return output
    except Exception as e:
        return str(e)

examples = ["dog.jpg",
            "cat.jpg"]

# Creating Gradio interface
iface = gr.Interface(
    fn=predict, 
    inputs=gr.inputs.Image(shape=(224, 224)), 
    outputs="text",
    title = '🐱 x 🐶 Image Recognition Application',
    description="""<br> This model was trained to predict whether an image contains a cat or a dog. <br> 
    <br> You can see how this model was trained on the following <a href = "https://github.com/ayushaggarwalx/Cats-vs-Dogs-Image-Recognition-App">GitHub</a>.
    <br>Upload a photo to see the how the model predicts!""",
    examples = examples
)

iface.launch()