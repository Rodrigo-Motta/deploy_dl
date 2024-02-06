from flask import Flask, render_template

# Imports
import io
import platform
from PIL import Image
from urllib.request import urlopen

import flasgger
from flask_restful import Api
from flask_restful import Resource, fields, marshal
from flask import Flask, render_template_string, request, redirect


import torch
from torchvision import models
import torchvision.transforms as transforms



# Load a pre-trainied DenseNet model from torchvision.models
model = models.densenet121(pretrained=True)

# Switch the model to evaluation mode
model.eval()

# Load the class labels from a file
class_labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
class_labels = urlopen(class_labels_url).read().decode("utf-8").split("\n")

# Define the transformation of the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(model, transform, image, class_labels):
  # Transform the image and convert it to a tensor
  image_tensor = transform(image).unsqueeze(0)

  # Pass the image through the model
  with torch.no_grad():
    output = model(image_tensor)


  # Select the class with the higherst probability and look up the name
  m = torch.nn.Softmax(dim=1)
  class_prob = round(float(m(output).max()),3)*100
  class_id = torch.argmax(output).item()
  class_name = class_labels[class_id]

  # Return the class name
  return str(class_name) + " - Confidence: " + str(class_prob) +  '%'


index_template = """
<html>
    <head>
        <!-- Load vue.js and axois.js -->
        <script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/axios/0.21.1/axios.min.js"></script>
    </head>
    <body>
        <!-- The APP UI -->
        <div id="app" style="width: 50%; margin: 200px auto">
            <form id="imageForm" enctype="multipart/form-data" method="POST" style="text-align: center; display: block">
                <label for="imageFile">Select image to classify:</label
                ><input id="imageFile" name="file" type="file" style="margin-left: 10px" />

                <img v-if="image" :src="image" style="width: 250px; display: block; margin: 50px auto 10px" />
                <div v-if="prediction" style="font-size: 32px; font-weight: bold; text-align: center">
                    {{ prediction }}
                </div>
                <input v-if="image" type="submit" value="Classify Image" style="margin: 20px 20px" />
            </form>
        </div>

        <script>
            <!-- The Vue application -->
            var app = new Vue({
                el: "#app",
                data() {
                    return {
                        image: null,
                        prediction: null,
                    };
                },
            });

            <!-- Calling the predict API when the form is submitted -->
            document.getElementById("imageForm").addEventListener("submit", (e) => {
                axios
                    .post("/predict", new FormData(document.getElementById("imageForm")), {
                        headers: {
                            "Content-Type": "multipart/form-data",
                        },
                    })
                    .then((response) => (app.prediction = response.data));

                e.preventDefault();
            });

            <!-- Display the selected image -->
            document.getElementById("imageFile").addEventListener("change", (e) => {
                const [file] = document.getElementById("imageFile").files;
                if (file) {
                    app.image = URL.createObjectURL(file);
                }
            });
        </script>
    </body>
</html>
"""

app = Flask(__name__)

# Serve the template with the interactive UI
@app.route("/")
def home():
  return index_template


# Classification API
@app.route("/predict", methods=['POST'])
def predict_api():
  # Fetch the image from the request and convert it to a Pillow image
  image_file = request.files['file']
  image_bytes = image_file.read()
  image = Image.open(io.BytesIO(image_bytes))

  # Predict the class from the image
  class_name = predict(model, transform, image, class_labels)

  # Return the result
  return class_name


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

