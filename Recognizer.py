import cv2
# from keras.models import model_from_json
from tensorflow.keras.models import load_model
import numpy as np

def img_2_encodings(image_path, model):

    img = cv2.imread(image_path)
    img = np.expand_dims(img, axis=0)

    embedding = model.embeddings(img)
    return embedding

# encoding_orig = img_2_encodings('faces_database/Daniel/0.jpg', model)
# encoding_1 = img_2_encodings('faces_database/Daniel/1.jpg', model)
# encoding_2 = img_2_encodings('faces_database/Daniel/2.jpg', model)

# json_file = open('keras-facenet-h5/model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
model = load_model("facenet_keras.h5")
print(model.summary())


# from facenet_pytorch import InceptionResnetV1
#
# from PIL import Image
#
# resnet = InceptionResnetV1(pretrained='vggface2').eval()
# img = Image.open('faces_database/Daniel/0.jpg')
# img_embedding = resnet(img.unsqueeze(0))
# print(img_embedding, '\n', img_embedding.shape)