# Face-Recognition
## About The Project
This face recognition project uses mediapipe and FaceNet to detect and recognize faces that it sees in a video stream or a static image. 
Media pipe uses a simple and fast SSD network to detect faces and then, the faces go through the FaceNet network and will be transformed into 128-dimensional vectors.
We then simply use these vectors to identify the faces we previously passed through the network.

## Packages 
To run this project you need TensorFlow >= 2.2 and mediapipe installed. Go ahead and create a virtual environment and install the following packages:
```bash
mediapipe==0.9.0.1
numpy==1.24.1
opencv_python==4.6.0.66
scikit_learn==1.2.1
scipy==1.9.0
tensorflow==2.11.0
```
Alternatively, you can use the requirement.txt file to install the necessary packages:

```sh
pip install -r requirements.txt
```

## Usage
### Creating the database
First of all, you need to create your database of faces. Create a folder called `faces_database`. Inside this folder, you need to create a folder person and put at least 
one picture in it. the picture should be a cropped picture of the person's pace. For example, I create a folder called 'Daniel' and put a picture of my face inside it.
Alternatively, you can use the `main.py` to get a cropped picture of a face:
```sh
python3 main.py --use_vid 0 --pic_path [path to your picture]
```
This command would find and save faces in your picture for you
### Creating encodings from pictures
After you created the faces folders, use the `Face2Encoding.py` to create a dictionary of persons and their respective encoding.
```sh
python3 Face2Encoding.py
```
### Run using a webcam or a static image
Finally, you can use the `main.py` file to detect faces using your webcam
```sh
python3 Face2Encoding.py
```

## Acknowledgments
Huge shot out to Rj4jn for implementing FaceNet in TensorFlow 2.x <br />
Originally FaceNet was trained on TensorFlow 1.3.0 old pre-trained models were incompatible with the new TensorFlow
* [@R4j4n](https://github.com/R4j4n)



## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
