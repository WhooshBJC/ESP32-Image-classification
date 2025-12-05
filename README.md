# ESP32-Image-classification
ResNet Neural Network model implemented in pytorch. This set up to be used by ESP32S3 cam to stream live image and send the processed image back to a website

#Training the model in Python
The ResNet Architecture is in ResNetArchitecture2.py. You may use it to fine tune your ResNet Model

#Before training the model, run DataLoader.py to troubleshoot importing data into the program

#Once you have successfully ran DataLoader.py, you may run train.py to train your model. The program should save the weight of the model once it meets an validating accuracy of 85% or above

#Setting up the ESP32
Import cam.cpp to main and configure it as you should

#Using the ESP32 to predict and output into a webpage
Run backend_processing3.py, then connect the ESP32 that has been set up to a power source, then run test1.html on your desired browser.
