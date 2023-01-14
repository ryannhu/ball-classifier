# ball-classifier
This is a neural network that can classify images of balls into 15 different categories. 
It uses image recognition techniques to analyze the visual characteristics of an image and assigns it to one of the 15 designated categories. 

This neural network was built using PyTorch and the images used for the dataset can be found 
[here](https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification).


The model was trained for 100 epochs using the train dataset and achieved an accuracy of 47.6%. 
Note that during the image preprocessing stage, the images were heavily downscaled which may have resulted in some loss of detail that could 
have affected the model's performance.

![image](https://user-images.githubusercontent.com/92134792/212484040-a7b2b371-6796-4076-ba8f-c8c03186fab4.png)

To try to train your model, run test.py to generate a model. 

To test the model on a drawing, run the draw.py file. This will open a paint-style editor. Once you are satisfied with your drawing, click the save button. 
The editor will then use the model.pth file to load the model and make a prediction. 
The results of the prediction will be output to the console for you to see.


