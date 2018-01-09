# Object-localization



# Preprocess

For creating the training and testing dataset I had to extract useful information
from the xml file. For this I used Element-Tree, which is a very famous xml parser.
Since I had to do object localization (creating bounding  boxes), I used Element-Tree
to create a dictionary containing keys as the name of the images and values as a list
containing x-min, y-min, x-max, y-max , from the annotation. This was later used for
validation by finding out the difference in predicted bounding values to the one in 
the annotation.



# CNN Model

I have used Keras for building the CNN Model. I used a sequential model which has
the embedding input layer that takes in matrices of 500x500x3. This layer is followed
by 4 Convolution layer, 2 Hidden Dense layer and an Output Dense layer. As for the 
object detection, a bounding box of 4 vertices has to be the output, I used 4 units
in the output dense layer.



# Results

I was able to get the accuracy of 92% (according to mse loss function) for the bounding box.
