# EmojiEmotion
Final Project repository for computer vision - Spring 2020

Before running the program, make sure all the correct dependencies are installed on your local machine by running the following command in your virtual environment: 

pip install -r requirements.txt
 
After all the dependencies have been installed, you are ready to go! You can run our live emotion recognition application by navigating into the code directory and running the LiveEmotion_WebCam.py file. By default the program runs with our model, however you can compare our model to another model from online by setting the --OnlineModel flag to "true" when you run the program. 

Example 1: python LiveEmotion_WebCam.py 
Example 2: python LiveEmotion_WebCam.py --OnlineModel true

It's as simple as that! 

If you want to see how our live emotion recognition program runs in terms of computational time, you can turn off the predictions by hitting the "n" key while the live video feed is open. This will clear the image of any emojis. You can turn the live predictions back on by hitting the "y" key. These keys can be hit indefinitely to toggle between modes. 

If you want to check out our model architecture, you can open the EmotionLearning.py file and investigate how we set up our neural network. 