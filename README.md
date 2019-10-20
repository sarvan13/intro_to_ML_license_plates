# intro_to_ML_lisence_plates
Using machine learning (CNN) to read letters and numbers off of a randomly generated image of a lisence plate.

After training, the code simply compares the expected plate number against the output and displays a list of the expected vs output along with a total percentage of how accurate the model is.

This small project is mostly just me navigating machine learning and convolution neural networks for the first time through keras. I am currently trainig the model locally, but the code can be easily transefered to google collaboratory.

The depenencies can be downloaded as follows:
```
sudo apt-get install python-PIL
pip install numpy
pip install matplotlib
pip install opencv
pip install keras
pip install tensorflow
```

In order to train the model, we must first generate random lisence plate images. Running:
```
python plate_generator.py
```
This will generate 15 random lisence plates. I would suggest running this command a few times or changing the code in **plate_generator.py** to generate enough data to train the model. 

After data has been generated train the model by running:
```
python plate_seperator.py
```
It should output 2 plots (might need to close one to view the next).

Now we can generate more lisence plates, comment out the last 2 lines in **plate_seperator.py** and add the line:
**test()**

Finally run: 
```
python plate_seperator.py
```


