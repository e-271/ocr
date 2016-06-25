# ocr
Optical character recognition project.

Learn.py
    Contains an implementation of logistic regression to learn to identify images.
    Run this module to run the demos.

Learn.MNIST_demo()
    Uses the MNIST dataset, which is a large set of handwritten digits 0-9, and learns to how read them. Or you can use already-trained parameters saved in mnist.p by setting retrain=False.
    Once it has trained some parameters, it will prompt you to draw a character on-screen, and try to predict what you drew.

Learn.custom_demo()
    (is not currently working well, but I'll fix it)
    Lets you train the program to distingush between images you draw. Doesn't use the MNIST dataset, though it could.
    Repeatedly prompts for user input, then asks you for a label.
    Every time you input an image it is added to the dataset, with its label, and prediction parameters are retrained.
    Once you're done there's an option to save the dataset you input to disk at the end.
    If you want to load old data sets saved to disk, set use_old_data=True.

Draw.py
    Used for inputting images from the screen.
    Designed to be most naturally usable on a touchscreen, but should work with any mouse type device.

Data.py
    Holds features, labels, various metadata. mnist.p and custom_data.p are pickled Data objects.
