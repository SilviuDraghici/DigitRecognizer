# DigitRecognizer

This is a neural net written in C from scratch.
It was trained to recongize hand written numbers (0-9).

The input images are 28x28 greyscale pictures of hand written digits from a public database.
There is also a bias term for a total of 785 inputs to the network.

There are 10 output neurons, where each neuron is responsible for recognizing one of 0 to 9.
Each neuron outputs a double between 0 and 1, representing the neuron's confidence 
that the input is a picture of its digit.
The highest confidence neuron determines the chosen digit for the network.

The network can be configred to use a hidden layer with a configurable number of neurons.

The activation function is also configurable: Logistic and Hyperbolic  Tangent are the available options.

---

This is a visualization of the network learning its weights.  
![TV]

Each box is a different output neuron. The top left is the 0 recognizer and the bottom right is the 9 recognizer.

The bias term is not represented.
There is no hidden layer in this case.

The weights are assigned randomly in the begining. 

If a weight is dark, the neuron seeing that pixel filled in on the input image decreases its confidence in recongnizing the neuron's digit. If a weight is bright it means seeing that pixel filled in on the input image increases the neuron's confidence in recognizing its digit. Grey weights mean that the neuron doesn't pay much attention to those input pixels.

---
Training Results:

1-Layer Network:

	Logistic activation function: 
	
           Training data (last training round):
                Digit 0, correct classification rate=0.944444
                Digit 1, correct classification rate=0.969283
                Digit 2, correct classification rate=0.806517
                Digit 3, correct classification rate=0.826742
                Digit 4, correct classification rate=0.876827
                Digit 5, correct classification rate=0.718468
                Digit 6, correct classification rate=0.943867
                Digit 7, correct classification rate=0.914062
                Digit 8, correct classification rate=0.755230
                Digit 9, correct classification rate=0.767206
                Average correct classification rate: 0.852265
           
           Testing data:
                Digit 0, correct classification rate=0.968367
                Digit 1, correct classification rate=0.967401
                Digit 2, correct classification rate=0.808140
                Digit 3, correct classification rate=0.873267
                Digit 4, correct classification rate=0.879837
                Digit 5, correct classification rate=0.723094
                Digit 6, correct classification rate=0.925887
                Digit 7, correct classification rate=0.884241
                Digit 8, correct classification rate=0.806982
                Digit 9, correct classification rate=0.823588
                Average correct classification rate: 0.866080


	Hyperbolic Tangent activation function:

           Training data (last training round):
                Digit 0, correct classification rate=0.956612
                Digit 1, correct classification rate=0.969259
                Digit 2, correct classification rate=0.884232
                Digit 3, correct classification rate=0.891144
                Digit 4, correct classification rate=0.924335
                Digit 5, correct classification rate=0.803571
                Digit 6, correct classification rate=0.957090
                Digit 7, correct classification rate=0.922348
                Digit 8, correct classification rate=0.856287
                Digit 9, correct classification rate=0.835443
                Average correct classification rate: 0.900032

            Testing data:
                Digit 0, correct classification rate=0.982653
                Digit 1, correct classification rate=0.970044
                Digit 2, correct classification rate=0.875000
                Digit 3, correct classification rate=0.903960
                Digit 4, correct classification rate=0.914460
                Digit 5, correct classification rate=0.819507
                Digit 6, correct classification rate=0.953027
                Digit 7, correct classification rate=0.917315
                Digit 8, correct classification rate=0.843943
                Digit 9, correct classification rate=0.872151
                Average correct classification rate: 0.905206

	Note: 
		The Hyperbolic Tangent produced better classification results and reached
		above %80 correct classification much faster.

2-Layer Network with Hyperbolic Tangent activation function:

	150 hidden neurons:         
           
           Training data (last training round):
                Digit 0, correct classification rate=0.977083
                Digit 1, correct classification rate=0.990017
                Digit 2, correct classification rate=0.940695
                Digit 3, correct classification rate=0.949393
                Digit 4, correct classification rate=0.951515
                Digit 5, correct classification rate=0.892779
                Digit 6, correct classification rate=0.969325
                Digit 7, correct classification rate=0.970205
                Digit 8, correct classification rate=0.921162
                Digit 9, correct classification rate=0.936975
                Average correct classification rate: 0.949915
           Testing data:
                Digit 0, correct classification rate=0.975510
                Digit 1, correct classification rate=0.964758
                Digit 2, correct classification rate=0.891473
                Digit 3, correct classification rate=0.865347
                Digit 4, correct classification rate=0.900204
                Digit 5, correct classification rate=0.869955
                Digit 6, correct classification rate=0.962422
                Digit 7, correct classification rate=0.911479
                Digit 8, correct classification rate=0.938398
                Digit 9, correct classification rate=0.917740
                Average correct classification rate: 0.919729


	10 hidden neurons:
	
            Training data (last training round):
                Digit 0, correct classification rate=0.933594
                Digit 1, correct classification rate=0.951957
                Digit 2, correct classification rate=0.824017
                Digit 3, correct classification rate=0.774859
                Digit 4, correct classification rate=0.902041
                Digit 5, correct classification rate=0.757447
                Digit 6, correct classification rate=0.904762
                Digit 7, correct classification rate=0.864000
                Digit 8, correct classification rate=0.874743
                Digit 9, correct classification rate=0.832244
                Average correct classification rate: 0.861966

            Testing data:
                Digit 0, correct classification rate=0.925510
                Digit 1, correct classification rate=0.962115
                Digit 2, correct classification rate=0.767442
                Digit 3, correct classification rate=0.801980
                Digit 4, correct classification rate=0.869654
                Digit 5, correct classification rate=0.747758
                Digit 6, correct classification rate=0.907098
                Digit 7, correct classification rate=0.877432
                Digit 8, correct classification rate=0.839836
                Digit 9, correct classification rate=0.776016
                Average correct classification rate: 0.847484


	3 hidden neurons:
	
            Training data (last training round):
                Digit 0, correct classification rate=0.766595
                Digit 1, correct classification rate=0.986509
                Digit 2, correct classification rate=0.087379
                Digit 3, correct classification rate=0.027197
                Digit 4, correct classification rate=0.053061
                Digit 5, correct classification rate=0.396476
                Digit 6, correct classification rate=0.308511
                Digit 7, correct classification rate=0.615242
                Digit 8, correct classification rate=0.011976
                Digit 9, correct classification rate=0.329960
                Average correct classification rate: 0.358290
            
            Testing data:
                Digit 0, correct classification rate=0.065306
                Digit 1, correct classification rate=0.992952
                Digit 2, correct classification rate=0.000000
                Digit 3, correct classification rate=0.003960
                Digit 4, correct classification rate=0.026477
                Digit 5, correct classification rate=0.000000
                Digit 6, correct classification rate=0.864301
                Digit 7, correct classification rate=0.763619
                Digit 8, correct classification rate=0.023614
                Digit 9, correct classification rate=0.000000
                Average correct classification rate: 0.274023
				
	Note:
		Increasing the number of hidden layers increases
        classification accuracy albeit with strongly diminishing returns.

Best Performance Achieved:

Hyperbolic Tangent activation function and 250 hidden neurons produced
a classification rate on testing data of:
    92.0047%


[TV]: https://github.com/SilviuDraghici/DigitRecognizer/raw/main/weights/Training_Visualization.gif
