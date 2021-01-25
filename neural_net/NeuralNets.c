/*
	Starter by: F.J.E.
	Code by Silviu Draghic and Brian Quach
*/

#include "NeuralNets.h"


int train_1layer_net(double sample[INPUTS],int label,double (*sigmoid)(double input), double weights_io[INPUTS][OUTPUTS])
{
 /*
  *  This is your main training function for 1-layer networks. What we are doing
  *  in effect is training 10 different classifiers, each of which will learn to distinguish one of our
  *  training digits.
  * 
  *  Inputs:
  *   sample  -  Array with the pixel values for the input digit - in this case a 28x28 image (784 pixels)
  *              with values in [0-255], plus one bias term (last entry in the array) which is always 1
  *   label  -   Correct label for this digit (our target class)
  *   sigmoid -  The sigmoid function being used, which will be either the logistic function or the hyperbolic
  *              tangent. You have to implement the logistic function, but math.h provides tanh() already
  *   weights_io - Array of weights connecting inputs to output neurons, weights[i][j] is the weight from input
  *                i to output neuron j. This array has a size of 785x10.
  *
  *   Return values:
  *     An int in [0,9] corresponding to the class that your current network has chosen for this training
  *   sample.
  * 
  */

    double activations[10] = { 0 };
    feedforward_1layer(sample, sigmoid, weights_io, activations);
    
    backprop_1layer(sample, activations, sigmoid, label, weights_io);
    
    return classify_1layer(sample, label, sigmoid, weights_io);
}

int classify_1layer(double sample[INPUTS],int label,double (*sigmoid)(double input), double weights_io[INPUTS][OUTPUTS])
{
 /*
  *  This function classifies an input sample given the current network weights. It returns a class in
  *  [0,9] corresponding to the digit the network has decided is present in the input sample
  * 
  *  Inputs:
  *   sample  -  Array with the pixel values for the input digit - in this case a 28x28 image (784 pixels)
  *              with values in [0-255], plus one bias term (last entry in the array) which is always 1
  *   label  -   Correct label for this digit (our target class)
  *   sigmoid -  The sigmoid function being used, which will be either the logistic function or the hyperbolic
  *              tangent. You have to implement the logistic function, but math.h provides tanh() already
  *   weights_io - Array of weights connecting inputs to output neurons, weights[i][j] is the weight from input
  *                i to output neuron j. This array has a size of 785x10.
  *
  *   Return values:
  *     An int in [0,9] corresponding to the class that your current network has chosen for this training
  *   sample.
  * 
  */
  
    double activations[10] = { 0 };
    feedforward_1layer(sample, sigmoid, weights_io, activations);
 
    double max = -INFINITY;
    int mostLikely = -1;
    for (int o = 0; o < OUTPUTS; o++){
        if (max < activations[o]){
            max = activations[o];
            mostLikely = o;
        }
    }
    return mostLikely;
}

void feedforward_1layer(double sample[785], double (*sigmoid)(double input), double weights_io[INPUTS][OUTPUTS], double activations[OUTPUTS])
{
 /*
  *  This function performs the feedforward pass of the network's computation - it propagates information
  *  from input to output, determines the input to each neuron, and calls the sigmoid function to
  *  calculate neuron activation.
  * 
  *  Inputs:
  *    sample -      The input sample (see above for a description)
  *    sigmoid -     The sigmoid function being used
  *    weights_op -  Array of current network weights
  *    activations - Array where your function will store the resulting activation for each output neuron
  * 
  *  Return values:
  *    Your function must update the 'activations' array with the output value for each neuron
  * 
  *  NOTE - You must *scale* the input to the sigmoid function using the SIGMOID_SCALE value. Otherwise
  *         the neurons will be totally saturated and learning won't happen.
  */ 

    for (int o = 0; o < OUTPUTS; o++){
    
        double tmp = 0.0; // Get weighted sum
        
        for (int i = 0; i < INPUTS; i++){
            tmp += sample[i]*weights_io[i][o];
        }
        
        activations[o] = sigmoid(tmp*SIGMOID_SCALE); // Apply sigmoid func
    }
  
}

void backprop_1layer(double sample[INPUTS], double activations[OUTPUTS], double (*sigmoid)(double input), int label, double weights_io[INPUTS][OUTPUTS])
{
  /*
   *  This function performs the core of the learning process for 1-layer networks. It takes
   *  as input the feed-forward activation for each neuron, the expected label for this training
   *  sample, and the weights array. Then it updates the weights in the array so as to minimize
   *  error across neuron outputs.
   * 
   *  Inputs:
   * 	sample - 	Input sample (see above for details)
   *    activations - 	Neuron outputs as computed above
   *    sigmoid -	Sigmoid function in use
   *    label - 	Correct class for this sample
   *    weights_io -	Network weights
   * 
   *  You have to:
   * 		* Determine the target value for each neuron
   * 			- This depends on the type of sigmoid being used, you should think about
   * 			  this: What should the neuron's output be if the neuron corresponds to
   * 			  the correct label, and what should the output be for every other neuron?
   * 		* Compute an error value given the neuron's target
   * 		* Compute the weight adjustment for each weight (the learning rate is in NeuralNets.h)
   */
      
    double target, error;
    
    for (int o = 0; o < OUTPUTS; o++){
        if (sigmoid == &logistic){
            target = (label == o) ? 1.0 : 0.0;
            error = target - activations[o];
            for (int i = 0; i < INPUTS; i++){
                weights_io[i][o] += ALPHA*sample[i] * activations[o]*(1.0-activations[o]) * error; //logistic'(x) = logistic(x)(1-logistic(x))
                //weights_io[i][o] += ALPHA*sample[i] * (1.0-activations[o]) * error; //Wrong, but works better for some reason...
            }
        
        } else { // if (sigmoid == &tanh){
            target = (label == o) ? 1.0 : -1.0;
            error = target - activations[o];
            for (int i = 0; i < INPUTS; i++){
                weights_io[i][o] += ALPHA*sample[i] * (1.0-activations[o]*activations[o]) * error; //tanh'(x) = 1 - tanh^2(x)
            }
        }
    }   
}


//------------------------------------------------------------------------------
//--------------------------- 2 layer network ----------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------



int train_2layer_net(double sample[INPUTS],int label,double (*sigmoid)(double input), int units, double weights_ih[INPUTS][MAX_HIDDEN], double weights_ho[MAX_HIDDEN][OUTPUTS])
{
 /*
  *   This is your main training function for 2-layer networks. Now you have to worry about the hidden
  *  layer at this time. *Do not work on this until you have completed the 1-layer network*.
  * 
  *  Inputs:
  *   sample  -  Array with the pixel values for the input digit - in this case a 28x28 image (784 pixels)
  *              with values in [0-255], plus one bias term (last entry in the array) which is always 1
  *   label  -   Correct label for this digit (our target class)
  *   sigmoid -  The sigmoid function being used, which will be either the logistic function or the hyperbolic
  *              tangent. You have to implement the logistic function, but math.h provides tanh() already
  *   units   -  Number of units in the hidden layer
  *   weights_ih - Array of weights connecting inputs to hidden-layer neurons, weights_ih[i][j] is the 
  *                weight from input i to hidden neuron j. This array has a size of units 785 x 10.
  *   weights_ho - Array of weights connecting hidden-layer units to output neurons, weights_ho[i][j] is the 
  *                weight from hidden unit i to output neuron j. This array has a size of units x 10.
  *
  *   Return values:
  *     An int in [0,9] corresponding to the class that your current network has chosen for this training
  *   sample.
  * 
  */

    double h_activations[units] = { 0 };
    double activations[10] = { 0 };

    feedforward_2layer(sample, sigmoid, weights_ih, weights_ho, h_activations, activations, units);
    backprop_2layer(sample, h_activations, activations, sigmoid, label, weights_ih, weights_ho, units);
    
    return classify_2layer(sample, label, sigmoid, units, weights_ih, weights_ho);
}

int classify_2layer(double sample[INPUTS],int label,double (*sigmoid)(double input), int units, double weights_ih[INPUTS][MAX_HIDDEN], double weights_ho[MAX_HIDDEN][OUTPUTS])
{
 /*
  *   This function takes an input sample and classifies it using the current network weights. It returns
  *  an int in [0,9] corresponding to which digit the network thinks is present in the input sample.
  * 
  *  Inputs:
  *   sample  -  Array with the pixel values for the input digit - in this case a 28x28 image (784 pixels)
  *              with values in [0-255], plus one bias term (last entry in the array) which is always 1
  *   label  -   Correct label for this digit (our target class)
  *   sigmoid -  The sigmoid function being used, which will be either the logistic function or the hyperbolic
  *              tangent. You have to implement the logistic function, but math.h provides tanh() already
  *   units   -  Number of units in the hidden layer
  *   weights_ih - Array of weights connecting inputs to hidden-layer neurons, weights_ih[i][j] is the 
  *                weight from input i to hidden neuron j. This array has a size of units 785 x 10.
  *   weights_ho - Array of weights connecting hidden-layer units to output neurons, weights_ho[i][j] is the 
  *                weight from hidden unit i to output neuron j. This array has a size of units x 10.
  *
  *   Return values:
  *     An int in [0,9] corresponding to the class that your current network has chosen for this training
  *   sample.
  * 
  */
  
    double h_activations[units];
    double activations[10] = { 0 };
    
    feedforward_2layer(sample, sigmoid, weights_ih, weights_ho, h_activations, activations, units);

    double max = -INFINITY;
    int mostLikely = -1;
    for (int o = 0; o < OUTPUTS; o++){
        if (max < activations[o]){
            max = activations[o];
            mostLikely = o;
        }
    }
    return mostLikely;  
}


void feedforward_2layer(double sample[INPUTS], double (*sigmoid)(double input), double weights_ih[INPUTS][MAX_HIDDEN], double weights_ho[MAX_HIDDEN][OUTPUTS], double h_activations[MAX_HIDDEN],double activations[OUTPUTS], int units)
{
 /*
  *  Here, implement the feedforward part of the two-layer network's computation.
  * 
  *  Inputs:
  *    sample -      The input sample (see above for a description)
  *    sigmoid -     The sigmoid function being used
  *    weights_ih -  Array of current input-to-hidden weights
  *    weights_ho -  Array of current hidden-to-output weights
  *    h_activations - Array of hidden layer unit activations
  *    activations   - Array of activations for output neurons
  *    units -         Number of units in the hidden layer
  * 
  *  Return values:
  *    Your function must update the 'activations' and 'h_activations' arrays with the output values for each neuron
  * 
  *  NOTE - You must *scale* the input to the sigmoid function using the SIGMOID_SCALE value. Otherwise
  *         the neurons will be totally saturated and learning won't happen.
  */

  /**************************************************************************************************
   * Important note - scaling inputs to neurouns is critical to ensure the neurons don't saturate.
   *                  Scaling for the hidden layer works just like it did for the 1 layer net,
   * 		       simply scale your input by SIGMOID_SCALE. However, for the output layer,
   *                  the scaling factor has to be adjusted by the factor
   *                  SIGMOID_SCALE*(MAX_HIDDEN/units).
   **************************************************************************************************/
    for (int h = 0; h < units; h++){
    
        double sum = 0.0; // Get weighted sum
        
        for (int i = 0; i < INPUTS; i++){
            sum += sample[i]*weights_ih[i][h];
        }
        
        h_activations[h] = sigmoid(sum*SIGMOID_SCALE); // Apply sigmoid func
    }

    for (int o = 0; o < OUTPUTS; o++){
        double sum = 0.0;

        for (int h = 0; h < units; h++){
            sum += h_activations[h]*weights_ho[h][o];
        }

        activations[o] = sigmoid(sum*SIGMOID_SCALE*(MAX_HIDDEN/units));
    }
}

void backprop_2layer(double sample[INPUTS],double h_activations[MAX_HIDDEN], double activations[OUTPUTS], double (*sigmoid)(double input), int label, double weights_ih[INPUTS][MAX_HIDDEN], double weights_ho[MAX_HIDDEN][OUTPUTS], int units)
{
  /*
   *  This function performs the core of the learning process for 2-layer networks. It performs
   *  the weights update as discussed in lecture. Note that you require the current weights
   *  between the hidden and output layers in order to update the weights from input to hidden,
   *  however the backprop. algorithm would have you update that weight first. So mind the order
   *  of updates and keep track of what you need.
   * 
   *  Inputs:
   * 	sample - 	Input sample (see above for details)
   *    h_activations - Hidden-layer activations
   *    activations -   Output-layer activations
   *    sigmoid -	Sigmoid function in use
   *    label - 	Correct class for this sample
   *    weights_ih -	Network weights from inputs to hidden layer
   *    weights_ho -    Network weights from hidden layer to output layer
   *    units -         Number of units in the hidden layer
   * 
   *  You have to:
   * 		* Determine the target value for each neuron
   * 			- This depends on the type of sigmoid being used, you should think about
   * 			  this: What should the neuron's output be if the neuron corresponds to
   * 			  the correct label, and what should the output be for every other neuron?
   * 		* Compute an error value given the neuron's target
   * 		* Compute the weight adjustment for each weight (the learning rate is in NeuralNets.h)
   */
  
    double target;
    double error[10];

    for (int o = 0; o < OUTPUTS; o++){
        if (sigmoid == &logistic){
            target = (label == o) ? 0.9 : 0.1;
            error[o] = target - activations[o];
        }else{
            target = (label == o) ? 1.0 : -1.0;
            error[o] = target - activations[o];
        }
    }
    
    double o_weights_ho[MAX_HIDDEN][OUTPUTS];
    memcpy(o_weights_ho, weights_ho, MAX_HIDDEN*OUTPUTS*sizeof(double));

    //update weight_ho
    for (int o = 0; o < OUTPUTS; o++){
        if (sigmoid == &logistic){
            for (int h = 0; h < units; h++){
                weights_ho[h][o] += ALPHA*h_activations[h] * activations[o]*(1.0-activations[o]) * error[o]; //logistic'(x) = logistic(x)(1-logistic(x))
                //weights_ho[h][o] += ALPHA*sample[h] * activations[o]*(1.0-activations[o]) * error[o]; //logistic'(x) = logistic(x)(1-logistic(x))
            }
        
        } else {
            for (int h = 0; h < units; h++){
                weights_ho[h][o] += ALPHA*h_activations[h] * (1.0-activations[o]*activations[o]) * error[o]; //tanh'(x) = 1 - tanh^2(x)
                //weights_ho[h][o] += ALPHA*sample[h] * (1.0-activations[o]*activations[o]) * error[o]; //tanh'(x) = 1 - tanh^2(x)
            }
        }
    }

    //update weights_ih
    for(int h = 0; h < units; h++){
        for (int i = 0; i < INPUTS; i++){
            double sum = 0.0;
            for(int o = 0; o < OUTPUTS; o++){
                if (sigmoid == &logistic){
                    sum += o_weights_ho[h][o]*error[o]*activations[o]*(1.0-activations[o]);
                } else {
                    sum += o_weights_ho[h][o]*error[o]*(1.0-activations[o]*activations[o]);
                }
            }
            weights_ih[i][h] += ALPHA*sample[i]*sum;
            //weights_ih[i][h] += -ALPHA*sum;
        }
    }
}

double logistic(double input)
{
 // This function returns the value of the logistic function evaluated on input

 return 1.0/(1.0 + exp(-1*input));
}
