// Simple neural nets: network
// (c) Alasdair Turner 2009

// Free software: you can redistribute this program and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This class is for the neural network,
// which is hard coded with three layers:
// input, hidden and output

class Network
{
  // this network is hard coded to only have one hidden layer
  Neuron [] m_input_layer; 
  Neuron [] m_hidden_layer;
  Neuron [] m_output_layer;
  // create a network specifying numbers of inputs, hidden layer neurons
  // and number of outputs, e.g. Network(4,4,3)
  Network(int inputs, int hidden, int outputs)
  {
    m_input_layer = new Neuron [inputs];
    m_hidden_layer = new Neuron [hidden];
    m_output_layer = new Neuron [outputs];
    // set up the network topology
    for (int i = 0; i < m_input_layer.length; i++) {
      m_input_layer[i] = new Neuron();
    }
    // route the input layer to the hidden layer
    for (int j = 0; j < m_hidden_layer.length; j++) {
      m_hidden_layer[j] = new Neuron(m_input_layer);
    }
    // route the hidden layer to the output layer
    for (int k = 0; k < m_output_layer.length; k++) {
      m_output_layer[k] = new Neuron(m_hidden_layer);
    }  
  }
  
  float[] respond(float [] inputs)
  {
    float [] responses = new float [m_output_layer.length];
    // feed forward
    // simply set the input layer to display the inputs
    for (int i = 0; i < m_input_layer.length; i++) {
      m_input_layer[i].m_output = inputs[i];
    }
    // now feed forward through the hidden layer
    for (int j = 0; j < m_hidden_layer.length; j++) {
      m_hidden_layer[j].respond();
    }
    // and finally feed forward to the output layer
    for (int k = 0; k < m_output_layer.length; k++) {
      responses[k] = m_output_layer[k].respond();
    }
    // now check the best response:
    /*int response = -1;
    float best = max(responses);
    for (int a = 0; a < responses.length; a++) {
      if (responses[a] == best) {
        response = a;
      }
    }*/
    return responses;
  }
  void train(float [] outputs)
  {
    // adjust the output layer
    for (int k = 0; k < m_output_layer.length; k++) {
      m_output_layer[k].finderror(outputs[k]);
      m_output_layer[k].train();
    }
    // propagate back to the hidden layer
    for (int j = 0; j < m_hidden_layer.length; j++) {
      m_hidden_layer[j].train();
    }
    // the input layer doesn't learn:
    // it is simply the inputs
  }
  void draw()
  {
    // note, this draw is hard-coded for Network(196,49,10)
    // which reflects my use of the MNIST database of handwritten digits
    for (int i = 0; i < m_input_layer.length; i++) {
      pushMatrix();
      translate(width * 0.58 + (i%36) * width / 100.0 + width * 0.00000001, (i/36) * height / 50.0 + height * 0.31);
      m_input_layer[i].draw(true);
      popMatrix();
    }
    for (int j = 0; j < m_hidden_layer.length; j++) {
      pushMatrix();
      translate(width * 0.498 + (j%50) * width / 100.0 + width * 0.00000001, (j/50) * height / 50.0 + height * 0.1);
      m_hidden_layer[j].draw(false);
      popMatrix();
    }
    // this is slightly tricky -- I've switched the order so the output
    // neurons are arrange 1,2,3...8,9,0 rather than 0,1,2...7,8,9
    // (that's what the (k+9) % 10 is doing)
    for (int k = 0; k < m_output_layer.length; k++) {
      pushMatrix();
      translate(width * 0.60 + (k%32) * width / 100.0 + width * 0.00001, (k/32) * height / 50.0 + height * 0.011);
      m_output_layer[k].draw(true);
      popMatrix();
    }
  }
}