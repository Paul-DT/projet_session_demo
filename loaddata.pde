// Simple neural nets: load data
// (c) Alasdair Turner 2009

// Free software: you can redistribute this program and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// this uses the MNIST database of handwritten digits
// http://yann.lecun.com/exdb/mnist/ (accessed 04.06.09)
// Yann LeCun and Corinna Cortes

// note: I have reduced the originals to 14 x 14 from 28 x 28

Datum training_set;
Datum testing_set;

class Datum
{
  float [] inputs;
  float [] outputs;
  int output;
  Datum()
  {
    inputs = new float [625];
    outputs = new float[128];
  }
  void loadPImage(PImage img)
  {
    for(int i = 0; i < 625; i++){
      int c = img.pixels[i]; // so we don't access the array too much
      int r=(c&0x00FF0000)>>16; // red part
      int g=(c&0x0000FF00)>>8; // green part
      int b=(c&0x000000FF); // blue part
      int grey=(r+b+g)/3;
      float pont = (float)grey / (float)255;
      inputs[i] = pont;
    }
  }
}

void loadData(PImage imgToTest)
{
  
  testing_set = new Datum();
  testing_set.loadPImage(imgToTest);
  
}