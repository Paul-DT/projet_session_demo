//****************************************************************************************************************************************
//README
//Pour utiliser ce programmme il faut installer OpenCV dans Processing
//Pour cela il sufit d'aller dans l'onglet 'Sketch' selectionner 'Importer une librairie...' puis cliquer sur 'Ajouter une librairie...'
//une fenetre appeler 'Contribution Manager' devrait s'ouvrir, dans cette fenetre, entrer "OpenCV" dans la bare
//cliquer sur le resultat 'OpenCV for Processing' puis cliquer sur 'Install', une fois l'intallation terminer redemarrer Processing
//****************************************************************************************************************************************
//Références
//Ce projet utilise Opencv pour Processing dont la documentation se trouve a l'adresse suivante
//      https://github.com/atduskgreg/opencv-processing
//Ce projet utilise un réseau de neurone pré-consu qui se trouve a l'adresse suivante
//      https://www.openprocessing.org/sketch/2292#
//Le code du projet est basé sur les concepts aborder dans les sites suivant
//      https://medium.com/@ageitgey/machine-learning-is-fun-80ea3ec3c471
//      http://natureofcode.com/book/chapter-10-neural-networks/
//      http://neuroph.sourceforge.net/image_recognition.html


import gab.opencv.*;
import processing.video.*;
import java.awt.*;

Capture video;
Capture Face;
Network neuralNet;

OpenCV opencv;

Rectangle[] faces;
PImage theFace;
int delayFaceDetect;

int camWidth;
int camHeight;

int faceWidth;
int faceHeight;

String addText;
boolean addTextSelected;

String trainText;
boolean trainTextSelected;

boolean faceError;
boolean isTraining;
boolean aTest;
boolean notif;
boolean waitTouch;
PImage imgInFiles;
PImage imgActual;

int delayDemo;
int delayTest;
int delayTrain;

Movie movie;

void setup() {
  //video = new Capture(this, width/scl, height/scl);
  //video = new Capture(this, 640, 360);
  //video.start();
  movie = new Movie(this,"obama.mp4");
  //opencv = new OpenCV(this, width/scl, height/scl);
  opencv = new OpenCV(this,640,360);
  //opencv = new OpenCV(this, "test.jpg");
  movie.loop();
  movie.play();
  camWidth = 640;
  camHeight = 360;
  opencv.loadCascade(OpenCV.CASCADE_FRONTALFACE);  
  
  delayFaceDetect = 0;
  
  fullScreen();
  //surface.setSize((int)(camWidth * 2.9), camHeight * 2);
  textSize(24);
  
  addTextSelected = false;
  addText = "Nom de l'utilisateur";
  trainTextSelected = false;
  //trainText = "Nom de l'utilisateur";
  trainText = "obama";
  theFace = new PImage();
  faceError = true;
  isTraining = false;
  aTest = false;
  notif = false;
  waitTouch = false;
  setupSigmoid();
  neuralNet = new Network(625,500,128);
  delayDemo = 6100;
  delayTest = 150;
  delayTrain = 0;
  
}

void draw() {
  if(!waitTouch){
    updateImage();
    drawFaces();
    //drawOptions();
    if(!getFaceError()) {
      //reconnaissence faciale;
      float[] resultActualFace;
      getFaceForRecognition();
      loadData(theFace);
      resultActualFace = neuralNet.respond(testing_set.inputs);
      if(isTraining){
        imgInFiles = loadImage(trainText + ".jpg");
        loadData(imgInFiles);
        float[] reference = neuralNet.respond(testing_set.inputs);
        for(int i = 0; i < 1; i++){
          loadData(theFace);
          resultActualFace = neuralNet.respond(testing_set.inputs);
          
          for(int j = 0; j < resultActualFace.length; j++){
            if(resultActualFace[j] < reference[i] - 0.001){
              resultActualFace[j] = reference[i] - 0.001;
            }
            if(resultActualFace[j] > reference[i] + 0.001){
              resultActualFace[j] = reference[i] + 0.001;
            }
          }
          neuralNet.train(resultActualFace);
          
          //neuralNet.train(reference);
          String[] users = loadStrings("data/Users.txt");
          for(int j = 0; j < users.length; j++){
            if(!users[j].equals(trainText)){
              imgInFiles = loadImage(users[j] + ".jpg");
              loadData(imgInFiles);
              float[] temp = neuralNet.respond(testing_set.inputs);
              for(int y = 0; y < temp.length; y++){
                  if(temp[y] > 0){
                    temp[y] = random(0, 0.5);
                  }else if(temp[y] < 0){
                    temp[y] = random(0, -0.5);
                  }
                  /*if(temp[y] > 0){
                    temp[y] = temp[y] - 0.5;
                  }else if(temp[y] < 0){
                    temp[y] = temp[y] + 0.5;
                  }else{
                    temp[y] = random(-0.49, 0.49);
                  }*/
              }
              neuralNet.train(temp);
            }
          }
        }
      }
      
      if(aTest){
        String[] users = loadStrings("data/Users.txt");
        for(int i = 0; i < users.length; i++){
          imgInFiles = loadImage(users[i] + ".jpg");
          loadData(imgInFiles);
          float[] reference = neuralNet.respond(testing_set.inputs);
          float diff = calculDiff(resultActualFace, reference);
          if(diff < 0){
            diff = diff * -1;
          }
          float percent = (1 - diff) * 100;
          fill(255);
          stroke(255);
          line(camWidth + (i*150), height * 0.95 + 20, camWidth + (i*150), (height * 0.95 + 20) - (percent * 2));
          text(nf(percent, 2, 2) + "%", camWidth + 5 + (i*150), height * 0.95);
          boolean trunc = false;
          while(users[i].length() > 6){
            users[i] = users[i].substring(0, users[i].length() - 1);
            trunc = true;
          }
          if(trunc){
          text(users[i] + "...", camWidth + 5 + (i*150), height * 0.95 + 20);
          }else {
            text(users[i], camWidth + 5 + (i*150), height * 0.95 + 20);
          }
        }
        aTest = false;
      }
    }
    if(notif){
      delay(2000);
      notif = false;
    }
    neuralNet.draw();
  }
  if(isTraining){
    fill(0,255,0);
    text("Le programme est en entrainement", 15, camHeight + 345); 
  }
  delayDemo--;
  if(delayTest > 0){
    delayTest--;
    aTest = true;
    if(delayTest == 0){
      delayTrain = 600;
      aTest = false;
      isTraining = true;
    }
  }
  
  if(delayTrain > 0){
    delayTrain--;
    isTraining = true;
    if(delayTrain == 0){
      delayTest = 150;
      isTraining = false;
      aTest = true;
    }
  }
  
  if(delayDemo > 0){
    delayDemo--;
  }else{
    exit();
  }
}

void updateImage() {
  background(127);
  movie.read();
  //if(movie.available()){
  //  exit();
  //}
  opencv.loadImage(movie);
  image(movie, 0, 0); 
}

void drawOptions(){
  fill(255);
  stroke(0);
  strokeWeight(2);
  //rect(addTextPos.x, addTextPos.y, addTextSize.x, addTextSize.y);
  rect(10, camHeight + 110, 350, 30);
  rect(400, camHeight + 110, 100, 30);
  rect(10, camHeight + 190, 350, 30);
  rect(400, camHeight + 190, 100, 30);
  rect(520, camHeight + 190, 100, 30);
  rect(250, camHeight + 245, 100, 30);
  fill(255);
  text("Ajouter un utilisateur :", 15, camHeight + 100);
  text("Entrainer le programme :", 15, camHeight + 180);
  text("Tester un visage :", 15, camHeight + 270);
  fill(0);
  text(addText, 15, camHeight + 134);
  text("Ajouter", 408, camHeight + 134);
  text(trainText, 15, camHeight + 214);
  text("Début", 415, camHeight + 214);
  text("Arrêt", 540, camHeight + 214);
  text("Tester", 265, camHeight + 269);
}

void drawFaces() {
  if(delayFaceDetect == 0){
    faces=opencv.detect();
    delayFaceDetect = 15;
  }
  else{
    delayFaceDetect--;
  }
  for (int i = 0; i < faces.length; i++) {
    noFill();
    stroke(0, 255, 0);
    strokeWeight(3);
    rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
    stroke(0);
    fill(255);
  }
}

boolean getFaceError() {
  fill(255);
  text(faces.length + " visage(s)", 10, camHeight + 25);
  if(faces.length == 0) {
    text("Erreur aucun visage détecté", 10, camHeight + 45);
    text("Fonctionnalités de reconnaissance faciale indisponibles", 10, camHeight + 65);
    faceError = true;
    return true;
  }
  else if(faces.length > 1) {
    text("Erreur plusieurs visages détecté", 10, camHeight + 45);
    text("Fonctionnalités de reconnaissance faciale indisponibles", 10, camHeight + 65);
    faceError = true;
    return true;
  }
  else {
    text("Fonctionnalités de reconnaissance faciale disponibles", 10, camHeight + 45);
    faceError = false;
    return false;
  }
}

void getFaceForRecognition(){
  PImage f = get(faces[0].x, faces[0].y, faces[0].width, faces[0].height);
  f.resize(162,162);
  f.filter(GRAY);
  //f.filter(BLUR,1);
  image(f, camWidth + 10, 10);
  int[] thePixels = new int[256];
  for(int i = 0; i < 256; i++){
    thePixels[i]=0;
  }
  for(int i = 0; i < f.pixels.length; i++){
  int c = f.pixels[i]; // so we don't access the array too much
  int r=(c&0x00FF0000)>>16; // red part
  int g=(c&0x0000FF00)>>8; // green part
  int b=(c&0x000000FF); // blue part
  int grey=(r+b+g)/3;
  thePixels[grey]++;
  }
  for(int i = 0; i < 256; i++){
    line(camWidth + 100 + i,camHeight + 100 - thePixels[i]/4, camWidth + 100 + i,camHeight + 100);
  }
  f.resize(25,25);
  theFace = f.copy();
}

float calculDiff(float[] testing, float[] imgReference){
  float totalDiff = 0;
  for(int i = 0; i < testing.length; i++){
  float variance = testing[i] - imgReference[i]; 
    if(variance < 0){
      variance = variance * -1;
    }
    totalDiff += variance;
  }
  float moyenneDiff = totalDiff / testing.length;
  return moyenneDiff;
}

void captureEvent(Capture c) {
  if(c.available()==true)
    c.read();
}

/*void mouseClicked(){
  if(waitTouch){
    waitTouch = false;
    return;
  }
  
  if((mouseY >= (camHeight + 110) && mouseY <= (camHeight + 150)) && (mouseX >= 10 && mouseX <= 360)){
    if(addText == "Nom de l'utilisateur"){
      addText = "";
    }
    addTextSelected = true;
    trainTextSelected = false;
  }else if((mouseY >= (camHeight + 190) && mouseY <= (camHeight + 220)) && (mouseX >= 10 && mouseX <= 360)){
    if(isTraining){
      fill(255,0,0);
      text("Le programme est en entrainement", 15, camHeight + 300);
      notif = true;
    }else {
      trainTextSelected = true;
      addTextSelected = false;
    }
    if(trainText == "Nom de l'utilisateur"){
        trainText = "";
      }
  }else {
      addTextSelected = false;
      trainTextSelected = false;
  }
  
  if((mouseY >= (camHeight + 110) && mouseY <= (camHeight + 150)) && (mouseX >= 400 && mouseX <= 500) && !faceError){
    if(addText.equals("test")){
      fill(255,0,0);
      text("Pour enregistrer un utilisateur son nom doit etre different de 'test'", 15, camHeight + 300);
      notif = true;
    }else if(addText != "Nom de l'utilisateur" && !addText.equals("")){
      theFace.save("data/" + addText + ".jpg");
      String[] users = loadStrings("data/Users.txt");
      boolean toAddInTextFile = true;
      for(int i = 0; i < users.length; i++){
        if(addText.equals(users[i])){
          toAddInTextFile = false;
        }
      }
      if(toAddInTextFile){
      String[] newUsers = append(users, addText);
      saveStrings("data/Users.txt", newUsers);
      }
      fill(0,255,0);
      text("L'utilisateur '" + addText + "' a ete enregistrer", 15, camHeight + 300);
      notif = true;
    }else {
      fill(255,0,0);
      text("Pour enregistrer un utilisateur il faut entrer un nom d'utilisateur", 15, camHeight + 300);
      notif = true;
    }
  }
  if((mouseY >= (camHeight + 190) && mouseY <= (camHeight + 220)) && (mouseX >= 400 && mouseX <= 500)){
    if(trainText != "Nom de l'utilisateur" && trainText != ""){
      boolean userExits = false;
      String[] users = loadStrings("data/Users.txt");
      for(int i = 0; i < users.length; i++){
        if(trainText.equals(users[i])){
          userExits = true;
        }
      }
      if(userExits){
        isTraining = true;
        trainTextSelected = false;
      }else {
        fill(255,0,0);
        text("L'utilisateur '" + trainText + "' n'est pas enregistrer", 15, camHeight + 300);
        notif = true;
      }
    }else {
      fill(255,0,0);
      text("Pour entrainer le programme il faut entrer un nom d'utilisateur", 15, camHeight + 300);
      notif = true;
    }
  }
  if((mouseY >= (camHeight + 190) && mouseY <= (camHeight + 220)) && (mouseX >= 520 && mouseX <= 620)){
    isTraining = false;
  }
  
  if((mouseY >= (camHeight + 245) && mouseY <= (camHeight + 275)) && (mouseX >= 250 && mouseX <= 350)){
    String[] users = loadStrings("data/Users.txt");
    boolean thereAreUser = false;
    if(users.length > 0){
      thereAreUser = true;
    }
    if(thereAreUser){
      aTest = true;
      isTraining = false;
    }else {
        fill(255,0,0);
        text("Il n'y a pas d'utilisateur enregistrer pour un test", 15, camHeight + 300);
        notif = true;
    }
  }
}

void keyPressed(){
  if(addTextSelected){
    if((key >= 65 && key <= 90)||(key >= 97 && key <= 122) || key == 32){
      addText += key;
    }
    if(key == 8){
      String tempText = addText;
      addText = "";
      for(int i = 0; i < tempText.length() - 1; i++){
        addText += tempText.charAt(i);
      }
    }
  }
  
  if(trainTextSelected){
    if((key >= 65 && key <= 90)||(key >= 97 && key <= 122) || key == 32){
      trainText += key;
    }
    if(key == 8){
      String tempText = trainText;
      trainText = "";
      for(int i = 0; i < tempText.length() - 1; i++){
        trainText += tempText.charAt(i);
      }
    }
  } 
}*/