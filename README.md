# Installation

For node.js
``` 
install nodejs, https://nodejs.org/en/download/
open powershell, activate your python env
pip install thrift
cd to your project directory
npm install .
npm install electron -g
npm install thrift
npm start
```
Put your video into ``` gui-smart-traffic-light\py\frames\videos ``` <br>
run``` download_weights.sh ``` which in ``` gui-smart-traffic-light\py\yolo_detector\weights ```

Optional:
```
9. (create your cv.thrift)
10. .\thrift-0.13.0.exe -out gen-nodejs/ --gen js:node thrift/cv.thrift
11. .\thrift-0.13.0.exe -out py/gen_py/ --gen py thrift/cv.thrift
```

For yolo detector
```
1.install Spyder via anaconda navigator
2.conda install -c conda-forge opencv
3.conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

# Demonstration
Click the following thumbnail for watching project's demonstration video.
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/P5skFgb6Pg0/0.jpg)](https://www.youtube.com/watch?v=P5skFgb6Pg0)

