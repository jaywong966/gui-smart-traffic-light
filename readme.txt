For node.js
0. (install nodejs, https://nodejs.org/en/download/)
1. (open powershell, activate your python env)
2. pip install thrift
3. (cd to your project directory)
4. npm install .
5. npm install electron -g
6. npm install thrift
7. npm start

Optional:
7. (create your cv.thrift)
8. .\thrift-0.13.0.exe -out gen-nodejs/ --gen js:node thrift/cv.thrift
9. .\thrift-0.13.0.exe -out py/gen_py/ --gen py thrift/cv.thrift


For yolo detector
1.install Spyder via anaconda navigator
2.conda install -c conda-forge opencv
3.conda install pytorch torchvision cudatoolkit=10.2 -c pytorch