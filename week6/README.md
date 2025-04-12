# Gait Phase Classification with Multi-Layer Perceptreon (MLP)

Overview

This project Implement a Multi-Layer Perceptro (MLP) for Gait phase classification that using supervised trainning methode. the implementation is written in C++ and is designed for a amplitude signal that go brrrrr

## Table of Contents

- [Directory Structure](#directorystructure)
- [Requirements](#requirements)
- [Compilation and Execution](#Compilation_and_Execution)



## Directory Structure
```sh
project/
GaitPhase/
|  # Assigment report documentation
├── AssigmentReport3.pdf   
|  # Contains raw amplitude data 
├── data                   
│   └── dataset.txt
|
|  # Header for dataset handling
├── dataset.hpp             
|  # Implementation file
├── main.cpp                
|
|  # Header MLP Implementation model
├── model.hpp               
|   # Contain amplitude data that been manipulate
└── test                    
    └── testCase.txt
```

## Requirements

- C++17 or later
- A C++ compiler such as g++/gcc

## Compilation and Execution
### linux
To compile the project, use:
```sh
g++ main.cpp -o gaitPhase
```
To run the program:
```sh
./gaitPhase
```

### Windows 10/11
To compile the project on Windows, use(if gcc):
```sh
gcc main.cpp -o gaitPhase -lstdc++ | g++ main.cpp -o gaitPhase
```
To run the program:
```sh
click the gaitPhase.exe
```


## Author
I Wayan Firdaus Winarta Putra