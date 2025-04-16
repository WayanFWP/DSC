# Multi-Layer Perceptron (MLP) with Error Backpropagation Algorithm (EBPA)

## Overview

This project implements a Multi-Layer Perceptron (MLP) using the Error Backpropagation Algorithm (EBPA). The implementation is written in C++ and is designed for character recognition using a dataset of labeled text files.

## Table of Contents

- [Directory Structure](#directorystructure)
- [Requirements](#requirements)
- [Compilation and Execution](#Compilation_and_Execution)



## Directory Structure
```sh
project/
├── MLP_EBPA.hpp        # Header MLP implementation
├── MLP.cpp             # Implementation file for MLP
├── dataset.hpp         # Header file for dataset handling
├── dataset/            # Contains labeled text
│   ├── A.txt
│   ├── B.txt
│   ├── C.txt
│   ├── ...
│   └── H.txt
├── testcase/               # Contains labeled test samples
│   ├── A1.txt
│   ├── A2.txt
│   ├── B1.txt
│   ├── ...
│   └── H2.txt
└── AssigmentReport2.pdf # Assignment report documentation
```

## Requirements

- C++17 or later
- A C++ compiler such as g++

## Compilation and Execution
### linux
To compile the project, use:
```sh
g++ MLP.cpp -o mlp
```
To run the program:
```sh
./mlp
```

### Windows 10/11
To compile the project on Windows, use(if gcc):
```sh
gcc MLP.cpp -o mlp -lstdc++
```
To run the program:
```sh
click the mlp.exe
```


## Author
I Wayan Firdaus Winarta Putra
