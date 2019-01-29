# Arvato Capstone Project - Customer Segmentation and Prediction

### Table of Contents

1. [Installation](#installation)
2. [Introduction](#introduction)
3. [Project Motivation](#motivation)
4. [Files](#files)
5. [Instructions](#instructions)
6. [Results](#results)
7. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
Besides the libraries included in the Anaconda distribution for Python 3.6 the following libraries have been included in this project:
* `progressbar` - library to display progressbar
* `LGBM` -  gradient boosting framework that uses tree based learning algorithms.
* `XGBoost` - optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.
* `skopt` - simple and efficient library to minimize (very) expensive and noisy black-box functions.

## Introduction <a name="introduction"></a>
This project has 2 major parts and 1 minor at the end:
1. Customer Segmentation Report (Unsupervised Learning) - EDA, PCA and clustering analysis of the general and customer population with the goal of being able to describe the core customer base of the company.
2. Predict Customer Report (Supervised Learning) - Use what was observed in part 1 or parts of it to create a supervised learning model that could predict whether an individual would respond to a mail campaign.
3. Kaggle Competition - As a last step the predictions generated from the best performing model from part 2 could be submitted to Kaggle and compared against the other student submissions.

## Project Motivation <a name="motivation"></a>
This project provided by Arvato Financial Solutions was one of the available capstone projects. I chose this project mainly for two reasons:
* The data in this project is real and almost no cleaning has been done to it.
* Last part of the project gives me the opportunity to participate in a Kaggle competition with the other students.

## Files <a name="files"></a>
Arvato provided a couple of files for this project but as part of the terms and conditions I'm not allowed to share/include them in this repository.

## Instructions <a name="instructions"></a>
As stated above the data for this project is not publicly available. For this reason the notebook and models provided by this repository cannot be used, but are made available to serve as a snapshot of the state of the project at the end.

## Results <a name="results"></a>
See blog post [here](https://medium.com/@michel.naslund/exploring-customer-segments-and-predicting-customer-response-361e1f097bd9) 

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
I would like to thank Arvato Financial Solutions for the datasets/files and Udacity for making this project possible.