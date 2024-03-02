This project aims to implement the SVM optimization algorithm described in the paper "A Dual Coordinate Descent Method for Large-scale Linear SVM" (Hsieh et al., 2008), which is available here:

https://icml.cc/Conferences/2008/papers/166.pdf

This is a simple Python implementation for binary classification. It utilizes random order of variables at each iteration and variable shrinking. Data used to fit the model should be given in the form of numpy arrays, with labels in {-1, 1}.

The implementation can be found in the file 'libsvm.py'. The notebook 'evaluation.ipynb' contains a short report on the algorithm's performance based on the 'a9a' dataset from OpenML.
