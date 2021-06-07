# Static-Code-Summarization
The repository contains code for the static anbalysis tool developed for code summarization.
It is an IR + NN based tool.
We have used Encoder-Decoder framework for Natural Language Generation.
Source code is being parsed in AST form into the Encoder.
Enoder is a stack of 3 LSTM units.
In all the files with prefix "run", we have trained the model using different paramaters.
The Encoder.py and Decoder.py files are alternative structures for the framework but havent been used in the experiments.
The attentiom.py file contains the attention layer for the framework.
To run the tool, just execute any file with prefix "run" and suffix ".py" by executing "python3 file.py"
