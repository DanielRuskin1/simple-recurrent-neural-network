# Simple Recurrent Neural Network

This repository contains a simple implementation of a recurrent neural network.  It allows you to train a single-layer RNN with stochastic gradient descent and backpropagation through time (BPTT).

This project does not use any machine learning libraries aside from the linear algebra library armadillo.  SGD and BPTT are all implemented from scratch.

# Documentation

A UML diagram is enclosed in the `uml` folder.  This details the structure of the codebase.

In addition, you can specify the `--help` option to have the program output the help message.

# Example Commands

The following example command allows you to train a model:

```
./simple_recurrent_neural_network --command train_model --num_threads 30 --data_file data.txt --text_vocab_size 5000 --saved_state_size 100 --learning_rate 0.01 --num_epochs 10 --num_samples_per_batch 500 --test_data_frac 0.05 --bptt_truncate 5
```

The following example command allows you to make predictions from an existing model:

```
./simple_recurrent_neural_network --command predict_text_gen --predict_prev_output_folder ".output-20180410T223153/" --predict_text_gen_first_char "the" --predict_text_gen_last_char "END_TOKEN" --predict_text_gen_max_chars 100
```

# Dependencies and Compiling

This program requires Boost (1.60.0) and armadillo (8.400.0).  You can compile the program with the `src/Makefile_linux` makefile.

# Scripts

Several scripts are included in the scripts directory. These may be helpful for transforming your data or generating new data.

# References

The following references were used while preparing this program:

```
Britz, Denny. “Recurrent Neural Networks Tutorial.” WildML, 8 July 2016, www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/. Parts 1 through 4.

Brown, Carter N. “Gradients for an RNN.” Github, github.com/go2carter/nn-learn/blob/master/grad-deriv-tex/rnn-grad-deriv.pdf.

Wenchy. Github Gist. Github, gist.github.com/Wenchy/64db1636845a3da0c4c7.
```
