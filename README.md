# GAN-et-al-2014

This repository is implementation of the paper written by __Ian Goodfellow__ named [_Generative Adversarial Networks_](https://arxiv.org/abs/1406.2661)

Two models are trained simultaneously, a generative model 'G' and and discriminative model 'D'. This framework corresponds to a minimax two-player game.
In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples. 

### Generative model (G)
It captures the data distribution
Training procedure is to maximize the probability of D making a mistake

### Discriminative model (D)
Estimates the probability that a sample came from training data rather than G


## How to run the project?
Just clone the repository, and run main.py

At every 400 iterations, a model output image would be __stored__ in output/ folder
