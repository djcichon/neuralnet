# Neural Nets

This is my implementation of neural networks.  It implements fully connected networks and as a demonstration driver.py
will run it against the MNIST dataset.  With the default configuration (100 hidden neurons), expect about 98% accuracy.

# Installation

```git clone https://github.com/djcichon/neuralnet
cd neuralnet
pip install -r requirements.txt
```

# Planned Features

1. Process mini-batches in a single pass
2. Abstract activation functions, similar to cost functions
3. Layer types (i.e. fully connected, softmax, convolutional)
