# CUDA Implementation of the LRP
In this repository I attempt to implement the Layerwise Relevance Propagation (LRP) algorithm using CUDA. This implementation will allow for batching and thus be more parallelizable than the original implementation. The code will be optimized to the TITAN RTX architecture.

## Relevant System Information
* OS: Linux
* Architecture: x86_64
* Distribution: Ubuntu (18.04)
* GPU: TITAN RTX
* CUDA: 10.2

## What is LRP?
Take a look at the original repository here: [iNNvestigate neural networks!](https://github.com/albermax/innvestigate)

LRP is a technique in explainable AI. It works with all kinds of Neural Networks by taking the activations of the last layers and reinterpreting those activations as relevances. The relevances are redistributed to the previous layers depending on all intermendiate layer activations and the network weights.

The result of the LRP is a heatmap of relevances over the input feature space. In an image classification task i will be able to ask the LRP: "Which pixels were relevant for the prediction 'cat'?" The LRP will then show me a heatmap highlighting pixels that were relevant for the prediction 'cat'. This typically means, that the pixels that show the cat are 'hot' and all other pixels are 'cold'. In most cases LRP therefore implicitly also segments the image!
