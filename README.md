# Weight-Space Programming

This is a basic implementation of weight-space programming, inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).

I trained a 33-parameter neural network on the two moons dataset, then deliberately changed its output bias so it predicted the same class for every input. I gave ChatGPT the model’s weights and 20 labeled examples and asked it to find the problem. Its one-parameter patch fixed all 10 incorrect predictions in those examples without retraining the network.
