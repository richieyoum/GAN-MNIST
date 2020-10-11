# Hand written number generation using GAN
This is a PyTorch implementation of a simple GAN that generates handwritten digits similar to that of a human's. Feel free to play around!

![generated output](https://user-images.githubusercontent.com/43356500/95690791-2b32cd80-0be8-11eb-8cf3-81f2e88553c0.png)

## Getting started
You can install the required dependency with
```
pip install -r requirements.txt
```
I've tested on Python 3.7 on Ubuntu, but should work on both Windows and MacOS with most recent versions of Python & Pytorch.

Note, training GANs can take a very long time especially if you're running it on CPU, although it is supported. 

With GTX 1660Ti, it took me about 20 minutes of training with current configs.

The final model weights are saved in "model_weights" folder after training. I've already included my model weights if you just want to use the model for generating the digits.

## Results
Generated images are saved under 'images' folder. 

Further training and fine-tuning will improve the results, but there are clear limitations with simple GAN. There will be future updates with a more sophisticated methods, so stay tuned!

Human digits

![original](https://user-images.githubusercontent.com/43356500/95690841-a0060780-0be8-11eb-9096-30b670c4ff4c.png)

Generated digits

![generated output](https://user-images.githubusercontent.com/43356500/95690791-2b32cd80-0be8-11eb-8cf3-81f2e88553c0.png)

Could still use some work, but not too bad! :)

## Author
* **Richie Youm**

## Credit
This project was inspired by deeplearning.ai specialization for GANs. I have referenced their model parameters and visualization techniques for convenience.
