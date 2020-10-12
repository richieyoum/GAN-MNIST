# Hand written number generation using GAN
This is a PyTorch implementation of DCGAN and SimpleGAN that can generate handwritten digits similar to that of a human's. 

Feel free to play around!
![dcgan output](https://user-images.githubusercontent.com/43356500/95772146-45bd8300-0c8a-11eb-88ff-e4f6e1a49cdb.png)


## Getting started
You can install the required dependency with
```
pip install -r requirements.txt
```
And you can just run below to train
```
python main.py
```

I've tested on Python 3.7 on Ubuntu, but should work on both Windows and MacOS with most recent versions of Python & Pytorch.

Note, training GANs can take a very long time especially if you're running it on CPU, although it is supported. 

With GTX 1660Ti, it took me about 20 minutes of training for simpleGAN with 200 epochs & 128 batch size.

The final model weights are saved in "model_weights" folder after training. I've already included my model weights for both DCGAN and SimpleGAN if you just want to use the model for generating the digits.

## Results
Generated images are saved under 'images' folder. Generally, DCGAN outperforms a simple GAN using linear layers, but that can be further improved with more training and fine-tuning of the parameters.

As you train, it would save the generated image from the generator progressively, so you can compare how it has improved over the epochs.

Human digits

![original](https://user-images.githubusercontent.com/43356500/95690841-a0060780-0be8-11eb-9096-30b670c4ff4c.png)

SimpleGAN output

![generated output](https://user-images.githubusercontent.com/43356500/95690791-2b32cd80-0be8-11eb-8cf3-81f2e88553c0.png)

DCGAN output
![dcgan output](https://user-images.githubusercontent.com/43356500/95772146-45bd8300-0c8a-11eb-88ff-e4f6e1a49cdb.png)

Could still use some work, but not too bad! :)

## Todo
* Add image generation mode separate from training

## Author
* **Richie Youm**

## Credit
This project was inspired by deeplearning.ai specialization for GANs. I have referenced their model parameters and visualization techniques for convenience.
