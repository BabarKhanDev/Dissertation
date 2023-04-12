
# TODO

1. More Literature Review
2. Finish Introduction
3. Complete Methodology
4. Complete Experimental Results
a. I need to train the final model
b. What were the best hyperparameters?
    Best norm function for Generator and Discriminator: BatchNorm2d
    Best Number of channels:
        Generator:
            Encoder {3,32,64,128,256,512}
            Decoder {512, 256, 128, 64,32}
        Discriminator
            {6,100,200,400,800,1600}
    Best Optimiser:
        Generator: SGD, LR = 0.01
        Discriminator: Adam, LR = 0.001
    This gave us a loss of ~1.02

    Some other good parameter options were:
    Generator: 
        Encoder: (3, 100, 200, 400, 800, 1600)
        Decoder: (1600, 800, 400, 200, 100)
    Discriminator: 
        (6, 100, 200, 400, 800, 1600)
    Optimiser: Adam + Adam with default learning rates

c. Plot the graphs of some of the best and alternate hyperparameter options
d. Explain why we chose a subset of hyperparameters to test
    If I wanted to test every combination of hyperparameter option that would be a very large search space.
    2 normalisation layer options for generator
    2 normalisation layer options for discriminator
    4 channel layouts for generator
    4 channel layouts for discriminator
    3 optimisers for generator
    3 optimisers for discriminator
    3 learning rates per optimiser
    If optimiser is adam then 3 different beta options
    ~5000 possible combinations!
    This is far too many to evaluate, therefore it was important to pick the important ones.
    I chose some default values for my hyperparameters:
        Optimiser: Adam, betas = (0.9, 0.999)
        LR = 0.001
        Generator Channels:
            Encoder: (3,64,128,256,512,1024)
            Decoder: (1024,512,256,128,64)
        Discriminator Channels:
            (6,64,128,256,512,1024)
    I then looped through each parameter and varied the defaults with the options available, if any variant in parameter performed better than the current setup then it would repalce the existing parameter option. 
    Instead of having ~5000 parameter variations, we now have ~100. This is much better! 
e. Plot variations e.g. varying loss function etc.
f. Show our final results after lots of epochs of training
g. Analyse our results.

5. Complete Conclusion

Read Adadelta paper?