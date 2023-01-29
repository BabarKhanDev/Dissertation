import torch

# Takes an 11 channel one hot encoded tensor and returns an image:
# in the range [-1,1]
# of shape (3,n,n)
def onehotToImage(tensor):
    red = tensor[0:5]
    green = tensor[6:]
    print(red.min(), red.max())
    print(green.min(), green.max())

    for d, channel in enumerate(red):
        red[d] = torch.mul(channel, (d+1)*51)
    red = torch.sum(red, dim=0)

    for d, channel in enumerate(green):
        green[d] = torch.mul(channel, (d+1)*51)
       
    green = torch.sum(green, dim=0)

    output = torch.stack([red, green, torch.zeros_like(red)])

    print(output.shape)
    print(red.min(), red.max())
    print(green.min(), green.max())
    
    print(output.min(), output.max())
    output = (output/255)
    print(output.min(), output.max())

    return output