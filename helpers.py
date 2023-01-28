import torch

# Takes an 11 channel one hot encoded tensor and returns an image:
# in the range [-1,1]
# of shape (3,n,n)
def onehotToImage(tensor):
    print(tensor.shape)
    _, H, W = tensor.shape

    output = torch.zeros((3, H, W))

    red = tensor[1:5]
    black = tensor[5]
    green = tensor[6:]

    for d, channel in enumerate(green):
        for i, row in enumerate(channel):
            for j, x in enumerate(row):
                if int(green[d][i][j].item()):
                    output[1][i][j] = (d+1)*51

    return ((output/255)*2)-1