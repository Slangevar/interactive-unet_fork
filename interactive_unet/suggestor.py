import numpy as np

import torch
import torch.nn as nn

from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

from interactive_unet import utils, metrics

class Suggestor(nn.Module):

    def __init__(self, num_channels, num_classes):
        super().__init__()


        import segmentation_models_pytorch as smp

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.unet = smp.Unet(encoder_name='mobilenet_v2',
                             encoder_weights='imagenet',
                             in_channels=self.num_channels,
                             classes=self.num_classes)
                            #  encoder_depth=2,
                            #  decoder_channels=(32, 16))

        self.model = nn.Sequential(self.unet,
                                   nn.Softmax(dim=1)) 

            
    def forward(self, x):

        b, c, h, w = x.shape

        pred = self.model(x)

        return pred

def make_suggestions(image_features, mask, lr=0.0005, steps=30, model=None):

    torch.set_float32_matmul_precision('medium')

    image_size = mask.shape[0]
    
    unique_colors = utils.get_unique_colors(mask)[1:]
    num_classes = len(unique_colors)

    if num_classes == 1:
        # Return all same class
        suggestions = (np.ones((image_size, image_size, 3)) * unique_colors[0][None,None,:]).astype('uint8')
    else:

        mask, _ = utils.colored_to_categorical(mask)
        mask = (mask > 127)

        x = torch.tensor(image_features).to(torch.float32).cuda()
        y = torch.tensor(np.moveaxis(mask,-1,0)[None,...]).to(torch.float32).cuda()

        mask = np.sum([mask[:,:,i] * (i + 1) for i in range(num_classes)], 0)
        w = torch.tensor(np.repeat((mask > 0)[None,None,...], num_classes, 1)).to(torch.float32).cuda()

        if model is None:
            model = Suggestor(x.shape[1], num_classes).cuda()
        elif model.num_classes != num_classes:
            model = Suggestor(x.shape[1], num_classes).cuda()

        best_model = model.state_dict()
        best_loss = 100
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        transforms = v2.Compose([v2.RandomHorizontalFlip(p=0.5),
                                 v2.RandomVerticalFlip(p=0.5),
                                 v2.RandomRotation(degrees=(-360,360), interpolation=InterpolationMode.NEAREST),
                                 ])

        x = tv_tensors.Image(x)
        y = tv_tensors.Mask(y)
        w = tv_tensors.Mask(w)

        for t in range(steps):

            xt, yt, wt = transforms(x, y, w)
            
            y_pred = model(xt)
            loss = metrics.mcc_ce_loss(y_pred, yt, wt)

            if loss.isnan().any():
                model = Suggestor(x.shape[1], num_classes).cuda()
                best_model = model.state_dict()
                best_loss = 100

            if loss < best_loss:
                best_loss = loss
                best_model = model.state_dict()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.load_state_dict(best_model)
        model.eval()

        predictions = model(x).detach().cpu().numpy()
        predictions = np.argmax(predictions[0],0).reshape((image_size,image_size))
        
        suggestions = np.zeros((image_size,image_size,3)).astype('uint8')
        for i in range(len(unique_colors)):
            suggestions[predictions == i,:] = unique_colors[i]
            
    return suggestions, model
