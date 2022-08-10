import torch
import torch.nn as nn
from model.conv_bn_relu import ConvBNRelu
import torch
import torchvision.transforms as T


class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, FLAGS):
        super(Encoder, self).__init__()
        self.H = FLAGS.image_size
        self.W = FLAGS.image_size
        self.conv_channels = FLAGS.encoder_channels
        self.num_blocks = FLAGS.encoder_blocks

        layers = [ConvBNRelu(3, self.conv_channels,padding='valid')]
        layers.append(T.Pad(padding=(1, 1, 1, 1), padding_mode='symmetric'))

        for _ in range(FLAGS.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels,padding='valid')
            layers.append(layer)
            layers.append(T.Pad(padding=(1, 1, 1, 1), padding_mode='symmetric'))

        self.conv_layers = nn.Sequential(*layers)

        self.after_concat_layer = ConvBNRelu(self.conv_channels + FLAGS.redundant_length,self.conv_channels,padding='valid')
        self.after_concat_layer2 = ConvBNRelu(self.conv_channels ,self.conv_channels,padding='valid')
        self.final_layer = nn.Conv2d(self.conv_channels + 3, 3, kernel_size=1,padding='valid')
        self.expad = T.Pad(padding=(1, 1, 1, 1), padding_mode='symmetric')

    def forward(self, image, message):
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)
        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)
        encoded_image = self.conv_layers(image)
        concat = torch.cat([expanded_message, encoded_image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.expad(im_w)
        im_w = self.after_concat_layer2(im_w)
        im_w = self.expad(im_w)
        im_w = torch.cat([im_w,image], dim = 1)
        im_w = self.final_layer(im_w)
        #print("im_w:",im_w.shape)
        return im_w
