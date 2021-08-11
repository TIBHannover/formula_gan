import functools
import torch
from torch import nn
import torch.nn.functional as F

from models.attention_layer import SelfAttention2d
from models.norm_layer import ProjectionBatchNorm2d


class BigGANGeneratorBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, emb_channels, upsample=None, bn_eps=1e-5, bn_momentum=0.1,
    ):
        super(BigGANGeneratorBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels

        # upsample layers
        if upsample is None:
            self.upsample = functools.partial(F.interpolate, scale_factor=2)
        else:
            self.upsample = upsample

        self.activation = nn.ReLU()

        # Conv layers
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=[3, 3], padding=[1, 1])
        )
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=[3, 3], padding=[1, 1])
        )
        self.learnable_sc = in_channels != out_channels or self.upsample is not None
        if self.learnable_sc:
            self.conv_sc = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
        # Batchnorm layers
        self.bn1 = ProjectionBatchNorm2d(in_channels, emb_channels=emb_channels)
        self.bn2 = ProjectionBatchNorm2d(out_channels, emb_channels=emb_channels)

    def forward(self, x, emb):
        h = self.activation(self.bn1(x, emb))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h, emb))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


# Residual block for the discriminator
class BigGANDiscriminatorBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, downsample_first=False, skip_first_act=False, activation=None, downsample=None,
    ):
        super(BigGANDiscriminatorBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        wide = True
        self.hidden_channels = self.out_channels if wide else self.in_channels

        self.skip_first_act = skip_first_act
        self.downsample_first = downsample_first

        if activation is None:
            self.activation = nn.ReLU(inplace=False)
        else:
            self.activation = activation

        if downsample is None:
            self.downsample = nn.AvgPool2d(2)
        else:
            self.downsample = downsample

        # Conv layers
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=[3, 3], padding=[1, 1])
        )
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=[3, 3], padding=[1, 1])
        )
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))

    def shortcut(self, x):
        if not self.downsample_first:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.skip_first_act:
            # h = self.activation(x) # NOT TODAY SATAN
            # Andy's note: This line *must* be an out-of-place ReLU or it
            #              will negatively affect the shortcut connection.
            h = self.activation(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h + self.shortcut(x)



class GANPix2Pix(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_classes=2, d=1.0, noise_encoder=True, init="ortho", use_self_attention=True, **kwargs):
        super(GANPix2Pix, self).__init__()
        self.init = init
        self.encoder = nn.Sequential(
            BigGANDiscriminatorBlock(in_channels, int(64 * d), downsample_first=True, skip_first_act=True),  # 64x64
            BigGANDiscriminatorBlock(int(64 * d), int(2 * 64 * d)),  # 32x32
            BigGANDiscriminatorBlock(int(2 * 64 * d), int(4 * 64 * d)),  # 16x16
            BigGANDiscriminatorBlock(int(4 * 64 * d), int(8 * 64 * d)),  # 8x8
        )

        # TODO
        if noise_encoder:
            self.noise_encoder = nn.utils.spectral_norm(nn.Conv2d(20, 512, kernel_size=[1, 1]))
        else:
            self.noise_encoder = None

        decoder_list = [
                BigGANGeneratorBlock(int(8 * 64 * d), int(8 * 64 * d), emb_channels=148),
                BigGANGeneratorBlock(int(8 * 64 * d), int(4 * 64 * d), emb_channels=148),]

        if use_self_attention:
            decoder_list.extend([
                SelfAttention2d(int(4 * 64 * d), spectral_norm=True),
            ])

        decoder_list.extend([
             BigGANGeneratorBlock(int(4 * 64 * d), int(2 * 64 * d), emb_channels=148),
                BigGANGeneratorBlock(int(2 * 64 * d), int(64 * d), emb_channels=148),
                nn.BatchNorm2d(int(64 * d), affine=True, momentum=0.9999, eps=1e-5),
                nn.ReLU(inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(int(64 * d), out_channels, kernel_size=[3, 3], padding=[1, 1],)),
                nn.Sigmoid(),
        ])

        self.decoder = nn.ModuleList(decoder_list)

        self.embedding = nn.utils.spectral_norm(nn.Embedding(num_classes, 128))

        self.init_weights()

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
                if self.init == "ortho":
                    torch.nn.init.orthogonal_(module.weight)
                elif self.init == "N02":
                    torch.nn.init.normal_(module.weight, 0, 0.02)
                elif self.init in ["glorot", "xavier"]:
                    torch.nn.init.xavier_uniform_(module.weight)
                else:
                    print("Init style not recognized...")
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print("Param count for G" "s initialized parameters: %d" % self.param_count)

    # @auto_move_data
    def forward(self, input, z, domain):
        z_splits = torch.chunk(z, 6, dim=1)

        emb = self.embedding(domain)

        x = self.encoder(input)

        if self.noise_encoder is not None:
            noise = self.noise_encoder(z_splits[0].view(x.shape[0], z_splits[0].shape[1], 1, 1))
            x = noise + x
        index = 1
        for m in self.decoder:
            if isinstance(m, BigGANGeneratorBlock):
                x = m(x, torch.cat([emb, z_splits[index]], dim=1))
                index += 1
            else:
                x = m(x)
        return x


class GANDiscriminator(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, d=1.0, init="ortho", use_self_attention=True,**kwargs):
        super(GANDiscriminator, self).__init__()
        self.init = init


        decoder_list = [
            BigGANDiscriminatorBlock(in_channels, int(64 * d), downsample_first=True, skip_first_act=True),  # 64x64
            BigGANDiscriminatorBlock(int(64 * d), int(2 * 64 * d)),  # 32x32]
        ]

        if use_self_attention:
            decoder_list.extend([
                SelfAttention2d(int(2 * 64 * d), spectral_norm=True),
            ])

        decoder_list.extend([
            BigGANDiscriminatorBlock(int(2 * 64 * d), int(4 * 64 * d)),  # 16x16
            BigGANDiscriminatorBlock(int(4 * 64 * d), int(8 * 64 * d)),  # 8x8
        ])

        self.decoder = nn.Sequential(*decoder_list )
        self.output = nn.utils.spectral_norm(nn.Linear(int(8 * 64 * d), 1))
        self.embedding = nn.utils.spectral_norm(nn.Embedding(num_classes, 512))

        self.init_weights()

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
                if self.init == "ortho":
                    torch.nn.init.orthogonal_(module.weight)
                elif self.init == "N02":
                    torch.nn.init.normal_(module.weight, 0, 0.02)
                elif self.init in ["glorot", "xavier"]:
                    torch.nn.init.xavier_uniform_(module.weight)
                else:
                    print("Init style not recognized...")
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print("Param count for D" "s initialized parameters: %d" % self.param_count)

    # @auto_move_data
    def forward(self, input, domain):

        x = self.decoder(input)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        output = self.output(x)
        emb = self.embedding(domain)
        return output + torch.sum(x * emb)
