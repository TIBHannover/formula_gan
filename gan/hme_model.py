import os
import sys
import re
import imageio


import torchvision
from torch import nn
from torch.nn import functional as F
import torch

from formula_model import FormulaOCR


from config import config_add_options, ConfigEntry
from torch.nn.utils.rnn import pack_padded_sequence


from collections import OrderedDict

from model_utils import RowEncoder, Attention, DecoderWithAttention

cfgs = {
    'cnn1': [(64,False), ('M',2,2), (128,False), ('M',2,2), (256,True), (256,False), ('M',1,2), (512, True), ('M',2,1), 512],
    'cnn2': [(50,True), (100,True), ('M',2,2), (150,True), (200,True), ('M',2,2), (250,True,0.2), (300,True,0.2), ('M',2,2), (350,True,0.2), (400,True,0.2), ('M',2,2), (512,True,0.2)],
    'cnn3': [(100,True), (100,True), (100,True), ('M',2,2), (200,True), (200,True),(200,True), ('M',2,2), (300,True,0.2), (300,True,0.2), (300,True,0.2), ('M',1,2), (400,True,0.2), (400,True,0.2), (400,True,0.2), ('M',2,1), (512,True,0.2)],
}



class VGG(nn.Module):

    def __init__(self, features,  init_weights=True):
        super(VGG, self).__init__()
        self.features = features

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg):
    layers = []
    in_channels = 1
    for v in cfg:
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=v[1:3], stride=v[1:3])]
        else:
            conv2d = nn.Conv2d(in_channels, v[0], kernel_size=3, padding=1)
            if v[1]:
                layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            
            if len(v) > 2:
                layers += [nn.Dropout(v[2])]
            in_channels = v[0]
    return nn.Sequential(*layers)


class HMEEncoder(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, type='cnn3' ):

        super(HMEEncoder, self).__init__()

        # First convolution
        self.features = VGG(make_layers(cfgs[type]))
        self.row_encoder = RowEncoder(512, 256)

    def forward(self, x):
        features = self.features(x)
        out = self.row_encoder(features)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)

        return out



def read_dictionary(dictionary_path):

    # print("Read dictionary")
    dictionary = {}
    with open(dictionary_path, "r") as fr:
        for line in fr.readlines():
            word, index = line.split("\t")
            dictionary[word] = int(index) + 2
    dictionary["<start>"] = 1
    dictionary["<end>"] = 2
    dictionary["<pad>"] = 0
    return dictionary


class HMEModel(FormulaOCR):
    def __init__(self, *args, **kwargs):
        super(HMEModel, self).__init__(*args, **kwargs)

        self.decoder = DecoderWithAttention(
            attention_dim=self.params.model.attention_dim,
            embed_dim=self.params.model.emb_dim,
            decoder_dim=self.params.model.decoder_dim,
            vocab_size=113,
            dropout=self.params.model.dropout,
            encoder_dim=512,
        )

        self.encoder = HMEEncoder()

        self.dictionary = read_dictionary(self.params.test_dataloader.dictionary_path)
        self.inv_dictionary = {v: k for k, v in self.dictionary.items()}

    def forward(self, image):

        return self.encoder(image)

    def training_step(self, batch, batch_idx):

        image_embedding = self(batch["image"])

        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(
            image_embedding, batch["sequence"], torch.sum(batch["sequence_mask"], dim=1), device =  batch["image"].device.index
        )
        targets = caps_sorted[:, 1:]
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        loss= F.cross_entropy(scores, targets)
        perplexity  = torch.exp(loss)

        
        return {"loss":loss, "perplexity": perplexity, 'image': batch['image']}


    def training_step_end(self, batch_parts_outputs):

        if self.global_step%self.trainer.log_save_interval ==0:
            grid = torchvision.utils.make_grid(batch_parts_outputs['image'])
            self.logger.experiment.add_image(f'train/images', grid, self.global_step)
            self.logger.experiment.add_histogram(f'train/images', batch_parts_outputs['image'], self.global_step)

        

        return {'loss':batch_parts_outputs['loss'].mean(), 'log': {'train_loss': batch_parts_outputs['loss'].mean(), 'train_perplexity': batch_parts_outputs['perplexity'].mean()}}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL

        image_embedding = self.encoder(batch["image"])
        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(
            image_embedding, batch["sequence"], torch.sum(batch["sequence_mask"], dim=1), device =  batch["image"].device.index
        )
        targets = caps_sorted[:, 1:]
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        return {"val_loss": F.cross_entropy(scores, targets)}

    def validation_end(self, outputs):
        # print(outputs)
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        perplexity  = torch.exp(avg_loss)

        # self.logger.experiment.add_scalar(f'val/loss', avg_loss, self.global_step)
        # self.logger.experiment.add_scalar(f'val/perplexity', perplexity, self.global_step)
        
        # global_step is for storing
        return {"val_loss": avg_loss, "val_perplexity": perplexity, "global_step": self.global_step , 'log':{
             'val_loss': avg_loss, 'val_perplexity': perplexity
        }}

    def validation_epoch_end(self, outputs):
        # print(outputs)
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        perplexity  = torch.exp(avg_loss)

        # self.logger.experiment.add_scalar(f'val/loss', avg_loss, self.global_step)
        # self.logger.experiment.add_scalar(f'val/perplexity', perplexity, self.global_step)
        
        # global_step is for storing
        return {"val_loss": avg_loss, "val_perplexity": perplexity, "global_step": self.global_step , 'log':{
             'val_loss': avg_loss, 'val_perplexity': perplexity
        }}


    def test_step(self, batch, batch_idx):
        
        hypotheses = []

        k = 1#self.params.test.beam_size

        image_embedding = self(batch["image"])
        enc_image_size = image_embedding.size(1)
        encoder_dim = image_embedding.size(3)

        # Flatten encoding
        image_embedding = image_embedding.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = image_embedding.size(1)

        # We'll treat the problem as having a batch size of k
        image_embedding = image_embedding.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[self.dictionary['<start>']]] * k).cuda( batch["image"].device.index)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).cuda( batch["image"].device.index)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = self.decoder.init_hidden_state(image_embedding)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = self.decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = self.decoder.attention(image_embedding, h)  # (s, encoder_dim), (s, num_pixels)

            gate = self.decoder.sigmoid(self.decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = self.decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)
            # print(scores)
            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / len(self.dictionary.keys())# vocab_size  # (s)
            next_word_inds = top_k_words % len(self.dictionary.keys())# vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != self.dictionary['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            image_embedding = image_embedding[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            # print(top_k_scores)
            # Break if things have been going on too long
            if step > 500:
                break
            step += 1


        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # # References
        # img_caps = allcaps[0].tolist()
        # img_captions = list(
        #     map(lambda c: [w for w in c if w not in {self.dictionary['<start>'], self.dictionary['<end>'], self.dictionary['<pad>']}],
        #         img_caps))  # remove <start> and pads
        # references.append(img_captions)

        # Hypotheses

        result_idx =[w for w in seq if w not in {self.dictionary['<start>'], self.dictionary['<end>'], self.dictionary['<pad>']}]
        result_str = [self.inv_dictionary[w] for w in result_idx]

        gt = batch['sequence'].squeeze(0).cpu().numpy().tolist()

        gt_idx =[w for w in gt if w not in {self.dictionary['<start>'], self.dictionary['<end>'], self.dictionary['<pad>']}]
        gt_str = [self.inv_dictionary[w] for w in gt_idx]

        if self.params.test.prediction_output_path is not None:
            image_out = os.path.join(self.params.test.prediction_output_path,'img')
            gt_out = os.path.join(self.params.test.prediction_output_path,'gt')
            res_out = os.path.join(self.params.test.prediction_output_path,'res')

            os.makedirs( image_out, exist_ok = True)
            os.makedirs( gt_out, exist_ok = True)
            os.makedirs( res_out, exist_ok = True)

            filename = os.path.splitext(os.path.basename(batch['path'][0]))[0]

            with open(os.path.join(gt_out, f'{filename}.tex'), 'w') as f:
                f.write('$' +' '.join(gt_str) + '$\n')
            
            with open(os.path.join(res_out, f'{filename}.tex'), 'w') as f:
                f.write('$' +' '.join(result_str) + '$\n')

            imageio.imwrite(os.path.join(image_out, f'{filename}.jpg'), batch['image'].squeeze(0).squeeze(0).cpu().numpy())

        
        return {}

    

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        optimizer = torch.optim.AdamW(
            params=list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.params.optimizer.lr, weight_decay=self.params.optimizer.weight_decay
        )
        # optimizer = torch.optim.SGD(
        #     params=list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.params.optimizer.lr, weight_decay=self.params.optimizer.weight_decay,momentum=0.9
        # )
        return optimizer

    # def val_dataloader(self):
    #     # OPTIONAL
    #     return DataLoader(
    #         MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()),
    #         batch_size=self.params.val_dataloader.batch_size,
    #     )

    # def test_dataloader(self):
    #     # OPTIONAL
    #     return DataLoader(
    #         MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()),
    #         batch_size=self.params.test_dataloader.batch_size,
    #     )
    @config_add_options("optimizer")
    def config_optimizer():
        return {
            "lr": ConfigEntry(default=1e-3, type=float),  # learning rate for encoder if fine-tuning
            "weight_decay": ConfigEntry(default=1e-4, type=float)
        }

    @config_add_options("model")
    def config_optimizer():
        return {
            "emb_dim": ConfigEntry(default=512),  # dimension of word embeddings
            "attention_dim": ConfigEntry(default=512),  # dimension of attention linear layers
            "decoder_dim": ConfigEntry(default=512),  # dimension of decoder RNN
            "dropout": ConfigEntry(default=0.5),
            "alpha_c": ConfigEntry(default=1.0),
        }


    @config_add_options("test")
    def config_optimizer():
        return {
            "beam_size": ConfigEntry(default=4),
            "prediction_output_path": ConfigEntry(default=None, type=str)
        }

{
    # Training parameters
    # 'epochs': ConfigEntry(default=120),  # number of epochs to train for (if early stopping is not triggered)
    # 'epochs_since_improvement': ConfigEntry(default=0),  # keeps track of number of epochs since there's been an improvement in validation BLEU
    # # 'batch_size': ConfigEntry(default=32)
    # 'workers': ConfigEntry(default=1),  # for data-loading; right now, only 1 works with h5py
    # 'grad_clip': ConfigEntry(default=5.),  # clip gradients at an absolute value of
    # 'alpha_c': ConfigEntry(default=1.),  # regularization parameter for 'doubly stochastic attention', as in the paper
    # 'best_bleu4': ConfigEntry(default=0.),  # BLEU-4 score right now
    # 'print_freq': ConfigEntry(default=100),  # print training/validation stats every __ batches
    # 'fine_tune_encoder': ConfigEntry(default=False),  # fine-tune encoder?
    # 'checkpoint': ConfigEntry(default=None),  # path to checkpoint, None if none
}
