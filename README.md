# Unsupervised Training Data Generation of Handwritten Formulas using Generative Adversarial Networks with Self-Attention

## Generator

The gan folder contains the generator that can translate images from rendered to handwritten formulas. The dependencies are defined in a requirements.txt and a Dockerfile.

```sh
cd gan
```

To create a docker container with all dependencies and the source code you need to run the following in the gan subfolder:

```sh
docker build . -t gan 
```

### Inference

To use the pre-trained model by default, first download the model checkpoint by running:


```sh 
wget https://github.com/TIBHannover/formula_gan/releases/download/0.9/gan_task_1_no_rendered_synth.model_300000.ckpt -O ../data/models/gan_task_1_no_rendered_synth/model_300000.ckpt
```

The trained generator can convert a folder with formula images into handwritten versions with the following command:

```sh
sudo docker run -v "$(pwd)"/../data:/data gan python gan_infer.py \
--infer.checkpoint_path /data/models/gan_task_1_no_rendered_synth/model_300000.ckpt \
--infer_dataloader.path /data/examples/ \
--infer.output_path /data/output/ \
--gan.sync_bn False
```

## Pregenerated Datasets

### NTCIR-12 MathIR

Randomly generated images from the NTCIR-12 MathIR dataset, based on a random latex font and resolution. Dateset is stored in a webdataset format and contains more than 400000 synthetic images from the gan. (The dateset contains a few more examples than specified in the paper. Unfortunately the original files were on a failed HDD and could not be reconstructed.)

[Link to Webdataset](https://tib.eu/cloud/s/33zLLydQPy6SrtG)

### Im2Latex

Randomly generated images from the Im2Latex dataset, based on a random latex font and resolution. Dateset is stored in a webdataset format and contains more than 290000 synthetic images from the gan. (The dateset contains a few more examples than specified in the paper. Unfortunately the original files were on a failed HDD and could not be reconstructed.)

[Link to Webdataset](https://tib.eu/cloud/s/TqDZ6EeGfd3eEpd)