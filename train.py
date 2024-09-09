import custom_layers
import custom_models

import keras
from keras import layers, models, ops, optimizers, preprocessing

import os

import tensorflow as tf

## adjust config dictionaries for better results. widths are far too low, higher depth, higher latent_filters for better recon. use kl_annealing for lower kl

model_config = {
    "encoder_widths": [16, 32],
    "decoder_widths": [32, 16],
    "kernel_size": 3,
    "activation": "swish",
    "use_bn": True,
    "depth": 1,
    "latent_filters": 3,
    "output_channels": 3,
    "kl_beta": 1
}

dataset_config = {
    "directory": "celeba",
    "image_size": (64, 64),
    "label_mode": None,
    "batch_size": 32,
}

model_compile = {
    "optimizer": optimizers.Adam(learning_rate=1e-4)
}


config = {
    "model_config": model_config,
    "optimizer_config": model_compile,
    "dataset_config": dataset_config
}


dataset = preprocessing.image_dataset_from_directory(**config["dataset_config"])
dataset = dataset.map(lambda x: (x / 255)).repeat()


vae = custom_models.VAE(**config["model_config"])

## file loading not working

if os.path.exists("vae_encoder.weights.h5"):
    vae.encoder.load_weights("vae_encoder.weights.h5")
    print("encoder save file found")

if os.path.exists("vae_decoder.weights.h5"):
    vae.encoder.load_weights("vae_decoder.weights.h5")
    print("decoder save file found")


def save_model(epoch=None, logs=None):

    vae.encoder.save_weights("vae_encoder.weights.h5")
    vae.decoder.save_weights("vae_decoder.weights.h5")
    print(f"\n Weights saved")

vae.compile(**config["optimizer_config"])

vae.fit(dataset, epochs=1000, steps_per_epoch=1000, callbacks=keras.callbacks.LambdaCallback(vae.save_model))
