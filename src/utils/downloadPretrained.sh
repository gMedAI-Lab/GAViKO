#!/bin/bash
SCRIPT_DIR=$(pwd)

if curl -L "https://huggingface.co/google/vit-base-patch16-224-in21k/resolve/main/pytorch_model.bin" -o "$SCRIPT_DIR/vit-base-patch16-224-in21k.pth"; then
    echo "Pretrained VIT downloading successful at path: $SCRIPT_DIR/vit-base-patch16-224-in21k.pth"
else
    echo "Pretrained VIT downloading failed. Please check your internet connection, the available disk space or the availability of the resource."
fi