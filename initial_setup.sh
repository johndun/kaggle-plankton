#!/bin/bash
mkdir img
mkdir log
mkdir model
mkdir result
cd data
wget http://s3.amazonaws.com/johndun.aws.bucket/kaggle-plankton-data/train.zip
wget http://s3.amazonaws.com/johndun.aws.bucket/kaggle-plankton-data/test.zip
unzip -q train.zip
unzip -q test.zip
rm train.zip
rm test.zip
cd ../
