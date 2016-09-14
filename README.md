# Readme

This repo applies character-based recurrent neural network to Yelp's review data.
The idea is based on Andrej Karpathy's blog post (http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
Instead of using Andrej's original code, I have re-implemented the network in Tensorflow.

## Content
The main code is in the review_generator folder.

The trained_model contains a model trained on GTX 1070 for 2 days, using the config in rnn_config.json.

Some samples of generated reviews with the trained model is in generated_review.txt

## How to run

First download the yelp academic dataset from https://www.yelp.com/dataset_challenge

Unzip the file and you should find a file called `yelp_academic_dataset_review.json`

We first parse the JSON file into a plain text file with reviews only:

$ python -m review_generator.data_preprocess --input-file yelp_academic_dataset_review.json

This will output to a file called all_reviews.txt

Then train the model:

$ python -m review_generator.train --input-file all_reviews.txt --model-config rnn_config.json

By default the model is saved to a file `char_rnn_model.ckpt`
After the training is done you can use it to generate some synthesized reviews:

$ python -m review_generator.generate --model-config rnn_config.json