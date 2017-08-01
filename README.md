# Yelp Review Generator

This repo applies character-based recurrent neural network to Yelp's review data.
The idea is based on [Andrej Karpathy's blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
Instead of using Andrej's original code, I have re-implemented the network in Tensorflow.

## Content
The main code is in the review_generator folder.

The trained_model contains a model trained on GTX 1070 for 2 days, using the config in rnn_config.json.

You may find examples of generated reviews with the trained model in [generated_review.txt](./generated_review.txt).

Some funny examples:

I don't know why I've really dealt with this club. However you'll see some of these gems instead> a repeat blad quinen to reds, welcoming, meat, (will NEVER know despite cheese mediterranean...don't come here for wings) This building requires an empty reservation of food! 5 to 4 stars. If you want to go all of the way, I personally must say that will tell you I'll am sure there are more copys who order rub in day at home.

The staff is always always friendly and nice. And you refer chances and offer tips. One' needed a bright business training hands down. If you're coming out of my way for a booth before you call it a place charge, I've also read printing new ones, maybe we will love the bar pils in mind or tired. Just give these pictures and treat your main dammion, but I have experiences everything they have.

## How to run

1. Download the [yelp academic dataset](https://www.yelp.com/dataset_challenge)

2. Unzip the file and you should find a file called `yelp_academic_dataset_review.json`

3. We first parse the JSON file into a plain text file with reviews only:

    `$ python -m review_generator.data_preprocess --input-file yelp_academic_dataset_review.json`

    This will output to a file called `all_reviews.txt`

4. Then train the model:

    `$ python -m review_generator.train --input-file all_reviews.txt --model-config rnn_config.json`

    By default the model is saved to a file `char_rnn_model.ckpt`

5. After the training is done you can use it to generate some synthesized reviews:

    `$ python -m review_generator.generate --model-config rnn_config.json`
