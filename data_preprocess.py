# Process the academic dataset for training/test split
import simplejson as json
import random

train = open('review_training.txt', 'w')
test = open('review_test.txt', 'w')

for line in open('yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json'):
    data = json.loads(line)
    review = data['text'].encode('ascii', 'ignore') + '\n\0'
    if random.random() < 0.8:
        train.write(review)
    else:
        test.write(review)
train.close()
test.close()
