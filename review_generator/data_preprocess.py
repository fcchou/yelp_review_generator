"""Process the Yelp academic dataset to aggregate the reviews"""
import json
import argparse

parser = argparse.ArgumentParser(description='Parse Yelp dataset and extract the review texts')
parser.add_argument('--input-file', dest='input_file', help='input Yelp review data', required=True)
parser.add_argument(
    '--output-file',
    dest='output_file',
    help='Output location for the processed review file',
    default='all_reviews.txt',
)
args = parser.parse_args()

with open(args.output_file, 'wb') as out, open(args.input_file) as infile:
    for line in infile:
        data = json.loads(line)
        review = data['text'].encode('ascii', 'ignore')
        review += b'\0'
        out.write(review)
