"""Segmenting small-scale Chinese language corpora"""

import argparse
import jieba
from tqdm import tqdm


def segment_text(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    words = jieba.cut(text)
    segmented_text = ' '.join(words)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(segmented_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment Chinese text using jieba")
    parser.add_argument("-input_file", help="Input file containing Chinese text", required=True)
    parser.add_argument("-output_file", help="Output file to save segmented text", required=True)
    args = parser.parse_args()

    with tqdm(total=100, desc="Segmenting") as pbar:
        segment_text(args.input_file, args.output_file)
        pbar.update(100)
