"""When dealing with large files that require segmentation, it's necessary to process them in chunks."""

import argparse
import jieba
from tqdm import tqdm


def segment_text(input_file, output_file, batch_size=10000):
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open(input_file, 'r', encoding='utf-8') as f:
        with open(output_file, 'w', encoding='utf-8') as out_file:
            pbar = tqdm(total=total_lines, desc="Segmenting")
            while True:
                lines = f.readlines(batch_size)
                if not lines:
                    break

                segmented_lines = []
                for line in lines:
                    # Segmentation using jieba
                    words = jieba.cut(line.strip())
                    segmented_line = ' '.join(words)
                    segmented_lines.append(segmented_line + '\n')

                out_file.writelines(segmented_lines)
                pbar.update(len(lines))

            pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment Chinese text using jieba")
    parser.add_argument("-input_file", help="Input file containing Chinese text", required=True)
    parser.add_argument("-output_file", help="Output file to save segmented text", required=True)
    args = parser.parse_args()

    segment_text(args.input_file, args.output_file)
