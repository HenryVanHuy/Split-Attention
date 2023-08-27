"""Splitting parallel corpora according to specified sentence lengths - lengths"""


# en_file: Source language corpus address; de_file: Target language corpus address
def split_sentences_by_length(en_file, de_file):
    lengths = [(0, 10), (10, 15), (15, 21), (21, 32), (32, float('inf'))]

    # Create a list for storing the segmented sentences
    sentence_lists = [[] for _ in range(len(lengths))]

    with open(en_file, 'r', encoding='utf-8') as en, open(de_file, 'r', encoding='utf-8') as de:
        en_sentences = en.readlines()
        de_sentences = de.readlines()

    # Distribute sentences to corresponding lists based on their lengths
    for en_sentence, de_sentence in zip(en_sentences, de_sentences):
        length = len(en_sentence.strip().split())
        for i, (min_len, max_len) in enumerate(lengths):
            if min_len <= length < max_len:
                sentence_lists[i].append((en_sentence, de_sentence))
                break

    # Write the segmented sentences into different files
    for i, (min_len, max_len) in enumerate(lengths):
        en_output_file = f"test.zh_{min_len}-{max_len}.txt"
        de_output_file = f"test.en_{min_len}-{max_len}.txt"
        with open(en_output_file, 'w', encoding='utf-8') as en_output, open(de_output_file, 'w',
                                                                            encoding='utf-8') as de_output:
            for en_sentence, de_sentence in sentence_lists[i]:
                en_output.write(en_sentence)
                de_output.write(de_sentence)


# IWSLT17 Chinese to English Translation Task
zh_file_path = r'C:\Users\Administrator\PycharmProjects\Thesis_Experiment\extra_data_process\IWSLT17-zh-en\test.zh'
en_file_path = r'C:\Users\Administrator\PycharmProjects\Thesis_Experiment\extra_data_process\IWSLT17-zh-en\test.en'

split_sentences_by_length(zh_file_path, en_file_path)
