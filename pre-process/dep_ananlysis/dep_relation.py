"""The input to this script is a corpus document, and the output is a set of binary relation tuples.
Segmentation must be performed using spaces to tokenize the entire document. A pre-trained model from HanLP is employed
to extract dependency syntactic relationships. The final generated binary relation tuples are deduplicated, yet the
directional edges persist. For instance, (A, B) represents the relationship B -> A, while (B, A) signifies the
 relationship A -> B."""

import hanlp
from tqdm import tqdm
from time import time

t1 = time()


def generate_relation_doc(dataset_address, edge_file):
    def generate_relation_single(seg_result, depen_result):
        relation_list = []
        relation_single = ()
        for index1, i in enumerate(depen_result):
            for _ in seg_result:
                if i[0] == 0:
                    pass
                else:
                    relation_single = (seg_result[index1], seg_result[i[0] - 1])
            relation_list.append(relation_single)
        relation_list = list(set(relation_list))
        return relation_list

    contents = []
    with open(dataset_address, encoding='UTF-8') as f:
        contents_raw = f.readlines()
        for i in contents_raw:
            contents.append(i.strip().split())
    t2 = time()
    print(f'Segment_time is：{t2 - t1}')

    """The word_list below concatenates all tokenized corpus entries into a single list, 
    eliminating duplicates. The result should encompass all string nodes present in the corpus"""
    word_list = []
    for j in contents:
        word_list.extend(j)  # 将所有的语料拼接在一个列表中
    word_list = set(word_list)  # 对拼接后的列表进行去重
    print(
        f'By segmenting the corpus using spaces, achieving tokenization, the length of the deduplicated dictionary is：{len(word_list)}')

    t3 = time()
    print(f'The time for tallying characters present in the corpus is：{t3 - t2}')
    # seg = hanlp.load(hanlp.pretrained.tok.UD_TOK_MMINILMV2L6)
    # dep = hanlp.load(hanlp.pretrained.dep.PTB_BIAFFINE_DEP_EN)
    dep = hanlp.load(hanlp.pretrained.dep.CTB9_DEP_ELECTRA_SMALL)

    t4 = time()
    print(f'Time to load pre_model is：{t4 - t3}')
    relation_result_list = []
    for index, corpus in enumerate(tqdm(contents)):
        if index <= 2000000:
            dep_result = dep(corpus, conll=False)
            relation_result = generate_relation_single(corpus, dep_result)
            relation_result_list.extend(relation_result)
        else:
            break
    t5 = time()
    relation_result_list = list(set(relation_result_list))
    t6 = time()
    print(f'The time taken for reducing the binary tuple pairs is:{t6 - t5}')
    print(f'The final count of binary tuple pairs formed is：{len(relation_result_list)}')

    relation_list = []
    print(f'the length of relation_result_list is:{len(relation_result_list)}')
    for index, i in enumerate(relation_result_list):
        if i != ():
            d = i[0] + ' ' + i[1] + '\n'
            relation_list.append(d)

    with open(edge_file, 'w', encoding='utf-8', ) as f:
        f.writelines(relation_list)


# generate_relation_doc(
#     r'C:\Users\Administrator\PycharmProjects\Thesis_Experiment\dep_ananlysis\dataset\iwslt14de-en\train.de',
#     r'C:\Users\Administrator\PycharmProjects\Thesis_Experiment\dep_ananlysis\dataset\iwslt14de-en\edge.csv')
# generate_relation_doc(
#     r'C:\Users\Administrator\PycharmProjects\Thesis_Experiment\dep_ananlysis\dataset\iwslt17de-en\train.de',
#     r'C:\Users\Administrator\PycharmProjects\Thesis_Experiment\dep_ananlysis\dataset\iwslt17de-en\edge.csv')
# generate_relation_doc(
#     r'C:\Users\Administrator\PycharmProjects\Thesis_Experiment\dep_ananlysis\dataset\iwslt20es-de\train.es',
#     r'C:\Users\Administrator\PycharmProjects\Thesis_Experiment\dep_ananlysis\dataset\iwslt20es-de\edge.csv')
# generate_relation_doc(
#     r'C:\Users\Administrator\PycharmProjects\Thesis_Experiment\dep_ananlysis\dataset\wmt14fr-en-ncv9\train.fr',
#     r'C:\Users\Administrator\PycharmProjects\Thesis_Experiment\dep_ananlysis\dataset\wmt14fr-en-ncv9\edge.csv')
generate_relation_doc(
    r'C:\Users\Administrator\PycharmProjects\Thesis_Experiment\dep_ananlysis\dataset\iwslt17zh-en\train.zh',
    r'C:\Users\Administrator\PycharmProjects\Thesis_Experiment\dep_ananlysis\dataset\iwslt17zh-en\edge.csv')
