# Directory Structure

## fairseq：Running the foundational toolkit for machine translation.

## model-code：Core code related to the model.

```markup
SGSE: The method proposed in this paper.
	baseFunc.py: Fundamental methods for constructing DGL graphs, message passing, and building edges.
	buildGraph.py: Building DGL graphs.
	transformer_encoder.py: The method proposed in this paper (Encoder).
	transformer_decoder.py: Decoder file.
	transformer_layer.py: Network layer for self-attention construction.
baseline: Baseline method for comparison.
```

### Note：

```markup
1. Place baseFunc.py, buildGraph.py, transformer_encoder.py, and transformer_decoder.py in the fairseq\fairseq\models\transformer folder.
2. Place transformer_layer.py in the fairseq\fairseq\modules folder.
```

## pre-process：Preprocessing of parallel corpus data.

```markdown
chinese_token: Additional tokenization process for Chinese corpus.
chinese_token_jieba_big.py: Handling of large-scale Chinese corpora.
chinese_token_jieba_small.py: Handling of small-scale Chinese corpora.
corpus_process: Data preprocessing for the original corpus.
dep_analysis:
dep_relation.py: Dependency syntactic analysis.
extra_data_process:
split_via_length.py: Splitting based on corpus length.
```
