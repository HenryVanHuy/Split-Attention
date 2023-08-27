#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000


if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=zh
tgt=en
lang=zh-en
prep=UN-Conference
task=UN-Conference

tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

python chinese_token_jieba_small.py -input_file $orig/$lang/$task/train.tags_untoken.$lang.zh -output_file $orig/$lang/$task/train.tags.$lang.zh
python chinese_token_jieba_small.py -input_file $orig/$lang/$task/valid.tags_untoken.$lang.zh -output_file $orig/$lang/$task/valid.tags.$lang.zh
python chinese_token_jieba_small.py -input_file $orig/$lang/$task/test.tags_untoken.$lang.zh -output_file $orig/$lang/$task/test.tags.$lang.zh

echo "pre-processing train data...step2.."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat $orig/$lang/$task/$f | \
	perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done

echo "pre-processing valid data...step2.."
for l in $src $tgt; do
    f=valid.tags.$lang.$l
    tok=valid.tags.$lang.tok.$l

    cat $orig/$lang/$task/$f | \
	perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done

echo "pre-processing test data...step2.."
for l in $src $tgt; do
    f=test.tags.$lang.$l
    tok=test.tags.$lang.tok.$l
    cat $orig/$lang/$task/$f | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done

perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175
perl $CLEAN -ratio 1.5 $tmp/valid.tags.$lang.tok $src $tgt $tmp/valid.tags.$lang.clean 1 175

for l in $src $tgt; do
    perl $LC < $tmp/train.tags.$lang.clean.$l > $tmp/train.$l
	perl $LC < $tmp/valid.tags.$lang.clean.$l > $tmp/valid.$l
	perl $LC < $tmp/test.tags.$lang.tok.$l > $tmp/test.$l
done


TRAIN=$tmp/train.$lang
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done
