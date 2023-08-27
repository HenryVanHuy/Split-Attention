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
prep=iwslt17.tokenized.zh-en
year=IWSLT17

tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep
cd $orig

cd ..

echo "pre-processing train data...step1.."
for l in $src $tgt; do
    f=train_raw.tags.$lang.$l
    cat $orig/$lang/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
	grep -v '<doc docid' | \
	grep -v '<speaker>' | \
	grep -v '<reviewer' | \
	grep -v '<translator' | \
	grep -v '<\/doc>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>TED Talk Subtitles and Transcript://g' | \
    sed -e 's/<\/description>//g' \
	> $orig/$lang/train.tags_untoken.$lang.$l 
    echo ""
done

echo "pre-processing valid/test data...step1.."
for l in $src $tgt; do
    for o in `ls $orig/$lang/$year.TED*.$l.xml`; do
    fname=${o##*/}
    f=$orig/$lang/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > $f
    echo ""
    done
done

echo "creating valid, test..."
for l in $src $tgt; do
    cat $orig/$lang/$year.TED.dev2010.$lang.$l \
        $orig/$lang/$year.TED.tst2010.$lang.$l \
        $orig/$lang/$year.TED.tst2011.$lang.$l \
        > $orig/$lang/valid.tags_untoken.$lang.$l
		
	cat $orig/$lang/$year.TED.tst2012.$lang.$l \
        $orig/$lang/$year.TED.tst2013.$lang.$l \
        $orig/$lang/$year.TED.tst2014.$lang.$l \
		$orig/$lang/$year.TED.tst2015.$lang.$l \
        > $orig/$lang/test.tags_untoken.$lang.$l
done

for filename in orig/$lang/*_untoken.zh-en.en; do
    base=$(basename "$filename")
    new_filename=${base/_untoken/}
    new_path="orig/$lang/$new_filename"
    mv "$filename" "$new_path"
    echo "修改文件名: $filename -> $new_path"
done
python chinese_token_jieba_small.py -input_file $orig/$lang/train.tags_untoken.$lang.zh -output_file $orig/$lang/train.tags.$lang.zh
python chinese_token_jieba_small.py -input_file $orig/$lang/valid.tags_untoken.$lang.zh -output_file $orig/$lang/valid.tags.$lang.zh
python chinese_token_jieba_small.py -input_file $orig/$lang/test.tags_untoken.$lang.zh -output_file $orig/$lang/test.tags.$lang.zh

echo "pre-processing train data...step2.."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat $orig/$lang/$f | \
	perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done

echo "pre-processing valid data...step2.."
for l in $src $tgt; do
    f=valid.tags.$lang.$l
    tok=valid.tags.$lang.tok.$l

    cat $orig/$lang/$f | \
	perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done

echo "pre-processing test data...step2.."
for l in $src $tgt; do
    f=test.tags.$lang.$l
    tok=test.tags.$lang.tok.$l
    cat $orig/$lang/$f | \
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
