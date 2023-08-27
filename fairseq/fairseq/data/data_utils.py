# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import contextlib
import itertools
import logging
import re
import warnings
from typing import Optional, Tuple

import numpy as np
import torch

from fairseq.file_io import PathManager
from fairseq import utils
import os

logger = logging.getLogger(__name__)


# 通过文件的路径来解析语言对，返回源语和目标语类型
def infer_language_pair(path):
    """Infer language pair from filename: <split>.<lang1>-<lang2>.(...).idx"""
    src, dst = None, None
    for filename in PathManager.ls(path):
        parts = filename.split(".")  # 将文件名按照“.”进行切分
        if len(parts) >= 3 and len(parts[1].split("-")) == 2:  # 如果文件名切分后的长度为3，切分的第二块按照“-”切分有两个词（代表源语和目标语）
            return parts[1].split("-")  # 返回切分的两个部分
    return src, dst


# collate 整理;校对。将一个一维tensor的列表转换成一个填充的二维列表，并实现在tensor中添加pad和结束符。
def collate_tokens(
        values,
        pad_idx,
        eos_idx=None,
        left_pad=False,
        move_eos_to_beginning=False,
        pad_to_length=None,
        pad_to_multiple=1,
        pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    # values表示一个batch输入的token，是一个列表，列表内的元素为一维tensor，每个tensor表示一条语料对应的token
    size = max(v.size(0) for v in values)  # 得到的size表示当前batch每个token的最大长度，也就是最大语料长度

    # 如果pad_to_length为None，则size从上面一句获取，否则size等于size和pad_to_length的最大值，pad_to_length为最大填充长度
    size = size if pad_to_length is None else max(size, pad_to_length)

    # 下面这个if判断没有运行，暂时不看，是对size值做调整
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    # 当pad_to_bsz为None时，batch_size的值等于values的长度，也就是列表中有多少个tensor。
    # 如果pad_to_bsz不为None，则batch_size的值等于values长度和pad_to_bsz的最大值，也就是可能会做填充
    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)

    # pad_dix的值为1，说明这是pad所对应的token序号，也就是[pad]所对应的token值为1，这和paddle里面是一样的
    # res为一个二维tensor，shape为batch_size*size，里面全部用1进行填充
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()  # 用来声明相等条件是否为真，如果不相等就会触发AssertionError异常

        # 由于move_eos_to_beginning的初始值为False，因此先执行底下的else语句
        if move_eos_to_beginning:
            # 模型训练前，未执行下面的if语句
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                # 如果没有指定eos_idx，则使用src中的最后一个token
                dst[0] = src[-1]
            # 模型训练前，第二步执行了下面的else语句,eos表示句子的末尾标识
            # 如果指定了eos_idx，则使用eos_idx作为dst开头的token，这里eos_idx的值为2，也就是结束标记的token值为2
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]  # 让dst从第二位到最后一位的值，等于src中从第一位到倒数第二位的值

        # 模型训练前，第一步执行了下面的else语句
        else:
            dst.copy_(src)  # 先将src的tensor值复制并覆盖到dst中，相当于生成一个src的副本

    for i, v in enumerate(values):
        # 由于left_pad为False，因此copy_tensor第二个实参为res[i][: len(v)]，表示从语料右侧进行填充
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][: len(v)])
    return res


# 用于加载固定索引的数据，输入为源语料和目标语料的地址
def load_indexed_dataset(
        path, dictionary=None, dataset_impl=None, combine=False, default="cached"
):
    """A helper function for loading indexed datasets.

    Args:
        path (str): path to indexed dataset (e.g., 'data-bin/train')
        dictionary (~fairseq.data.Dictionary): data dictionary
        dataset_impl (str, optional): which dataset implementation to use. If
            not provided, it will be inferred automatically. For legacy indexed
            data we use the 'cached' implementation by default.
        combine (bool, optional): automatically load and combine multiple
            datasets. For example, if *path* is 'data-bin/train', then we will
            combine 'data-bin/train', 'data-bin/train1', ... and return a
            single ConcatDataset instance.
    """
    # path--索引数据集的路径；dictionary--数据集字典；
    # dataset_impl--要使用哪个数据集实现，如果未提供，将自动推断。对于旧的数据索引，默认使用“缓存”实现
    # combine--自动加载和组合多个数据集。例如，如果path是data-bin/train，那么我们将组合data-bin/train、data-bin/train1
    # 并返回单个ConcatDataset实例
    import fairseq.data.indexed_dataset as indexed_dataset
    from fairseq.data.concat_dataset import ConcatDataset
    # 训练中path变量被打印了两次，
    # 一次是data-bin/iwslt14.tokenized.de-en\train.de-en.de，
    # 一次是data-bin/iwslt14.tokenized.de-en\train.de-en.en。说明针对源语料和目标语料的路径，分别进行了传参
    # print(f'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&{path}')

    datasets = []
    for k in itertools.count():  # 创建一个从0开始，步长为1的无穷序列
        # 如果k的值大于0，也就是不止1个path，就将k或者“”拼接到path上
        # 通过打印发现,在加载源语和目标语时，地址是相同的
        path_k = path + (str(k) if k > 0 else "")
        try:
            # 通过打印发现，path_k和path是相同的
            path_k = indexed_dataset.get_indexed_dataset_to_local(path_k)
        except Exception as e:
            if "StorageException: [404] Path not found" in str(e):
                logger.warning(f"path_k: {e} not found")
            else:
                raise e
        # 打印的dataset_impl和dataset_impl_k都是None
        dataset_impl_k = dataset_impl
        if dataset_impl_k is None:
            # 打印的dataset_impl_k的值为mmap，这个似乎是内存映射方式
            dataset_impl_k = indexed_dataset.infer_dataset_impl(path_k)

        # 打印的dataset是一个MMapIndexedDataset对象，这里dictionary的长度分别为8848和6640，说明已经读取到path中的字典文件信息
        # 打印的dataset分别为7284/7284、160250/160250，说明已经读取到path中的验证集和训练集信息
        dataset = indexed_dataset.make_dataset(
            path_k,
            impl=dataset_impl_k or default,
            fix_lua_indexing=True,
            dictionary=dictionary,
        )

        # 如果path中没有数据集，就中断训练
        if dataset is None:
            break
        # 控制台打印信息
        logger.info("loaded {:,} examples from: {}".format(len(dataset), path_k))
        # 将dataset进行拼接
        datasets.append(dataset)
        if not combine:
            break
    if len(datasets) == 0:
        return None
    # 一般情况下只有一个数据集
    elif len(datasets) == 1:
        return datasets[0]  # 在我的实验中，调用的是这种情况
    else:
        return ConcatDataset(datasets)


# numpy_seed函数通过addl_seeds参数来生成numpy的seed
@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    # 上下文管理器，为Numpy PRNG指定seeds，这里应当是为了生成一个seed
    # 打印的seed显示了两次，分别为1和4，addl_seeds为一个空元组
    # 下面的if语句没有执行
    if seed is None:
        yield
        return
    # 下面的if语句没有执行
    if len(addl_seeds) > 0:
        # 打印的addl_seeds为一个空的元组
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()  # 获取随机生成器np.random的状态
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)  # 使随机生成器random保持相同的状态


# 打印发现，collect_filtered函数没有执行
def collect_filtered(function, iterable, filtered):
    """
    Similar to :func:`filter` but collects filtered elements in ``filtered``.

    Args:
        function (callable): function that returns ``False`` for elements that
            should be filtered
        iterable (iterable): iterable to filter
        filtered (list): list to store filtered elements
    """
    for el in iterable:
        if function(el):
            yield el
        else:
            filtered.append(el)


# 打印发现，_filter_by_size_dynamic函数没有执行
def _filter_by_size_dynamic(indices, size_fn, max_positions, raise_exception=False):
    def compare_leq(a, b):
        return a <= b if not isinstance(a, tuple) else max(a) <= b

    def check_size(idx):
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            return size_fn(idx) <= max_positions
        elif isinstance(max_positions, dict):
            idx_size = size_fn(idx)
            assert isinstance(idx_size, dict)
            intersect_keys = set(max_positions.keys()) & set(idx_size.keys())
            return all(
                all(
                    a is None or b is None or a <= b
                    for a, b in zip(idx_size[key], max_positions[key])
                )
                for key in intersect_keys
            )
        else:
            # For MultiCorpusSampledDataset, will generalize it later
            if not isinstance(size_fn(idx), Iterable):
                return all(size_fn(idx) <= b for b in max_positions)
            return all(
                a is None or b is None or a <= b
                for a, b in zip(size_fn(idx), max_positions)
            )

    ignored = []
    itr = collect_filtered(check_size, indices, ignored)
    indices = np.fromiter(itr, dtype=np.int64, count=-1)
    return indices, ignored


# 打印发现，filter_by_size函数没有执行
def filter_by_size(indices, dataset, max_positions, raise_exception=False):
    """
    [deprecated] Filter indices based on their size.
    Use `FairseqDataset::filter_indices_by_size` instead.

    Args:
        indices (List[int]): ordered list of dataset indices
        dataset (FairseqDataset): fairseq dataset instance
        max_positions (tuple): filter elements larger than this size.
            Comparisons are done component-wise.
        raise_exception (bool, optional): if ``True``, raise an exception if
            any elements are filtered (default: False).
    """
    warnings.warn(
        "data_utils.filter_by_size is deprecated. "
        "Use `FairseqDataset::filter_indices_by_size` instead.",
        stacklevel=2,
    )
    if isinstance(max_positions, float) or isinstance(max_positions, int):
        if hasattr(dataset, "sizes") and isinstance(dataset.sizes, np.ndarray):
            ignored = indices[dataset.sizes[indices] > max_positions].tolist()
            indices = indices[dataset.sizes[indices] <= max_positions]
        elif (
                hasattr(dataset, "sizes")
                and isinstance(dataset.sizes, list)
                and len(dataset.sizes) == 1
        ):
            ignored = indices[dataset.sizes[0][indices] > max_positions].tolist()
            indices = indices[dataset.sizes[0][indices] <= max_positions]
        else:
            indices, ignored = _filter_by_size_dynamic(
                indices, dataset.size, max_positions
            )
    else:
        indices, ignored = _filter_by_size_dynamic(indices, dataset.size, max_positions)

    if len(ignored) > 0 and raise_exception:
        raise Exception(
            (
                "Size of sample #{} is invalid (={}) since max_positions={}, "
                "skip this example with --skip-invalid-size-inputs-valid-test"
            ).format(ignored[0], dataset.size(ignored[0]), max_positions)
        )
    if len(ignored) > 0:
        logger.warning(
            (
                "{} samples have invalid sizes and will be skipped, "
                "max_positions={}, first few sample ids={}"
            ).format(len(ignored), max_positions, ignored[:10])
        )
    return indices


# 过滤样本索引列表。删除那些长度超过max_size的语料
# src_sizes--打印的值为一维数组，长度为16250，应当是源语对应的语料长度列表，列表中的每个值表示当前语料的长度；
# tgt_sizes--打印的值为一维数组，长度为16250，应当是目标语对应的语料长度列表，列表中的每个值表示当前语料的长度；
# indices--打印的值为一维数组，长度为16250，应当是输入模型的语料token对应的index
# max_sizes值为(1024,1024)，应当指的是源语和目标语允许的最大token长度
# 返回值为indices--经过过滤的样例数组；
def filter_paired_dataset_indices_by_size(src_sizes, tgt_sizes, indices, max_sizes):
    """Filter a list of sample indices. Remove those that are longer
        than specified in max_sizes.

    Args:
        indices (np.array): original array of sample indices
        max_sizes (int or list[int] or tuple[int]): max sample size,
            can be defined separately for src and tgt (then list or tuple)

    Returns:
        np.array: filtered sample array
        list: list of removed indices
    """
    # 这个if没有执行，因为max_size有值，所以不执行
    if max_sizes is None:
        return indices, []
    # 这个if没有执行，因为max_sizes是元组，所以不执行
    if type(max_sizes) in (int, float):
        max_src_size, max_tgt_size = max_sizes, max_sizes
    # 程序执行了下方的else语句，将max_sizes元组中的两个值分别赋给max_src_size和max_tgt_size
    else:
        max_src_size, max_tgt_size = max_sizes
    # 这个if没有执行，因为tgt_sizes存在语料，不是None，所以不执行
    if tgt_sizes is None:
        ignored = indices[src_sizes[indices] > max_src_size]
    # 程序执行了下方的else语句，打印出的ignored是一个空列表，说明没有数据被舍弃
    else:
        ignored = indices[
            (src_sizes[indices] > max_src_size) | (tgt_sizes[indices] > max_tgt_size)
            ]
    # 这个if没有执行，因为没有数据被舍弃，因此不执行
    if len(ignored) > 0:
        # 这个if没有执行,如果执行的话，就是取出长度短于max_src_size的语料
        if tgt_sizes is None:
            indices = indices[src_sizes[indices] <= max_src_size]
        # 这个else没有执行
        else:
            indices = indices[
                (src_sizes[indices] <= max_src_size)
                & (tgt_sizes[indices] <= max_tgt_size)
                ]
    return indices, ignored.tolist()


# mini-batch按照size进行分组，每个batch中可以包含不同长度的序列，返回值为一个二维数组，长度为1103，表示1103个batch，batch_size不固定
# 但是经过分析，这里的函数输出仍然不是最终进入模型的语料顺序，每个mini-batch内的顺序没有变，但是mini-batch的顺序发生了变化
# indices的值为一个一维数组，表示数据集索引的有序列表，长度为160250；
# nu_tokens_fn是一个LanguagePairDataset对象，是一个函数，用来返回给定索引的token数；
# num_tokens_vec的值为160250，表示token的个数；
# max_tokens的值为4096，表示一个batch语料允许的最大token长度
# max_sentences的值为None，表示每个batch允许的最大语料个数；
# required_batch_size_multiple的值为8，表示要求batch_size小于N或者N的倍数（默认值为1）；
# fixed_shapes的值为None，如果给定，将仅使用给定shape来创建batches，则max_sentences和required_batch_size_multiple参数将被忽略
def batch_by_size(
        indices,
        num_tokens_fn,
        num_tokens_vec=None,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
        fixed_shapes=None,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        num_tokens_vec (List[int], optional): precomputed vector of the number
            of tokens for each index in indices (to enable faster batch generation)
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be less than N or a multiple of N (default: 1).
        fixed_shapes (List[Tuple[int, int]], optional): if given, batches will
            only be created with the given shapes. *max_sentences* and
            *required_batch_size_multiple* will be ignored (default: None).
    """
    try:
        # 打印发现，导入包没有问题，执行成功
        from fairseq.data.data_utils_fast import (
            batch_by_size_fn,
            batch_by_size_vec,
            batch_fixed_shapes_fast,
        )
    except ImportError:
        raise ImportError(
            "Please build Cython components with: "
            "`python setup.py build_ext --inplace`"
        )
    except ValueError:
        raise ValueError(
            "Please build (or rebuild) Cython components with `python setup.py build_ext --inplace`."
        )

    # added int() to avoid TypeError: an integer is required
    # 将max_tokens的值转换成int类型
    max_tokens = int(max_tokens) if max_tokens is not None else -1
    max_sentences = max_sentences if max_sentences is not None else -1
    bsz_mult = required_batch_size_multiple

    # 打印发现，下面的if未执行，因为indices是ndarray格式，因此不执行
    if not isinstance(indices, np.ndarray):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    # 打印发现，下面的if未执行，因为num_tokens_vec是ndarray格式，因此不执行
    if num_tokens_vec is not None and not isinstance(num_tokens_vec, np.ndarray):
        num_tokens_vec = np.fromiter(num_tokens_vec, dtype=np.int64, count=-1)

    # 打印发现，执行了下面的if，因为fixed_shapes是None，因此执行
    if fixed_shapes is None:
        # 打印发现，下面的if未执行，因为num_tokens_vec的值为160250，因此不执行
        if num_tokens_vec is None:

            return batch_by_size_fn(
                indices,
                num_tokens_fn,
                max_tokens,
                max_sentences,
                bsz_mult,
            )
        # 打印发现，执行了下面的else
        else:
            # 该函数的返回值为一个二维数组，每个一维数组表示一个batch输入的语料index，共有1103个batch
            # A = batch_by_size_vec(indices, num_tokens_vec, max_tokens, max_sentences, bsz_mult)
            # print(
            #     f'the token_index in data_utils.py is:{A[0]}')
            # for index1,i in enumerate(A):
            #     for index2,j in enumerate(i):
            #         if j==817:
            #             print(f'the location of the sentence is{[index1,index2]}')

            return batch_by_size_vec(
                indices,
                num_tokens_vec,
                max_tokens,
                max_sentences,
                bsz_mult,
            )
    # 打印发现，下面的else未执行
    else:
        fixed_shapes = np.array(fixed_shapes, dtype=np.int64)
        sort_order = np.lexsort(
            [
                fixed_shapes[:, 1].argsort(),  # length
                fixed_shapes[:, 0].argsort(),  # bsz
            ]
        )
        fixed_shapes_sorted = fixed_shapes[sort_order]
        return batch_fixed_shapes_fast(indices, num_tokens_fn, fixed_shapes_sorted)


# 这个函数在其他代码中有调用，因此不能缺少，但以下各if..else都没有执行，sentence也没有返回值，说明函数并没有被执行
# 应当是对于原始语料的字符预处理
def post_process(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == "wordpiece":
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == "letter":
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "silence":
        import re
        sentence = sentence.replace("<SIL>", "")
        sentence = re.sub(" +", " ", sentence).strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol in {"subword_nmt", "@@ ", "@@"}:
        if symbol == "subword_nmt":
            symbol = "@@ "
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    elif symbol == "none":
        pass
    elif symbol is not None:
        raise NotImplementedError(f"Unknown post_process option: {symbol}")
    return sentence


# 被调用，但未执行
def compute_mask_indices(
        shape: Tuple[int, int],
        padding_mask: Optional[torch.Tensor],
        mask_prob: float,
        mask_length: int,
        mask_type: str = "static",
        mask_other: float = 0.0,
        min_masks: int = 0,
        no_overlap: bool = False,
        min_space: int = 0,
        require_same_masks: bool = True,
        mask_dropout: float = 0.0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len and require_same_masks:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        if mask_dropout > 0:
            num_holes = np.rint(len(mask_idc) * mask_dropout).astype(int)
            mask_idc = np.random.choice(
                mask_idc, len(mask_idc) - num_holes, replace=False
            )

        mask[i, mask_idc] = True
    return mask


# 打印发现，get_mem_usage函数没有被调用，也未被执行
def get_mem_usage():
    try:
        import psutil

        mb = 1024 * 1024
        return f"used={psutil.virtual_memory().used / mb}Mb; avail={psutil.virtual_memory().available / mb}Mb"
    except ImportError:
        return "N/A"


# lens: torch.LongTensor
# returns: torch.BoolTensor
# 被调用，但未执行
def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


# lens: torch.LongTensor
# returns: torch.BoolTensor
# 被调用，但未执行
def lengths_to_mask(lens):
    return ~lengths_to_padding_mask(lens)


# 被调用，但未执行
def get_buckets(sizes, num_buckets):
    buckets = np.unique(
        np.percentile(
            sizes,
            np.linspace(0, 100, num_buckets + 1),
            interpolation="lower",
        )[1:]
    )
    return buckets


# 被调用，但未执行
def get_bucketed_sizes(orig_sizes, buckets):
    sizes = np.copy(orig_sizes)
    assert np.min(sizes) >= 0
    start_val = -1
    for end_val in buckets:
        mask = (sizes > start_val) & (sizes <= end_val)
        sizes[mask] = end_val
        start_val = end_val
    return sizes


# 该函数用于发现目录下是否存在额外的验证数据集，并将其文件拼接成一个集合
def _find_extra_valid_paths(dataset_path: str) -> set:
    paths = utils.split_paths(dataset_path)  # 将字符串形式的文件路径转换为外面套个列表，如果多个地址，可以放在列表的不同位置
    all_valid_paths = set()
    for sub_dir in paths:  # 由于只传了一个地址，因此这里sub_dir只有一个值
        contents = PathManager.ls(sub_dir)  # contents是获取的sub_dir底下的文件名
        # 如果目录下有多个valid验证文件，就拼成一个列表，这里valid_path是一个空列表
        valid_paths = [c for c in contents if re.match("valid*[0-9].*", c) is not None]
        # all_valid_paths是一个空集合
        all_valid_paths |= {os.path.basename(p) for p in valid_paths}
    # Remove .bin, .idx etc
    # roots是一个空集合
    roots = {os.path.splitext(p)[0] for p in all_valid_paths}
    return roots


# 这个函数被调用，被执行，但里面的两个if语句都没有被执行
def raise_if_valid_subsets_unintentionally_ignored(train_cfg) -> None:
    """Raises if there are paths matching 'valid*[0-9].*' which are not combined or ignored."""
    if (
            train_cfg.dataset.ignore_unused_valid_subsets
            or train_cfg.dataset.combine_valid_subsets
            or train_cfg.dataset.disable_validation
            or not hasattr(train_cfg.task, "data")
    ):
        return
    other_paths = _find_extra_valid_paths(train_cfg.task.data)
    specified_subsets = train_cfg.dataset.valid_subset.split(",")
    ignored_paths = [p for p in other_paths if p not in specified_subsets]
    if ignored_paths:
        advice = "Set --combine-val to combine them or --ignore-unused-valid-subsets to ignore them."
        msg = f"Valid paths {ignored_paths} will be ignored. {advice}"
        raise ValueError(msg)
