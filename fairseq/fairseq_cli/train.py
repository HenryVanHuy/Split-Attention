#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")  # 这个就是程序运行开始时的打印信息

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from fairseq import checkpoint_utils, options, quantization_utils, tasks, utils
from fairseq.data import data_utils, iterators
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.initialize import add_defaults
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap
from fairseq.distributed import utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer


def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)
    add_defaults(cfg)

    if (
            distributed_utils.is_master(cfg.distributed_training)
            and "job_logging_cfg" in cfg
    ):
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
            cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)  # 会发现这个常用的seed被设置为了1
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)  # 这句应当是判断模型文件的保存位置

    # Print args
    logger.info(cfg)  # 打印所有的参数

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)  # 启动一个任务，例如翻译、语言模型等

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))  # 这里应当是建立一个分布式的模型
    else:
        model = task.build_model(cfg.model)  # 否则，建立一个普通的单机模型
    criterion = task.build_criterion(cfg.criterion)  # 确定损失函数类型
    logger.info(model)  # 打印模型的网络结构
    logger.info("task: {}".format(task.__class__.__name__))  # 打印任务类型
    logger.info("model: {}".format(model.__class__.__name__))  # 打印模型类型
    logger.info("criterion: {}".format(criterion.__class__.__name__))  # 打印损失函数类型
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(
                p.numel() for p in model.parameters() if not getattr(p, "expert", False)
            ),
            sum(
                p.numel()
                for p in model.parameters()
                if not getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )  # 打印共享模型的参数量

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(
                p.numel()
                for p in model.parameters()
                if getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )  # 打印专家模型的参数量

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # 加载有效的数据集（根据最新的检查点加载以下训练数据）
    # We load the valid dataset AFTER building the model
    # 在构建模型后加载有效的数据集
    data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
    if cfg.dataset.combine_valid_subsets:
        task.load_dataset("valid", combine=True, epoch=1)
    else:
        for valid_sub_split in cfg.dataset.valid_subset.split(","):
            task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # (optionally) Configure quantization （可选）配置量化
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer 构建一个训练器
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)  # 非并行时的trainer
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)  # 并行时的trainer

    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )  # 打印GPU/TPU信息

    # 通过用户输入的命令行中获取max_tokens和batch_size信息，并将其打印
    logger.info(
        "max tokens per device = {} and max sentences per device = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )  # 打印max_token和batch_size信息

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    # 加载最新的checkpoint，并恢复相应的训练迭代器
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )
    if cfg.common.tpu:
        import torch_xla.core.xla_model as xm

        xm.rendezvous("load_checkpoint")  # wait for all workers 使用TPU，并等待所有的workers准备就绪

    max_epoch = cfg.optimization.max_epoch or math.inf  # 获取最大的epoch值

    # 以下应当是训练过程，也包含了数据处理的环节
    lr = trainer.get_lr()  # 获取学习率

    train_meter = meters.StopwatchMeter()
    train_meter.start()
    # 如果当前的学习率小于设定的最小终止学习率，则会终止训练过程，给打印相关信息
    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break

        # train for one epoch 调用train函数，进行一个epoch的训练，在训练结束后进行一次验证，并返回验证的loss值
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        # 仅使用第一次验证损失来更新学习率
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch 切分数据：获取下一个epoch的训练迭代器
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets 对于切分数据，不缓存epoch迭代器（啥意思？）
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    # 统计所有epoch的模型训练时间
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))

    # ioPath implementation to wait for all asynchronous file writes to complete.
    # ioPath实现等待所有异步文件写入完成（感觉是和）
    if cfg.checkpoint.write_checkpoints_asynchronously:
        logger.info(
            "ioPath PathManager waiting for all asynchronous checkpoint "
            "writes to finish."
        )
        PathManager.async_close()
        logger.info("ioPath PathManager finished waiting.")  #


# 判断是否需要提前终止训练，大体上是通过判断验证loss值的大小变化
def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    # 如果在当前epoch中未进行验证，则跳过检查
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


# train函数训练模型一个epoch，并且返回验证集的损失值以及是否终止训练。函数的输入为用户输入参数、Trainer(这个暂时未知,是导入的内容)、tasks.FairseqTask（也未知）、epoch_itr（这个应当是迭代次数）
@metrics.aggregate("train")
def train(
        cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator 初始化数据迭代器
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        # 判断是否在不同的GPU间打乱批次，如果为false，则降低整体随机性并可能影响精度，但不需要重新读取数据
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
        # curriculum为N时，则不用对前N个epoch的数据进行数据打乱。而next_epoch_idx应该是下一个epoch的序号
    )

    # 此处应当是更新间隔
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )

    # 迭代器
    itr = iterators.GroupedIterator(
        itr,
        update_freq,
        skip_remainder_batch=cfg.optimization.skip_remainder_batch,
    )

    # 以下是分布式和tpu训练部分，暂时不看
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        aim_repo=(
            cfg.common.aim_repo
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        aim_run_hash=(
            cfg.common.aim_run_hash
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        aim_param_checkpoint_dir=cfg.checkpoint.save_dir,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)  # 在每个epoch开始时调用

    valid_subsets = cfg.dataset.valid_subset.split(",")  # 验证子集
    # print(f'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&{valid_subsets}')  # 打印出来的值为['valid']
    should_stop = False  # 默认应当是不会模型早停
    num_updates = trainer.get_num_updates()

    # 以下代码应当是NMT训练开始时打印的那段话，且控制整个训练过程的log打印
    logger.info("Start iterating over samples")
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
                "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()

        # 通过调用validate_and_save函数，获得验证损失值和should_stop
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop  # 这个函数的返回值居然是验证集loss和是否训练终止


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    # 移除其他历史版本的命名空间，作为替代使用一个单一的args来传参
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
        cfg: DictConfig,
        trainer: Trainer,
        task: tasks.FairseqTask,
        epoch_itr,
        valid_subsets: List[str],
        end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on) 下面的几个if都是用来判断何时停止训练或保存模型

    should_stop = False

    # 如果总的更新数大于设定的最大更新数时，会停止训练
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)

    # 如果训练的小时数大于设定的终止小时数时，会停止训练
    if (
            cfg.optimization.stop_time_hours > 0
            and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    # do_save表示需要保存模型的条件（共3种情况）
    do_save = (
            (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
            or should_stop
            or (
                    cfg.checkpoint.save_interval_updates > 0
                    and num_updates > 0
                    and num_updates % cfg.checkpoint.save_interval_updates == 0
                    and num_updates >= cfg.dataset.validate_after_updates
            )
    )

    # do_validate表示需要进行模型验证的条件
    do_validate = (
            (
                    (not end_of_epoch and do_save)  # validate during mid-epoch saves 在训练epoch保存时进行验证
                    or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
                    or should_stop
                    or (
                            cfg.dataset.validate_interval_updates > 0
                            and num_updates > 0
                            and num_updates % cfg.dataset.validate_interval_updates == 0
                    )
            )
            and not cfg.dataset.disable_validation
            and num_updates >= cfg.dataset.validate_after_updates
    )

    # Validate
    valid_losses = [None]

    # 如果进行验证，计算验证损失值
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)  # 通过调用validate函数，获取验证损失的值

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint 模型的保存
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


# 在验证集上进行模型评估，并返回损失值
def validate(
        cfg: DictConfig,
        trainer: Trainer,
        task: tasks.FairseqTask,
        epoch_itr,
        subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset_idx, subset in enumerate(subsets):
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            aim_repo=(
                cfg.common.aim_repo
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            aim_run_hash=(
                cfg.common.aim_run_hash
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            aim_param_checkpoint_dir=cfg.checkpoint.save_dir,
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        # create a new root metrics aggregator so validation metrics 创建新的root精度聚合器，以便验证精度
        # don't pollute other aggregators (e.g., train meters) 不污染其他的聚合器
        with metrics.aggregate(new_root=True) as agg:
            for i, sample in enumerate(progress):
                if (
                        cfg.dataset.max_valid_steps is not None
                        and i > cfg.dataset.max_valid_steps
                ):
                    break
                trainer.valid_step(sample)

        # log validation stats 日志验证统计信息
        # only tracking the best metric on the 1st validation subset 仅追踪第一个验证子集上的最佳精确率

        tracking_best = subset_idx == 0
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values(), tracking_best)  # 通过get_valid_stats函数获得验证状态

        if hasattr(task, "post_validate"):
            task.post_validate(trainer.get_model(), stats, agg)

        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(
        cfg: DictConfig,
        trainer: Trainer,
        stats: Dict[str, Any],
        tracking_best: bool,
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if tracking_best and hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def cli_main(
        modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(
            f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}"
        )

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()


if __name__ == "__main__":
    cli_main()
