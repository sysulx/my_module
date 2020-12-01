# 修改说明：
# 功能大幅度改动
# 没有试验在分布式，多卡并行，TPU，以及断点保存继续训练等情况下是否适用

import os
import numpy as np
import torch
import torch.nn as nn
import logging
import json
import copy
import time
import re
import shutil

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple
from packaging import version
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torch.optim.lr_scheduler  import LambdaLR
from overrides import overrides
from tqdm.auto import tqdm, trange

from my_module.base_model import Model
from my_module.my_training_args import TrainingArguments
from transformers import PreTrainedModel, DataCollator, EvalPrediction, is_apex_available, is_torch_tpu_available
from transformers import Trainer as T_Trainer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalPrediction,
    TrainOutput,
    is_wandb_available,
    set_seed,
)
if is_apex_available():
    from apex import amp

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


logger = logging.getLogger(__name__)


class PredictionOutput(NamedTuple):
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    other_outputs: Optional[Tuple[np.ndarray]] = None

class _SimpleDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.length = len(self.data_list)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        return self.data_list[index]

class Trainer(T_Trainer):
    def __init__(self, 
            model: Model, args: TrainingArguments,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            prediction_loss_only=False, 
            tb_writer: Optional["SummaryWriter"] = None, 
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None, 
            no_use_scheduler = False, 
            # 自定义get_optimizer函数,不使用AdamW, 注意是一个函数，调用时的实际参数是self.model和self.args
            get_optimizer: Callable[[Model, TrainingArguments], torch.optim.Optimizer] = None, 
            # 每个batch的模型其他位置的输出（除去loss和logits这两个位置），如何组合到一起的问题，
            # 因为不同batch可能这个输出的Tensor的维度不一，不想loss，和logits的维度除了batch不一样，其他维度都是确定的，即都是1
            # 注意这个方法目前是应用到所有的位置，是reduce方法，即增量计算
            # 如果没有实现这个方法，则模型不会在Prediction时输出这些额外的输出
            # 注意这个方法的输出不会再进行任何数据格式转换，所以要detach/cpu/numpy,请在这里做好!!!
            # 因为 evaluate方法也会调用prediction_loop, 从而调用reduce_other_outputs,所以可能边训练边evaluate会有点卡
            # 如果仅仅需要在predict的时候使用other_outputs，可以考虑在predict方法调用时单独传入，而不是在这里一开始就传好参数。
            reduce_other_outputs: Callable[[Tuple[torch.tensor]], Any] = None, 
            output_the_same_dir: bool = True,  # 是否将模型也保存到log目录，方便随着日志一起删除
            compare_datasets: bool = False, # 是否开启不同数据集(train/eval)之间的对比，体现在tensorboard上。
        ):
        super().__init__(model, args, data_collator=data_collator, train_dataset=train_dataset,
            eval_dataset=eval_dataset, compute_metrics=compute_metrics, prediction_loss_only=prediction_loss_only, tb_writer=tb_writer, optimizers=optimizers)
        if output_the_same_dir:
            self.args = deepcopy(self.args)
            self.args.output_dir = self.args.logging_dir

        logger.info(self.args.to_json_string())
        
        self.exp_dir = ""
        
        self.auto_tb_writer = False
        if self.tb_writer is not None:
            self.auto_tb_writer = True
            self.tb_writer.close()
            log_dirs = os.listdir(self.args.logging_dir)
            for file in log_dirs:
                if file.startswith('events.out.tfevents'):
                    os.remove(os.path.join(self.args.logging_dir, file))
            time_str = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            if any(name.startswith("Exp-") for name in log_dirs):
                sorted_log_dirs = sorted([name for name in log_dirs if name.startswith("Exp-")]) # Exp-0001
                next_iter = int(sorted_log_dirs[-1][4:8])+1
            else:
                next_iter = 1
            next_iter_str = "%04d" % next_iter
            self.exp_dir = "Exp-"+next_iter_str+"-"+time_str
            log_dir = os.path.join(self.args.logging_dir, self.exp_dir)
            self.log_dir = log_dir
            self.compare_datasets = compare_datasets
            self.call_once_called = False
            
        self.eval_dataloader = None
        if self.args.evaluate_during_training:
            self.eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.eval_train_dataloader = None
        if self.args.evaluate_trainset_during_training:
            self.eval_train_dataloader = self.get_eval_dataloader(self.get_sample_trainset(train_dataset))
        self.no_use_scheduler = no_use_scheduler # 设置成True表示学习率不衰减，乘积因子始终为1(常数)的LambaLR 
        self.get_optimizer = get_optimizer
        self.reduce_other_outputs = reduce_other_outputs
    
    def call_once(self,):
        self.tb_writer = SummaryWriter(log_dir=self.log_dir)
        with open(os.path.join(self.log_dir, "training_args.json"), 'w') as f:
            f.write(self.args.to_json_string())
        model_save_path ="model save in "+str(os.path.join(self.args.output_dir, self.exp_dir))
        logger.info(model_save_path)
        log_save_path = "log save in "+str(self.log_dir)
        logger.info(log_save_path)
        if self.compare_datasets:
            self.tb_writer_for_train = SummaryWriter(log_dir=self.log_dir+"/train_set")
            self.tb_writer_for_eval = SummaryWriter(log_dir=self.log_dir+"/eval_set")

    def get_sample_trainset(self, train_dataset: Dataset) -> Dataset:
        if self.args.trainset_sampling > 0:
            total_num = len(train_dataset)
            if total_num <= self.args.trainset_sampling:
                return copy.deepcopy(train_dataset)
            indices = np.random.choice(range(0,total_num), self.args.trainset_sampling, replace=False)
            return _SimpleDataset([train_dataset[i] for i in indices])
        return copy.deepcopy(train_dataset)
    
    @overrides
    def log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None, description: str = "train") -> None:
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.global_step is None:
            # when logging evaluation metrics without training
            self.global_step = 0
        if self.tb_writer:
            for k, v in logs.items():
                k = k if k[0] != "_" else k[1:]
                if isinstance(v, (int, float)):
                    if k.startswith("eval_train"):
                        self.tb_writer.add_scalar("eval_trainset/"+k, v, self.global_step)
                    elif k.startswith("eval"):
                        self.tb_writer.add_scalar("eval_devset/"+k, v, self.global_step)
                    else:
                        self.tb_writer.add_scalar("training_state/"+k, v, self.global_step)
                    if self.compare_datasets:
                        if k.startswith("eval_train"):
                            self.tb_writer_for_train.add_scalar("compare_datasets/"+k[11:], v, self.global_step)
                        elif k.startswith("eval"):
                            self.tb_writer_for_eval.add_scalar("compare_datasets/"+k[5:], v, self.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        '"%s" of type %s for key "%s" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute.",
                        v,
                        type(v),
                        k,
                    )
            self.tb_writer.flush()
        #if is_wandb_available():
        #    if self.is_world_master():
        #        wandb.log(logs, step=self.global_step)
        output = {**logs, **{"step": self.global_step}}
        if iterator is not None:
            iterator.write(output)
        else:
            logger.info(output)
        
        file_name = description + '.log.txt'
        with open(os.path.join(self.log_dir, file_name),'a') as f:
            f.write(str(logs)+'\n')
    
    @overrides
    def get_optimizers(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers
        if self.get_optimizer is not None:
            optimizer = self.get_optimizer(self.model, self.args)
        else:
            # Prepare optimizer and schedule (linear warmup and decay)
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        
        
        if self.no_use_scheduler:
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )
        return optimizer, scheduler

    @overrides
    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
        """
        if not self.call_once_called and self.auto_tb_writer:
            self.call_once()
            self.call_once_called = True
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)
        
        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )
        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_master())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None
                
            tqdm_dict = {}

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                tr_loss += self.training_step(model, inputs, optimizer)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_torch_tpu_available():
                        xm.optimizer_step(optimizer)
                    else:
                        optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss = tr_loss
                        
                        tqdm_dict['train_loss'] = logs["loss"]

                        self.log(logs)

                    if self.args.evaluate_during_training and self.global_step % self.args.eval_steps == 0:
                        eval_metrics = self.evaluate(use_tqdm=False, log=True, description="eval") # also save the metrics
                        for k,v in eval_metrics.items():
                            if k[0] == '_':
                                tqdm_dict[k[1:]] = v
                    
                    if self.args.evaluate_trainset_during_training and self.global_step % self.args.eval_trainset_steps == 0:
                        train_metrics = self.evaluate(eval_dataloader=self.eval_train_dataloader,use_tqdm=False, log=True, description="eval_train")
                        for k,v in train_metrics.items():
                            if k[0] == '_':
                                tqdm_dict[k[1:]] = v

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert model.module is self.model
                        else:
                            assert model is self.model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, self.exp_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

                        self.save_model(output_dir)

                        if self.is_world_master():
                            self._rotate_checkpoints()

                        if is_torch_tpu_available():
                            xm.rendezvous("saving_optimizer_states")
                            xm.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            xm.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        elif self.is_world_master():
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                epoch_iterator.set_postfix(**tqdm_dict)
                train_iterator.set_postfix(**tqdm_dict)
                
                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
                
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.tb_writer:
            self.tb_writer.close()
            log_dirs = os.listdir(self.log_dir)
            for name in log_dirs:
                if re.match("\d{10}.*", name): # unknown dir generated by tensorboardX
                    shutil.rmtree(os.path.join(self.log_dir, name))
        if self.compare_datasets:
            self.tb_writer_for_train.close()
            self.tb_writer_for_eval.close()
            
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step)
    
    @overrides
    def evaluate(self, eval_dataloader: Optional[DataLoader] = False, eval_dataset: Optional[Dataset] = None, use_tqdm: Optional[bool] = True,  log:bool=False, description:str="eval") -> Dict[str, float]:
        if not self.call_once_called and self.auto_tb_writer:
            self.call_once()
            self.call_once_called = True
        eval_dataloader = eval_dataloader or self.eval_dataloader or self.get_eval_dataloader(eval_dataset,)
        output = self.prediction_loop(eval_dataloader, description=description, use_tqdm=use_tqdm)
        if log:
            self.log(output.metrics, description=description)
        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())
        return output.metrics
    
    @overrides
    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        with open(os.path.join(output_dir, "training_args.json"), 'w') as f:
            f.write(self.args.to_json_string())
    
    # 注意 prediction_loop也被evaluate函数调用，
    @overrides # 增加一个参数，用来控制训练evaluation的进度条
    def prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None, use_tqdm: Optional[bool] = True,
        reduce_other_outputs:Callable[[Tuple[torch.tensor]], Any] = None, 
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if hasattr(self, "_prediction_loop"):
            warnings.warn(
                "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                FutureWarning,
            )
            return self._prediction_loop(dataloader, description, prediction_loss_only=prediction_loss_only)

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        other_outputs: Tuple[torch.Tensor] = None
        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        data_iterators = tqdm(dataloader, desc=description) if use_tqdm else dataloader
        reduce_other_outputs = reduce_other_outputs if reduce_other_outputs is not None else self.reduce_other_outputs

        for inputs in data_iterators:
            loss, logits, labels, other_outputs_ = self.prediction_step(model, inputs, prediction_loss_only)
            if loss is not None:
                eval_losses.append(loss)
            if logits is not None:
                preds = logits if preds is None else torch.cat((preds, logits), dim=0)
            if labels is not None:
                label_ids = labels if label_ids is None else torch.cat((label_ids, labels), dim=0)
            if other_outputs_ is not None:
                #print(list(o.size() for o in other_outputs_))
                #if other_outputs is not None:
                #    print(list(o.size() for o in others))
                if reduce_other_outputs is not None:
                    other_outputs = other_outputs_ if other_outputs is None else tuple(
                        reduce_other_outputs(output, output_) for output, output_ in zip(other_outputs, other_outputs_)
                        ) # 不加tuple只用()就是generator

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
            #if other_outputs is not None: # [TODO] maybe error!!! 不熟悉distributed训练
            #    other_outputs = tuple(self.distributed_concat(o, num_total_examples=self.num_examples(dataloader)) for o in other_outputs)
        elif is_torch_tpu_available():
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            if preds is not None:
                preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
            if label_ids is not None:
                label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)
            #if others is not None: # [TODO] maybe error!!! 不熟悉tpu训练, 这里就不考虑了
            #others = tuple(xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)
        
        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()
        if other_outputs is not None: 
            other_outputs = other_outputs # 假设一切都在self.reduce_other_outputs中处理好了
        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics[f"{description}_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        #for key in list(metrics.keys()):
        #    if not key.startswith("eval_"):
        #        metrics[f"eval_{key}"] = metrics.pop(key)
        for key in list(metrics.keys()):
            if not key.startswith(description):
                tqdm_prefix = ""
                new_key = key
                if key[0] == "_":
                    tqdm_prefix = "_"
                    new_key = key[1:]
                
                metrics[tqdm_prefix+description+"_"+new_key] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics, other_outputs=other_outputs)

    @overrides # 增加了返回值, 模型不但可以返回logits和loss, 也可以返回其他值，模型forward的logits位置输出
    def prediction_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

        inputs = self._prepare_inputs(inputs, model)

        with torch.no_grad():
            outputs = model(**inputs)
            if has_labels:
                loss, logits = outputs[:2]
                loss = loss.mean().item()
                other_outputs = outputs[2:] # a tuple for all the rest position's outputs
            else:
                loss = None
                logits = outputs[0]
                other_outputs = outputs[1:] # a tuple for all the rest position's outputs
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None, None)

        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.detach()
        return (loss, logits.detach(), labels, other_outputs)

    @overrides 
    def predict(self, test_dataset: Dataset=None, test_dataloader:DataLoader=None, reduce_other_outputs:Callable[[Tuple[torch.tensor]], Any] = None) -> PredictionOutput:
        # 与evaluate的区别：
        #   1. predict返回值更多，evaluate只返回metrics;
        #   2. evaluate会进行log保存，而这个predict不会保存; 
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on.
        Returns:
            `NamedTuple`:
            predictions (:obj:`np.ndarray`):
                The predictions on :obj:`test_dataset`.
            label_ids (:obj:`np.ndarray`, `optional`):
                The labels (if the dataset contained some).
            metrics (:obj:`Dict[str, float]`, `optional`):
                The potential dictionary of metrics (if the dataset contained labels).
        """
        if test_dataloader is None:
            if test_dataset is None:
                logger.error("test_dataset or test_dataloader must be given one.")
                return 0
            test_dataloader = self.get_test_dataloader(test_dataset)

        return self.prediction_loop(test_dataloader, description="Prediction", reduce_other_outputs=reduce_other_outputs)
