import os
import sys
import re
import time
import copy
import logging

import json

import torch

import copy

import numpy as np
from mlcore import config


from enum import Enum


try:
    from apex import amp
except:
    amp = None


class TrainEvent(Enum):
    EPOCH_BEGIN = 1
    EPOCH_END = 2
    TRAIN_BATCH_BEGIN = 3
    TRAIN_BATCH_END = 4
    TEST_BATCH_BEGIN = 5
    TEST_BATCH_END = 6
    TRAIN_BEGIN = 7
    TRAIN_END = 8
    TEST_BEGIN = 9
    TEST_END = 10


class Trainer:
    def __init__(
        self,
        model,
        callbacks=[],
        model_args="image",
        optimizer=None,
        loss=None,
        lr_scheduler=None,
        path=None,
        use_apex=None,
    ):
        super(Trainer, self).__init__()
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._lr_scheduler = lr_scheduler

        self._path = path
        self._step = 0
        self._epoch = 0
        self._model_args = model_args

        self._callbacks = callbacks

        self._use_apex = use_apex

        if self._use_apex:
            if isinstance(self._model, torch.nn.DataParallel):
                self._model.module, self._optimizer = amp.initialize(
                    self._model.module, self._optimizer, opt_level="O1"
                )
            else:
                self._model, self._optimizer = amp.initialize(self._model, self._optimizer, opt_level="O1")

        for callback in self._callbacks:
            callback.set_trainer(self)

    def call_callbacks(self, event, **kwargs):
        data = kwargs
        for callback in self._callbacks:
            if event == TrainEvent.EPOCH_BEGIN:
                result = callback.on_epoch_begin(**data)
            if event == TrainEvent.EPOCH_END:
                result = callback.on_epoch_end(**data)
            if event == TrainEvent.TRAIN_BATCH_BEGIN:
                result = callback.on_train_batch_begin(**data)
            if event == TrainEvent.TRAIN_BATCH_END:
                result = callback.on_train_batch_end(**data)
            if event == TrainEvent.TEST_BATCH_BEGIN:
                result = callback.on_test_batch_begin(**data)
            if event == TrainEvent.TEST_BATCH_END:
                result = callback.on_test_batch_end(**data)
            if event == TrainEvent.TRAIN_BEGIN:
                result = callback.on_train_begin(**data)
            if event == TrainEvent.TRAIN_END:
                result = callback.on_train_end(**data)
            if event == TrainEvent.TEST_BEGIN:
                result = callback.on_test_begin(**data)
            if event == TrainEvent.TEST_END:
                result = callback.on_test_end(**data)

            if isinstance(result, dict):
                data.update(result)

    def get_model_args(self, sample, model_args, device):

        if callable(model_args):
            model_input = model_args(sample)
        if isinstance(model_args, str):
            model_input = sample[model_args].to(device)
        if isinstance(model_args, (list, set)):
            model_input = [self.get_model_args(sample, x, device) for x in model_args]
        return model_input

    def train_step(self, sample, device="cpu"):
        result = {}

        if "loss_weight" in sample:
            result.update({"loss_weight": torch.mean(sample["loss_weight"])})

        self._optimizer.zero_grad()

        model_input = self.get_model_args(sample, self._model_args, device)

        with torch.set_grad_enabled(True):
            # print(f'A: {image.shape} {source.shape}')
            model_output = self._model(model_input)
            loss = self._loss(model_output, sample)
            # softmax_op = torch.nn.Softmax2d()
            # result.update({'prediction': softmax_op(logits)})

            result.update({"batch_size": model_input.shape[0]})
            result.update({"loss": loss})
            result.update({"model_output": model_output})

            if self._use_apex:
                with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self._optimizer.step()

            if self._lr_scheduler:
                self._lr_scheduler.step()

                result.update({"lr": self._optimizer.param_groups[0]["lr"]})

        return {**sample, **result}

    def val_step(self, sample, device="cpu"):
        result = {}

        model_input = self.get_model_args(sample, self._model_args, device)

        with torch.set_grad_enabled(False):
            # print(f'A: {image.shape} {source.shape}')
            model_output = self._model(model_input)
            loss = self._loss(model_output, sample)

            result.update({"batch_size": model_input.shape[0]})
            result.update({"loss": loss})
            result.update({"model_output": model_output})

        return {**sample, **result}

    def infer_step(self, sample, device="cpu"):
        result = {}

        self._model.eval()

        model_input = self.get_model_args(sample, self._model_args, device)

        with torch.set_grad_enabled(False):
            # print(f'A: {image.shape} {source.shape}')
            model_output = self._model(model_input)

            result.update({"batch_size": model_input.shape[0]})
            result.update({"model_output": model_output})

        return {**sample, **result}

    def train_iter(
        self,
        train_dataloader,
        val_dataloader=None,
        num_epochs=None,
        num_steps=None,
        max_epochs=None,
        max_steps=None,
        device="cpu",
        logging_step=None,
        summary_step=None,
        save_epoch=1,
    ):

        init_step = self._step
        init_epoch = self._epoch

        self.call_callbacks(TrainEvent.TRAIN_BEGIN)
        self.call_callbacks(TrainEvent.EPOCH_BEGIN)
        dataloader_iter = iter(train_dataloader)

        training_done = False
        while not training_done:
            # Train
            sum_loss = 0.0
            count = 0
            result = {}
            while True:
                # print(f"{self._step} {self._epoch}")
                try:
                    sample = next(dataloader_iter)
                except StopIteration:

                    self._epoch += 1
                    if num_epochs:
                        if self._epoch > 0 and self._epoch % num_epochs == 0:
                            break

                    if max_epochs:
                        if self._epoch > 0 and max_epochs < self._epoch:
                            print(self._epoch)
                            print("WTFFF")
                            training_done = True
                            break

                    self.call_callbacks(TrainEvent.EPOCH_END)
                    self.call_callbacks(TrainEvent.EPOCH_BEGIN)
                    # Not sure if I should introduce another callback

                    logging.info(f"Trainer: reset train_dataloader")
                    dataloader_iter = iter(train_dataloader)
                    continue

                # print(f'B {self._step}')
                if max_steps:
                    if self._step > max_steps:
                        print("WTF")
                        training_done = True
                        break

                # Force a break after num_steps
                if num_steps:
                    if self._step > 0 and self._step % num_steps == 0:

                        self._step += 1
                        print("WTF2")
                        break

                # print(f'C {self._step}')

                self.call_callbacks(TrainEvent.TRAIN_BATCH_BEGIN, sample=sample)
                # print(f'TrainStep: {self._step}')
                result = self.train_step(sample, device=device)
                self._step += 1
                self.call_callbacks(TrainEvent.TRAIN_BATCH_END, **result)

            self._epoch += 1
            if "loss" in result and "batch_size" in result:
                sum_loss += result["loss"] * result["batch_size"]
                count += result["batch_size"]
            result.update({"loss": sum_loss / (count + 1e-10)})
            self.call_callbacks(TrainEvent.TRAIN_END, **result)

            # Save everything
            if save_epoch and self._epoch % save_epoch == 0:
                self.save()

            # Eval
            if val_dataloader is not None:
                self.call_callbacks(TrainEvent.TEST_BEGIN)
                sum_loss = 0.0
                count = 0
                result = {}
                for i, sample in enumerate(val_dataloader):
                    # print(i)
                    self.call_callbacks(TrainEvent.TEST_BATCH_BEGIN, sample=sample)
                    result = self.val_step(sample, device=device)
                    self.call_callbacks(TrainEvent.TEST_BATCH_END, **result)
                    if "loss" in result and "batch_size" in result:
                        sum_loss += result["loss"] * result["batch_size"]
                        count += result["batch_size"]
                result.update({"loss": sum_loss / count})

                self.call_callbacks(TrainEvent.TEST_END, **result)

            self.call_callbacks(TrainEvent.EPOCH_END)

    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        num_epochs=None,
        num_steps=None,
        max_epochs=None,
        max_steps=None,
        device="cpu",
        logging_step=None,
        summary_step=None,
    ):
        if num_epochs is None and num_steps is None:
            num_epochs = 1

        self.train_iter(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=num_epochs,
            num_steps=num_steps,
            max_epochs=max_epochs,
            max_steps=max_steps,
            device=device,
            logging_step=logging_step,
            summary_step=summary_step,
        )

    def infer(self, eval_dataloader, device="cpu"):
        for i, sample in enumerate(eval_dataloader):
            yield self.infer_step(sample, device=device)

    def state_dict(self):

        state_dict = {
            "optimizer": self._optimizer.state_dict(),
            "step": self._step,
            "epoch": self._epoch,
        }

        try:
            state_dict.update({"model": self._model.module.state_dict()})
            logging.info("Trainer: Model use DataParallel wrapper")
        except AttributeError:
            state_dict.update({"model": self._model.state_dict()})
            logging.info("Trainer: Model use DataParallel wrapper")

        if self._lr_scheduler is not None:
            state_dict.update({"lr_scheduler": self._lr_scheduler.state_dict()})

        callbacks = {}
        callbacks_index = {}
        for callback in self._callbacks:
            callback_data = callback.state_dict()
            callback_name = type(callback).__name__
            if callback_name not in callbacks_index:
                callbacks_index[callback_name] = 0
            callbacks[f"{callback_name}_{callbacks_index[callback_name]}"] = callback_data
            callbacks_index[callback_name] += 1

        state_dict["callbacks"] = callbacks

        if self._use_apex and amp is not None:
            state_dict.update({"amp": amp.state_dict()})

        return state_dict

    def load_state_dict(self, data: dict):
        # print(data.keys())
        # print(data["model"].keys())
        # print("###############")
        # print(self._model.state_dict().keys())
        # exit()
        try:
            self._model.load_state_dict(data["model"])
        except RuntimeError:
            logging.warn("Trainer: Save DataParallel model without using module")
            map_dict = {}
            for key, value in data["model"].items():
                if key.split(".")[0] == "module":
                    map_dict[".".join(key.split(".")[1:])] = value
                else:
                    map_dict["module." + key] = value
            self._model.load_state_dict(map_dict)
        if self._optimizer is not None and "optimizer" in data:
            self._optimizer.load_state_dict(data["optimizer"])

        self._step = data["step"]
        if "epoch" in data:
            self._epoch = data["epoch"]

        if self._lr_scheduler is not None and "lr_scheduler" in data:
            self._lr_scheduler.load_state_dict(data["lr_scheduler"])

        if "callbacks" in data:
            callbacks_index = {}
            # callback_re = re.compile(r'^(.*?)')

            for callback in self._callbacks:

                callback_name = type(callback).__name__
                if callback_name not in callbacks_index:
                    callbacks_index[callback_name] = 0

                if f"{callback_name}_{callbacks_index[callback_name]}" not in data["callbacks"]:
                    continue

                callback.load_state_dict(data["callbacks"][f"{callback_name}_{callbacks_index[callback_name]}"])
                callbacks_index[callback_name] += 1

        if self._use_apex and amp is not None:
            if "amp" in data:
                amp.load_state_dict(data["amp"])

    def save(self, path=None, step=None):
        if path is None:
            path = self._path

        if step is None:
            step = self._step

        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.state_dict(), os.path.join(path, f"model_{step}.pt"))

    def load(self, abspath=None, path=None, step=None, resume=None, device="cpu"):
        if abspath is not None:
            state_dict = torch.load(abspath, map_location=device)
            self.load_state_dict(state_dict)
            return

        if path is None:
            path = self._path

        if step is None:
            step = self._step

        if resume:
            steps = self.checkpoints(path=path)
            if len(steps) == 0:
                raise FileNotFoundError

            step = steps[-1]

        state_dict = torch.load(os.path.join(path, f"model_{step}.pt"), map_location=device)
        self.load_state_dict(state_dict)

    def checkpoints(self, path=None):
        if path is None:
            path = self._path
        if not os.path.exists(path):
            return []
        return sorted(
            [int(re.match(r"model_(\d+)\.pt", x)[1]) for x in os.listdir(path) if re.match(r"model_(\d+)\.pt", x)]
        )


class MeanTeacherTrainer(Trainer):
    def __init__(
        self,
        *args,
        teacher_model_args="teacher_image",
        loss_weight_args="loss_weight",
        consistency_loss=None,
        consistency_weight=1000.0,
        logit_distance_loss=None,
        logit_distance_weight=None,
        alpha=0.999,
        **kwargs,
    ):
        self._consistency_loss = consistency_loss
        self._teacher_model = copy.deepcopy(kwargs["model"])
        self._consistency_weight = consistency_weight
        self._teacher_model_args = teacher_model_args
        self._alpha = alpha

        self._logit_distance_loss = logit_distance_loss
        self._logit_distance_weight = logit_distance_weight
        self._mean_teacher_step = 0

        # TODO what?
        for param in self._teacher_model.parameters():
            param.detach_()

        super(MeanTeacherTrainer, self).__init__(*args, **kwargs)

    def update_teacher_variables(self):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self._mean_teacher_step + 1), self._alpha)
        for ema_param, param in zip(self._teacher_model.parameters(), self._model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.float().data)
        self._mean_teacher_step += 1

    def train_step(self, sample, device="cpu"):
        self._teacher_model.to(device)
        result = {}

        if "loss_weight" in sample:
            result.update({"loss_weight": torch.mean(sample["loss_weight"])})

        self._optimizer.zero_grad()

        model_input = self.get_model_args(sample, self._model_args, device)
        teacher_model_input = self.get_model_args(sample, self._teacher_model_args, device)

        with torch.set_grad_enabled(False):
            # print(f'A: {image.shape} {source.shape}')
            teacher_model_output = self._teacher_model(teacher_model_input)
            if self._logit_distance_loss is not None:
                assert isinstance(teacher_model_output, (list, set)), ""
                assert len(teacher_model_output) == 2, ""

                teacher_model_output, teacher_model_output_2 = teacher_model_output
            else:
                teacher_model_output_2 = teacher_model_output

        with torch.set_grad_enabled(True):
            # print(f'A: {image.shape} {source.shape}')
            model_output = self._model(model_input)
            if self._logit_distance_loss is not None:
                assert isinstance(model_output, (list, set)), ""
                assert len(model_output) == 2, ""
                model_output, model_output_2 = model_output
                logit_distance_loss = self._logit_distance_loss(model_output, model_output_2)
            else:
                model_output_2 = model_output
                logit_distance_loss = 0

            if callable(self._logit_distance_weight):
                logit_distance_weight = self._logit_distance_weight(self._step)
            else:
                logit_distance_weight = self._logit_distance_weight

            student_loss = self._loss(model_output, sample)

            consistency_loss = self._consistency_loss(model_output_2, teacher_model_output)

            if callable(self._consistency_weight):
                consistency_weight = self._consistency_weight(self._step)
            else:
                consistency_weight = self._consistency_weight

            loss = consistency_weight * consistency_loss + student_loss + logit_distance_weight * logit_distance_loss

            result.update({"batch_size": model_input.shape[0]})
            result.update({"loss": loss})

            result.update({"student_loss": student_loss})
            result.update({"model_output": model_output})

            result.update({"teacher_loss": loss})
            result.update({"teacher_model_output": model_output})

            result.update({"consistency_loss": consistency_loss})
            result.update({"consistency_weight": consistency_weight})

            result.update({"logit_distance_loss": logit_distance_loss})
            result.update({"logit_distance_weight": logit_distance_weight})

            if self._use_apex:
                with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self._optimizer.step()

            if self._lr_scheduler:
                self._lr_scheduler.step()
                result.update({"lr": self._optimizer.param_groups[0]["lr"]})  # Test this

        self.update_teacher_variables()

        return {**sample, **result}

    def val_step(self, sample, device="cpu"):
        self._teacher_model.to(device)
        result = {}

        model_input = self.get_model_args(sample, self._model_args, device)

        # with torch.set_grad_enabled(False):
        #     teacher_model_output = self._teacher_model(model_input)
        #     if self._logit_distance_loss is not None:
        #         assert isinstance(teacher_model_output, (list, set)), ""
        #         assert len(teacher_model_output) == 2, ""

        #         model_output, model_output_2 = teacher_model_output
        #     else:
        #         model_output = teacher_model_output

        #     loss = self._loss(model_output, sample)

        #     result.update({"batch_size": model_input.shape[0]})
        #     result.update({"loss": loss})
        #     result.update({"model_output": model_output})

        with torch.set_grad_enabled(False):
            # print(f'A: {image.shape} {source.shape}')
            teacher_model_output = self._teacher_model(model_input)
            if self._logit_distance_loss is not None:
                assert isinstance(teacher_model_output, (list, set)), ""
                assert len(teacher_model_output) == 2, ""

                teacher_model_output, teacher_model_output_2 = teacher_model_output
            else:
                teacher_model_output_2 = teacher_model_output

        with torch.set_grad_enabled(False):
            # print(f'A: {image.shape} {source.shape}')
            model_output = self._model(model_input)
            if self._logit_distance_loss is not None:
                assert isinstance(model_output, (list, set)), ""
                assert len(model_output) == 2, ""
                model_output, model_output_2 = model_output
                logit_distance_loss = self._logit_distance_loss(model_output, model_output_2)
            else:
                model_output_2 = model_output
                logit_distance_loss = 0

            if callable(self._logit_distance_weight):
                logit_distance_weight = self._logit_distance_weight(self._step)
            else:
                logit_distance_weight = self._logit_distance_weight

            student_loss = self._loss(model_output, sample)

            consistency_loss = self._consistency_loss(model_output_2, teacher_model_output)

            if callable(self._consistency_weight):
                consistency_weight = self._consistency_weight(self._step)
            else:
                consistency_weight = self._consistency_weight

            loss = consistency_weight * consistency_loss + student_loss + logit_distance_weight * logit_distance_loss

            result.update({"batch_size": model_input.shape[0]})
            result.update({"loss": self._loss(teacher_model_output, sample)})

            result.update({"student_loss": student_loss})
            result.update({"model_output": teacher_model_output})

            result.update({"teacher_loss": loss})
            result.update({"teacher_model_output": teacher_model_output})

            result.update({"consistency_loss": consistency_loss})
            result.update({"consistency_weight": consistency_weight})

            result.update({"logit_distance_loss": logit_distance_loss})
            result.update({"logit_distance_weight": logit_distance_weight})

        return {**sample, **result}

    def infer_step(self, sample, device="cpu"):

        self._teacher_model.to(device)
        result = {}

        self._teacher_model.eval()

        model_input = self.get_model_args(sample, self._model_args, device)
        with torch.set_grad_enabled(False):  # print(f'A: {image.shape} {source.shape}')
            teacher_model_output = self._teacher_model(model_input)
            if self._logit_distance_loss is not None:
                assert isinstance(teacher_model_output, (list, set)), ""
                assert len(teacher_model_output) == 2, ""

                teacher_model_output, teacher_model_output_2 = teacher_model_output
            else:
                teacher_model_output_2 = teacher_model_output

            result.update({"batch_size": model_input.shape[0]})
            result.update({"model_output": teacher_model_output})

        return {**sample, **result}

    def state_dict(self):
        state = super(MeanTeacherTrainer, self).state_dict()
        state.update({"ema_model": self._teacher_model.state_dict()})
        state.update({"mean_teacher_step": self._mean_teacher_step})
        return state

    def load_state_dict(self, data: dict):

        super(MeanTeacherTrainer, self).load_state_dict(data)

        try:
            self._teacher_model.load_state_dict(data["ema_model"])
        except RuntimeError:
            logging.warn("Trainer: Save DataParallel model without using module")
            map_dict = {}
            for key, value in data["ema_model"].items():
                if key.split(".")[0] == "module":
                    map_dict[".".join(key.split(".")[1:])] = value
                else:
                    map_dict["module." + key] = value
            self._teacher_model.load_state_dict(map_dict)
        except AttributeError:
            logging.warn("Trainer: EMA model stored without state_dict")
            try:
                self._teacher_model.load_state_dict(data["ema_model"].state_dict())
            except RuntimeError:
                logging.warn("Trainer: Save DataParallel model without using module")
                map_dict = {}
                for key, value in data["ema_model"].state_dict().items():
                    if key.split(".")[0] == "module":
                        map_dict[".".join(key.split(".")[1:])] = value
                    else:
                        map_dict["module." + key] = value
                self._teacher_model.load_state_dict(map_dict)

        self._mean_teacher_step = data["mean_teacher_step"]


class FixMatchTrainer(Trainer):
    def __init__(
        self,
        *args,
        consistency_loss=None,
        consistency_weight=1,
        strong_augmented_model_args="teacher_image",
        loss_weight_args="loss_weight",
        alpha=0.999,
        threshold=0.7,
        **kwargs,
    ):
        self._consistency_loss = consistency_loss
        self._average_model = copy.deepcopy(kwargs["model"])
        self._consistency_weight = consistency_weight
        self._strong_augmented_model_args = strong_augmented_model_args
        self._alpha = alpha

        # self._logit_distance_loss = logit_distance_loss
        # self._logit_distance_weight = logit_distance_weight

        # TODO what?
        for param in self._average_model.parameters():
            param.detach_()

        super(FixMatchTrainer, self).__init__(*args, **kwargs)

    def update_average_variables(self):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self._step + 1), self._alpha)
        for ema_param, param in zip(self._average_model.parameters(), self._model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.float().data)

    def train_step(self, sample, device="cpu"):
        self._average_model.to(device)
        result = {}

        if "loss_weight" in sample:
            result.update({"loss_weight": torch.mean(sample["loss_weight"])})

        self._optimizer.zero_grad()

        labeled_index = (sample["loss_weight"] == 1).to(device)
        unlabeled_index = (sample["loss_weight"] == 0).to(device)

        model_input = self.get_model_args(sample, self._model_args, device)
        strong_augmented_model_input = self.get_model_args(sample, self._strong_augmented_model_args, device)

        label = torch.index_select(sample["label"].to(device), 0, labeled_index.nonzero().flatten())

        unlabeled_weak = torch.index_select(model_input, 0, labeled_index.nonzero().flatten())
        labeled_weak = torch.index_select(model_input, 0, unlabeled_index.nonzero().flatten())

        unlabeled_strong = torch.index_select(strong_augmented_model_input, 0, labeled_index.nonzero().flatten())
        labeled_strong = torch.index_select(strong_augmented_model_input, 0, unlabeled_index.nonzero().flatten())

        with torch.set_grad_enabled(False):
            # print(f'A: {image.shape} {source.shape}')
            model_unlabeled_weak_output = self._model(unlabeled_weak)

        with torch.set_grad_enabled(True):
            # print(f'A: {image.shape} {source.shape}')
            model_labeled_weak_output = self._model(labeled_weak)
            model_unlabeled_strong_output = self._model(unlabeled_strong)

            labeled_loss = self._loss(model_labeled_weak_output, label)

            unlabeled_loss = self._consistency_loss(model_unlabeled_strong_output, model_unlabeled_weak_output)

            if callable(self._consistency_weight):
                consistency_weight = self._consistency_weight(self._step)
            else:
                consistency_weight = self._consistency_weight

            loss = consistency_weight * unlabeled_loss + labeled_loss

            result.update({"batch_size": model_input.shape[0]})
            result.update({"loss": loss})
            result.update({"labeled_loss": labeled_loss})
            # result.update({"assigment": torch.mean(assigment)})

            # result.update({"student_loss": student_loss})
            # result.update({"model_output": torch.cat([])})

            # result.update({"teacher_loss": loss})
            # result.update({"teacher_model_output": model_output})

            result.update({"unlabeled_loss": unlabeled_loss})
            result.update({"consistency_weight": consistency_weight})

            # result.update({"logit_distance_loss": logit_distance_loss})
            # result.update({"logit_distance_weight": logit_distance_weight})

            if self._use_apex:
                with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self._optimizer.step()

            if self._lr_scheduler:
                self._lr_scheduler.step()
                result.update({"lr": self._optimizer.param_groups[0]["lr"]})
        # Test this
        self.update_average_variables()

        return {**sample, **result}

    def val_step(self, sample, device="cpu"):
        self._average_model.to(device)
        result = {}

        model_input = self.get_model_args(sample, self._model_args, device)

        with torch.set_grad_enabled(False):
            model_output = self._average_model(model_input)

            loss = self._loss(model_output, sample["label"].to(device))

            result.update({"batch_size": model_input.shape[0]})
            result.update({"loss": loss})
            result.update({"model_output": model_output})

        return {**sample, **result}

    def infer_step(self, sample, device="cpu"):
        self._average_model.to(device)

        result = {}
        self._average_model.eval()

        model_input = self.get_model_args(sample, self._model_args, device)

        with torch.set_grad_enabled(False):
            model_output = self._average_model(model_input)

            result.update({"batch_size": model_input.shape[0]})
            result.update({"model_output": model_output})

        return {**sample, **result}

    def state_dict(self):
        state = super(FixMatchTrainer, self).state_dict()
        state.update({"ema_model": self._average_model.state_dict()})
        return state

    def load_state_dict(self, data: dict):
        super(FixMatchTrainer, self).load_state_dict(data)
        self._average_model = data["ema_model"]


from mlcore.self import simclr


class SimCLRTrainer(Trainer):
    def __init__(
        self,
        *args,
        supervised_loss=None,
        supervised_loss_weight=0.0,
        simclr_loss=simclr.NTXentLoss,
        first_augmented_model_args="image",
        second_augmented_model_args="teacher_image",
        loss_weight_args="loss_weight",
        temperature=0.5,
        use_cosine_similarity=True,
        **kwargs,
    ):
        super(SimCLRTrainer, self).__init__(*args, **kwargs)

        self._first_augmented_model_args = first_augmented_model_args
        self._second_augmented_model_args = second_augmented_model_args
        self._simclr_loss = simclr_loss(temperature=temperature, use_cosine_similarity=use_cosine_similarity)

        self._supervised_loss = supervised_loss
        self._supervised_loss_weight = supervised_loss_weight

    def train_step(self, sample, device="cpu"):
        self._model.train()

        result = {}

        if "loss_weight" in sample:
            result.update({"loss_weight": torch.mean(sample["loss_weight"])})

        self._optimizer.zero_grad()

        first_augmented_model_input = self.get_model_args(sample, self._first_augmented_model_args, device)
        second_augmented_model_input = self.get_model_args(sample, self._second_augmented_model_args, device)

        if "loss_weight" not in sample:

            labeled_index = torch.ones(first_augmented_model_input.shape[0]).to(device)
            unlabeled_index = torch.zeros(first_augmented_model_input.shape[0]).to(device)
        else:
            labeled_index = (sample["loss_weight"] == 1).to(device)
            unlabeled_index = (sample["loss_weight"] == 0).to(device)

        # label = torch.index_select(sample["label"].to(device), 0, labeled_index.nonzero().flatten())

        first_labeled = torch.index_select(first_augmented_model_input, 0, labeled_index.nonzero().flatten())
        first_unlabeled = torch.index_select(first_augmented_model_input, 0, unlabeled_index.nonzero().flatten())

        second_labeled = torch.index_select(second_augmented_model_input, 0, labeled_index.nonzero().flatten())
        second_unlabeled = torch.index_select(second_augmented_model_input, 0, unlabeled_index.nonzero().flatten())

        with torch.set_grad_enabled(True):
            # print(f'A: {image.shape} {source.shape}')
            model_output = self._model(
                torch.cat([first_labeled, first_unlabeled, second_labeled, second_unlabeled], dim=0)
            )
            if isinstance(model_output, dict):
                projection = model_output.get("projection")
            else:
                projection = getattr(model_output, "projection")

            begin_shape = 0
            first_labeled_projection = projection[begin_shape : first_labeled.shape[0]]
            begin_shape += first_labeled.shape[0]
            first_unlabeled_projection = projection[begin_shape : begin_shape + first_unlabeled.shape[0]]
            begin_shape += first_unlabeled.shape[0]
            second_labeled_projection = projection[begin_shape : begin_shape + second_labeled.shape[0]]
            begin_shape += second_labeled.shape[0]
            second_unlabeled_projection = projection[begin_shape : begin_shape + second_unlabeled.shape[0]]
            begin_shape += second_unlabeled.shape[0]

            self_supervised_loss = self._simclr_loss(
                torch.cat([first_labeled_projection, first_unlabeled_projection], dim=0),
                torch.cat([second_labeled_projection, second_unlabeled_projection], dim=0),
            )

            if self._supervised_loss is not None:
                supervised_loss = self._supervised_loss(
                    {k: v[: first_labeled.shape[0]] for k, v in model_output.items()}, sample
                )
            else:
                supervised_loss = 0.0

            loss = self_supervised_loss + self._supervised_loss_weight * supervised_loss

            result.update({"self_supervised_loss": self_supervised_loss})
            result.update({"supervised_loss": supervised_loss})
            result.update({"loss": loss})
            # result.update({"assigment": torch.mean(assigment)})

            # result.update({"student_loss": student_loss})
            # result.update({"model_output": torch.cat([])})

            # result.update({"teacher_loss": loss})
            # result.update({"teacher_model_output": model_output})

            # result.update({"logit_distance_loss": logit_distance_loss})
            # result.update({"logit_distance_weight": logit_distance_weight})

            if self._use_apex:
                with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self._optimizer.step()

            if self._lr_scheduler:
                self._lr_scheduler.step()
                result.update({"lr": self._optimizer.param_groups[0]["lr"]})

        return {**sample, **result}

    def val_step(self, sample, device="cpu"):
        self._model.eval()
        result = {}

        model_input = self.get_model_args(sample, self._model_args, device)
        with torch.set_grad_enabled(False):
            model_output = self._model(model_input)

            if self._supervised_loss is not None:
                supervised_loss = self._supervised_loss(model_output, sample)
            else:
                supervised_loss = 0.0

            result.update({"batch_size": model_input.shape[0]})
            result.update({"loss": supervised_loss})
            result.update({"model_output": model_output})

        return {**sample, **result}

    def infer_step(self, sample, device="cpu"):
        self._model.eval()

        result = {}

        model_input = self.get_model_args(sample, self._model_args, device)

        with torch.set_grad_enabled(False):
            model_output = self._average_model(model_input)

            result.update({"batch_size": model_input.shape[0]})
            result.update({"model_output": model_output})

        return {**sample, **result}
