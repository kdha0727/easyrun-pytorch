"""
One-time Pytorch Trainer Utility and Useful Functions:
    written by Dong Ha Kim, in Yonsei University.
"""

# Frameworks and Internal Modules

from collections import OrderedDict

import torch
import torch.nn.functional
import torch.utils.data


# Typing Variables

import typing
import types
from functools import partial
from os import PathLike
_function_union = typing.Union[
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    partial
]
_loss_function_type = typing.Union[
    torch.nn.Module,
    _function_union,
    str,
]
_data_type = typing.Optional[
    typing.Union[
        torch.utils.data.Dataset,
        torch.utils.data.DataLoader,
    ]
]
_optional_path_type = typing.Optional[
    typing.Union[
        PathLike,
        str
    ]
]
_t = typing.TypeVar('_t')
_v = typing.TypeVar('_v')


#
# Tool Functions
#

def runtime_info(device: typing.Union[torch.device, str] = None) -> str:
    """Runtime information utility."""
    import sys
    import platform
    return (
        f"<Runtime Information>\n"
        f"OS version: \t\t{platform.platform()}\n"
        f"Python version:\t\t{sys.version}\n"
        f"Torch version:\t\t{torch.__version__}\n"
        f"Torch device:\t\t{device or ('cuda' if torch.cuda.is_available() else 'cpu')}"
    )


def dataset_info(train: _data_type = None, val: _data_type = None, test: _data_type = None, *, loader=False) -> str:
    """Dataset information utility."""
    return (
        f"<{'DataLoader' if loader else 'Dataset'} Information>\n" +
        (f"Train {'Batch' if loader else 'Dataset'}: \t\t{len(train)}\n" if train else "") +
        (f"Validation {'Batch' if loader else 'Dataset'}: \t{len(val)}\n" if val else "") +
        (f"Test {'Batch' if loader else 'Dataset'}: \t\t{len(test)}" if test else "")
    )


#
# One-time Trainer Class
#

class Easyrun(object):
    """

    One-time Pytorch Trainer Utility:
        written by Dong Ha Kim, in Yonsei University.

    Available Parameters:
        :param model: (torch.nn.Module) model object to use.
        :param optimizer: (torch.optim.Optimizer) optimizer.
        :param criterion: (torch.nn.Module) loss function or model object. You can also provide string name.
        :param epoch: (int) total epochs.
        :param train_iter: train data loader, or train dataset.
        :param val_iter: validation data loader, or validation dataset.
        :param test_iter: test data loader, or test dataset.
        :param step_task: (function) task to be run in each epoch.
            no input, or (current_epoch, ), or (current_epoch, current_train_result_list)
            can be given as function input.
            functools.partial is recommended to implement this.
        :param snapshot_dir: (str) provide if you want to use parameter saving and loading.
            in this path name, model's weight parameter at best(least) loss will be temporarily saved.
        :param verbose: (bool) verbosity. with turning it on, you can view learning logs.
            default value is True.
        :param timer: (bool) provide with verbosity, if you want to use time-checking.
            default value is True.
        :param log_interval: (int) provide with verbosity, if you want to set your log interval.
            default value is 20.

    Available Methods:
        (): [call] repeat training and validating for all epochs, followed by testing.
        to(device): apply to(device) in model, criterion, and all tensors.
        train(): run training one time with train dataset.
            returns (train_loss, train_accuracy).
        evaluate(): run validating one time with validation dataset.
            returns (val_loss, val_accuracy).
        step(): run training, followed by validating one time.
            returns (train_loss, train_accuracy, val_loss, val_accuracy).
        run(): repeat training and validating for all epochs.
            returns train result list, which contains each epoch`s
            (train_loss, train_accuracy, val_loss, val_accuracy).
        test(): run testing one time with test dataset.
            returns (test_loss, test_accuracy).
        state_dict(): returns state dictionary of trainer class.
        load_state_dict(): loads state dictionary of trainer class.

    """

    #
    # Constructor
    #

    __initialized: bool = False

    def __init__(
            self,
            model: torch.nn.Module,
            criterion: _loss_function_type,
            optimizer: torch.optim.Optimizer,
            epoch: int,
            train_iter: _data_type = None,
            val_iter: _data_type = None,
            test_iter: _data_type = None,
            step_task: typing.Optional[_function_union] = None,
            snapshot_dir: _optional_path_type = None,
            verbose: typing.Optional[bool] = True,
            timer: typing.Optional[bool] = False,
            log_interval: typing.Optional[int] = 20,
    ) -> None:

        import inspect
        import math

        _dataset_type = (torch.utils.data.Dataset, torch.utils.data.DataLoader)
        assert isinstance(train_iter, _dataset_type), \
            "Invalid train_iter type: %s" % train_iter.__class__.__name__
        assert val_iter is None or isinstance(val_iter, _dataset_type), \
            "Invalid val_iter type: %s" % val_iter.__class__.__name__
        assert test_iter is None or isinstance(test_iter, _dataset_type), \
            "Invalid test_iter type: %s" % test_iter.__class__.__name__
        assert isinstance(model, torch.nn.Module), \
            "Invalid model type: %s" % model.__class__.__name__
        assert isinstance(optimizer, torch.optim.Optimizer), \
            "Invalid model type: %s" % optimizer.__class__.__name__
        assert isinstance(epoch, int) and epoch > 0, \
            "Epoch is expected to be positive int, got %s" % epoch
        assert isinstance(log_interval, int) and log_interval > 0, \
            "Log Interval is expected to be positive int, got %s" % log_interval
        if not callable(criterion):
            assert isinstance(criterion, str), \
                "Invalid criterion type: %s" % criterion.__class__.__name__
            assert (hasattr(torch.nn, criterion) or hasattr(torch.nn.functional, criterion)), \
                "Invalid criterion string: %s" % criterion
        if step_task:
            assert callable(step_task), \
                "Step Task function is expected to be callable, got %s" % step_task
            try:
                assert len(inspect.signature(step_task).parameters) in range(3), \
                    "Step Task function`s argument length should be 0, 1, or 2."
            except ValueError as exc:
                raise TypeError("Invalid Step Task function: %s" % step_task) from exc

        self.model: torch.nn.Module = model
        self.criterion: _loss_function_type = criterion if not isinstance(criterion, str) \
            else getattr(torch.nn.functional, criterion, getattr(torch.nn, criterion)())
        self.optimizer: torch.optim.Optimizer = optimizer
        self.total_epoch: int = epoch
        self.train_iter: _data_type = train_iter
        self.val_iter: _data_type = val_iter
        self.test_iter: typing.Optional[_data_type] = test_iter
        self.step_task: typing.Optional[_function_union] = step_task
        self.step_task_mode = len(inspect.signature(step_task).parameters) if step_task else None
        self.snapshot_dir: _optional_path_type = snapshot_dir
        self.verbose: bool = verbose
        self.use_timer: bool = timer
        self.log_interval: int = log_interval
        self.train_batch_size: int = train_iter.batch_size
        self.train_loader_length: int = len(train_iter)
        self.train_dataset_length: int = len(getattr(train_iter, 'dataset', train_iter))
        self.save_and_load: bool = bool(snapshot_dir is not None and val_iter is not None)

        self._closed: bool = False
        self._current_epoch: int = 0
        self._best_loss: float = math.inf
        self._time_start: typing.Optional[float] = None
        self._time_stop: typing.Optional[float] = None
        self._processing_fn: _optional_path_type = None
        self._current_run_result: typing.Optional[typing.List] = None

        self.__to_args: typing.Optional[tuple] = None
        self.__to_kwargs: typing.Optional[dict] = None

        self.__initialized = True

    #
    # De-constructor: executed in buffer-cleaning in python exit
    #

    def __del__(self) -> None:
        self._close()

    #
    # Context manager magic methods
    #

    def __enter__(self: _t) -> _t:
        self._open()
        return self

    def __exit__(self, exc_info, exc_class, exc_traceback) -> None:
        try:
            self._close()
        except Exception as exc:
            if (exc_info or exc_class or exc_traceback) is not None:
                pass  # executed in exception handling - just let python raise that exception
            else:
                raise exc

    #
    # Attribute magic methods
    #

    def __setattr__(self, key, value):
        if not key.startswith('_') and self.__initialized:
            raise AttributeError('Cannot set attributes after initialized.')
        object.__setattr__(self, key, value)

    def __delattr__(self, key):
        if not key.startswith('_') and self.__initialized:
            raise AttributeError('Cannot set attributes after initialized.')
        object.__delattr__(self, key)

    #
    # Call implement: run training, evaluating, followed by testing
    #

    def __call__(self: _t) -> _t:
        self.run()
        if self.test_iter is not None:
            self.test()
        return self

    #
    # Running Methods
    #

    def train(self) -> typing.Tuple[float, float]:

        result = self._train()
        self._current_epoch += 1
        return result

    def evaluate(self) -> typing.Tuple[float, float]:

        return self._evaluate(test=False)

    def test(self) -> typing.Tuple[float, float]:

        return self._evaluate(test=True)

    def step(self) -> typing.Tuple[float, float, typing.Optional[float], typing.Optional[float]]:

        self._log_step(self._current_epoch + 1)

        train_loss, train_accuracy = self._train()

        if self.val_iter:
            test_loss, test_accuracy = self._evaluate(test=False)

            # Save the model having the smallest validation loss
            if test_loss < self._best_loss:
                self._best_loss = test_loss
                self._save()

        else:
            test_loss = test_accuracy = None

        self._current_epoch += 1

        self._do_step_task()

        return train_loss, train_accuracy, test_loss, test_accuracy

    def run(self) -> typing.List[typing.Tuple[float, float, typing.Optional[float], typing.Optional[float]]]:

        try:
            self._open()
            self._current_run_result = result = []
            self._current_epoch = 0

            self._log_start()
            self._timer_start()

            try:
                while self._current_epoch < self.total_epoch:
                    result.append(self.step())

            except KeyboardInterrupt:
                pass

            finally:
                self._timer_stop()
                self._log_stop()

                if self.save_and_load and self._current_epoch:
                    self._load()

            return result

        finally:
            self._current_run_result = None
            self._close()

    #
    # State dictionary handler: used in saving and loading parameters
    #

    def state_dict(self) -> 'OrderedDict[str, typing.Union[int, float, OrderedDict[str, torch.Tensor]]]':

        state_dict = OrderedDict()
        state_dict['best_loss'] = self._best_loss
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        if isinstance(self.criterion, torch.nn.Module):
            state_dict['criterion'] = self.criterion.state_dict()
        return state_dict

    def load_state_dict(
            self,
            state_dict: 'OrderedDict[str, typing.Union[int, float, OrderedDict[str, torch.Tensor]]]'
    ) -> None:

        self._best_loss = state_dict['best_loss']
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if isinstance(self.criterion, torch.nn.Module):
            self.criterion.load_state_dict(state_dict['criterion'])

    #
    # Device-moving Methods
    #

    def to(self: _t, *args, **kwargs) -> _t:  # overwrite this in subclass, for further features

        self._to_set(*args, **kwargs)
        self._to_apply(self.model)
        if isinstance(self.criterion, torch.nn.Module):
            self._to_apply(self.criterion)
        return self

    # Internal Device-moving Methods

    def _to_set(self, *args, **kwargs) -> None:

        self.__to_args = args
        self.__to_kwargs = kwargs

    def _to_apply(self, v: _v) -> _v:

        if self.__to_args or self.__to_kwargs:
            return v.to(*self.__to_args, **self.__to_kwargs)
        return v

    def _to_apply_multi(self, *v: _v) -> typing.Sequence[_v]:

        return tuple(map(self._to_apply, v))

    # Internal Timing Functions

    def _timer_start(self) -> None:

        import time

        self._require_context()

        if self.use_timer:
            self._time_start = time.time()

    def _timer_stop(self) -> None:

        import time

        self._require_context()

        if self.use_timer:
            self._time_stop = time.time()

    # Internal Logging Methods

    def _log_start(self) -> None:

        if self.verbose:
            self.log_function(f"\n<Start Learning> \t\t\t\tTotal {self.total_epoch} epochs")

    def _log_step(self, epoch: int) -> None:

        if self.verbose:
            self.log_function(f'\nEpoch {epoch}')

    def _log_train_doing(self, loss: float, iteration: int) -> None:

        if self.verbose:
            self.log_function(
                f'\r[Train]\t '
                f'Progress: {iteration * self.train_batch_size}/{self.train_dataset_length} '
                f'({100. * iteration / self.train_loader_length:05.2f}%), \tLoss: {loss:.6f}',
                end=' '
            )

    def _log_train_done(self, loss: float, accuracy: float) -> None:

        if self.verbose:
            # self.log_function(
            #     f'\r[Train]\t '
            #     f'Progress: {self.train_dataset_length}/{self.train_dataset_length} (100.00%), '
            #     f'\tTotal accuracy: {100. * accuracy:.2f}%'
            # )
            self.log_function(
                f'\r[Train]\t '
                f'Average loss: {loss:.5f}, '
                f'\t\tTotal accuracy: {100. * accuracy:05.2f}% '
            )

    def _log_eval(self, loss: float, accuracy: float, test: typing.Optional[bool] = False) -> None:

        if self.verbose:
            mode = '\n[Test]' if test else '[Eval]'
            self.log_function(
                f'{mode}\t '
                f'Average loss: {loss:.5f}, '
                f'\t\tTotal accuracy: {100. * accuracy:05.2f}% '
            )

    def _log_stop(self) -> None:

        if self.verbose:

            if self.use_timer:
                duration = self._time_stop - self._time_start
                duration_min = int(duration // 60)
                duration_sec = duration % 60
                duration = "\tDuration: "
                duration += f"{duration_min:02}m " if duration_min else ""
                duration += f"{duration_sec:05.2f}s"

            else:
                duration = ""

            if self.save_and_load:
                ll = f"\tLeast loss: {self._best_loss:.4f}"
            else:
                ll = ""

            self.log_function(
                "\n<Stop Learning> " + ll + duration
            )

    # Internal Parameter Methods

    def _load(self) -> None:

        self._require_context()

        if self.save_and_load:
            self.load_state_dict(torch.load(self._processing_fn))

    def _save(self) -> None:

        self._require_context()

        if self.save_and_load:
            torch.save(self.state_dict(), self._processing_fn)

    # Internal Context Methods

    def _open(self):

        import os

        if self.save_and_load:
            self._processing_fn = os.path.join(self.snapshot_dir, f'_processing_{id(self)}.pt')
            os.makedirs(self.snapshot_dir, exist_ok=True)

        self._closed = False

    def _close(self) -> None:

        import os

        if self._closed:
            return

        if self.save_and_load:
            try:
                os.remove(self._processing_fn)
            except FileNotFoundError:
                pass
            self._processing_fn = None

        self._closed = True

    def _require_context(self):

        if self._closed:
            raise RuntimeError('Already closed: %r' % self)

    # Internal Running Methods

    def _train(self) -> typing.Tuple[float, float]:

        self._require_context()

        data = self.train_iter
        total_loss, total_accuracy = 0., 0.
        total_batch = len(data)
        verbose = self.verbose
        log_interval = self.log_interval

        self.model.train()

        for iteration, (x, y) in enumerate(self.train_iter, 1):
            x, y = self._to_apply_multi(x, y)
            prediction = self.model(x)
            loss = self.criterion(prediction, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            with torch.no_grad():
                total_loss += loss.item()
                total_accuracy += torch.eq(torch.argmax(prediction, 1), y).float().mean().item()

            if iteration % log_interval == 0 and verbose:
                self._log_train_doing(loss.item(), iteration)

        avg_loss = total_loss / total_batch
        avg_accuracy = total_accuracy / total_batch

        self._log_train_done(avg_loss, avg_accuracy)

        return avg_loss, avg_accuracy

    @torch.no_grad()
    def _evaluate(self, *, test: typing.Optional[bool] = False) -> typing.Tuple[float, float]:

        data = self.test_iter if test else self.val_iter
        assert data is not None, "You must provide dataset for evaluating method."
        total_loss, total_accuracy = 0., 0.
        total_batch = len(data)

        self.model.eval()

        for x, y in data:
            x, y = self._to_apply_multi(x, y)
            prediction = self.model(x)
            loss = self.criterion(prediction, y)

            total_loss += loss.item()
            total_accuracy += torch.eq(torch.argmax(prediction, 1), y).float().mean().item()

        avg_loss = total_loss / total_batch
        avg_accuracy = total_accuracy / total_batch

        self._log_eval(avg_loss, avg_accuracy, test)

        return avg_loss, avg_accuracy

    def _do_step_task(self):

        self._require_context()

        if self.step_task:
            args = (self._current_epoch, self._current_run_result)[:self.step_task_mode]
            self.step_task(*args)

    # Log function: overwrite this to use custom logging hook

    log_function = staticmethod(print)


# Clear typing variables from namespace

del typing, types, partial, PathLike, _data_type, _optional_path_type, _loss_function_type, _t, _v
