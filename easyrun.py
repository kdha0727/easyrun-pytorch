"""
One-time Pytorch Trainer Utility and Useful Functions:
    written by Dong Ha Kim, in Yonsei University.
"""

# Frameworks and Internal Modules

import platform
import os
import sys
import time

from collections import OrderedDict

import torch
import torch.nn.modules.loss
import torch.utils.data


# Typing Variables

import typing
import types
_loss_function_type = typing.Union[
    torch.nn.modules.loss._Loss,  # pylint: disable=protected-access
    types.FunctionType,
    types.BuiltinFunctionType,
    str,
]
_iterator_type = typing.Union[
    torch.utils.data.Dataset,
    torch.utils.data.DataLoader,
]
_optional_path_type = typing.Optional[
    typing.Union[os.PathLike, str]
]
_t = typing.TypeVar('_t')
_v = typing.TypeVar('_v')


#
# Tool Functions
#

def runtime_info(device: typing.Union[torch.device, str] = None) -> str:
    """Runtime information utility."""
    return (
        f"<Runtime Information>\n"
        f"OS version: \t\t{platform.platform()}\n"
        f"Python version:\t\t{sys.version[:0x20]}\n"
        f"Torch version:\t\t{torch.__version__}\n"
        f"Torch device:\t\t{device or ('cuda' if torch.cuda.is_available() else 'cpu')}"
    )


def dataset_info(train: _iterator_type, val: _iterator_type, test: _iterator_type, *, loader=False) -> str:
    """Dataset information utility."""
    return (
        f"<{'DataLoader' if loader else 'Dataset'} Information>\n"
        f"Train {'Batch' if loader else 'Dataset'}: \t\t{len(train)}\n"
        f"Validation {'Batch' if loader else 'Dataset'}: \t{len(val)}\n"
        f"Test {'Batch' if loader else 'Dataset'}: \t\t{len(test)}"
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
        evaluate(): run validating one time with validation dataset.
        step(): run training, followed by validating one time.
        run(): repeat training and validating for all epochs.
        test(): run testing one time with test dataset.
        state_dict(): returns state dictionary of trainer class.
        load_state_dict(): loads state dictionary of trainer class.

    """

    #
    # Constructor
    #

    def __init__(
            self,
            model: torch.nn.Module,
            criterion: _loss_function_type,
            optimizer: torch.optim.Optimizer,
            epoch: int,
            train_iter: _iterator_type,
            val_iter: _iterator_type,
            test_iter: typing.Optional[_iterator_type] = None,
            snapshot_dir: _optional_path_type = None,
            verbose: typing.Optional[bool] = True,
            timer: typing.Optional[bool] = False,
            log_interval: typing.Optional[int] = 20,
    ) -> None:

        assert isinstance(model, torch.nn.Module), \
            "Invalid model type: %s" % model.__class__.__name__
        assert isinstance(optimizer, torch.optim.Optimizer), \
            "Invalid model type: %s" % optimizer.__class__.__name__
        if not callable(criterion):
            if isinstance(criterion, str):
                criterion_cls = getattr(torch.nn, criterion, None)
                if criterion_cls is not None:
                    criterion = criterion_cls()
                else:
                    import torch.nn.functional as func
                    try:
                        criterion = getattr(func, criterion)
                    except AttributeError as exc:
                        raise TypeError("Invalid criterion name: %s" % criterion) from exc
            else:
                raise TypeError("Invalid criterion type: %s" % criterion.__class__.__name__)

        dataset_type = (
            torch.utils.data.Dataset,
            torch.utils.data.DataLoader,
        )
        assert isinstance(train_iter, dataset_type), \
            "Invalid train_iter type: %s" % train_iter.__class__.__name__
        assert isinstance(val_iter, dataset_type), \
            "Invalid val_iter type: %s" % val_iter.__class__.__name__
        assert test_iter is None or isinstance(test_iter, dataset_type), \
            "Invalid test_iter type: %s" % test_iter.__class__.__name__

        assert isinstance(epoch, int) and epoch > 0, \
            "Epoch is expected to be positive int, got %s" % epoch
        assert isinstance(log_interval, int) and log_interval > 0, \
            "Log Interval is expected to be positive int, got %s" % log_interval

        self.model: torch.nn.Module = model
        self.criterion: _loss_function_type = criterion
        self.optimizer: torch.optim.Optimizer = optimizer

        self.total_epoch: int = epoch

        self.train_iter: _iterator_type = train_iter
        self.val_iter: _iterator_type = val_iter
        self.test_iter: typing.Optional[_iterator_type] = test_iter

        self.snapshot_dir: _optional_path_type = snapshot_dir
        self.verbose: bool = verbose
        self.use_timer: bool = timer
        self.log_interval: int = log_interval

        self.current_epoch: int = 0
        self.best_loss: typing.Optional[float] = None
        self.processing_fn: _optional_path_type = None
        self._closed: bool = False
        self._time_start: typing.Optional[float] = None
        self._time_stop: typing.Optional[float] = None

        self.train_batch_size: int = train_iter.batch_size
        self.train_loader_length: int = len(train_iter)
        self.train_dataset_length: int = len(getattr(train_iter, 'dataset', train_iter))

        if snapshot_dir is not None:
            os.makedirs(snapshot_dir, exist_ok=True)
            self.processing_fn = os.path.join(snapshot_dir, f'_processing_{id(self)}.pt')

        self.__to_args: typing.Optional[tuple] = None
        self.__to_kwargs: typing.Optional[dict] = None

    #
    # De-constructor: executed in buffer-cleaning in python exit
    #

    def __del__(self) -> None:
        self._close()

    #
    # Context manager magic methods
    #

    def __enter__(self: _t) -> _t:
        return self

    def __exit__(self, exc_info, exc_class, exc_traceback) -> None:
        try:
            self._close()
        except Exception:
            if not (exc_info or exc_class or exc_traceback) is not None:
                pass  # executed in exception handling - just let python raise that exception
            else:
                raise

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
        self.current_epoch += 1
        return result

    def evaluate(self) -> typing.Tuple[float, float]:

        return self._evaluate(test=False)

    def test(self) -> typing.Tuple[float, float]:

        return self._evaluate(test=True)

    def step(self) -> typing.Tuple[float, float, float, float]:

        self._log_step(self.current_epoch + 1)

        train_loss, train_accuracy = self._train()
        test_loss, test_accuracy = self._evaluate(test=False)

        # Save the model having the smallest validation loss
        if self.best_loss is None or test_loss < self.best_loss:
            self.best_loss = test_loss
            self._save()

        self.current_epoch += 1

        return train_loss, train_accuracy, test_loss, test_accuracy

    def run(self) -> typing.List[typing.Tuple[float, float, float, float]]:

        result = []
        self._log_start()
        self._timer_start()

        while self.current_epoch < self.total_epoch:
            result.append(self.step())

        self._timer_stop()
        self._log_stop()

        if self.current_epoch:
            self._load()

        self._close()
        return result

    #
    # State dictionary handler: used in saving and loading parameters
    #

    def state_dict(self) -> 'OrderedDict[str, typing.Union[int, float, OrderedDict[str, torch.Tensor]]]':

        state_dict = OrderedDict()
        state_dict['best_loss'] = self.best_loss
        state_dict['current_epoch'] = self.current_epoch
        state_dict['model'] = self.model.state_dict()
        state_dict['criterion'] = self.criterion.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        return state_dict

    def load_state_dict(
            self,
            state_dict: 'OrderedDict[str, typing.Union[int, float, OrderedDict[str, torch.Tensor]]]'
    ) -> None:

        self.best_loss = state_dict['best_loss']
        self.current_epoch = state_dict['current_epoch']
        self.model.load_state_dict(state_dict['model'])
        self.criterion.load_state_dict(state_dict['criterion'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    #
    # Device-moving Methods
    #

    def to(self: _t, *args, **kwargs) -> _t:  # overwrite this in subclass, for further features

        self._to_set(*args, **kwargs)
        self._to_apply(self.model)
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

        if self.use_timer:
            self._time_start = time.time()

    def _timer_stop(self) -> None:

        if self.use_timer:
            self._time_stop = time.time()

    # Internal Logging Methods

    def _log_start(self) -> None:

        if self.verbose:
            self.log_function(f"\n<Start Learning> \t\t\t\tTotal {self.total_epoch} epochs", end='\n\n')

    def _log_step(self, epoch: int) -> None:

        if self.verbose:
            self.log_function(f'Epoch {epoch}', end='\n')

    def _log_train_doing(self, loss: float, iteration: int) -> None:

        if self.verbose:
            self.log_function(
                f'\r[Train]\t '
                f'Progress: {iteration * self.train_batch_size}/{self.train_dataset_length} '
                f'({100. * iteration / self.train_loader_length:.2f}%), \tLoss: {loss:.6f}',
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
                f'\t\tTotal accuracy: {100. * accuracy:.2f}%'
            )

    def _log_eval(self, loss: float, accuracy: float, test: typing.Optional[bool] = False) -> None:

        if self.verbose:
            mode = 'Test' if test else 'Eval'
            self.log_function(
                f'[{mode}]\t '
                f'Average loss: {loss:.5f}, '
                f'\t\tTotal accuracy: {100. * accuracy:.2f}%\n'
            )

    def _log_stop(self) -> None:

        if self.verbose:

            if self.use_timer:
                duration = self._time_stop - self._time_start
                duration_min = int(duration // 60)
                duration_sec = duration % 60
                duration = "Duration: "
                duration += f"{duration_min}m " if duration_min else ""
                duration += f"{duration_sec:2.2f}s"

            else:
                duration = ""

            self.log_function(
                f"<Stop Learning> \tLeast loss: {self.best_loss:.4f}\t{duration}", end='\n\n'
            )

    # Internal Parameter Methods

    def _load(self) -> None:

        if self.processing_fn is not None:
            self.load_state_dict(torch.load(self.processing_fn))

    def _save(self) -> None:

        if self.processing_fn is not None:
            torch.save(self.state_dict(), self.processing_fn)

    # Internal Cleaning Methods

    def _close(self) -> None:

        if not self._closed:
            if self.processing_fn is not None:
                try:
                    os.remove(self.processing_fn)
                except FileNotFoundError:
                    pass
            self._closed = True

    # Internal Running Methods

    def _train(self) -> typing.Tuple[float, float]:

        verbose = self.verbose
        log_interval = self.log_interval
        data = self.train_iter
        total_loss, total_accuracy = 0., 0.
        total_batch = len(data)

        self.model.train()

        for iteration, (x, y) in enumerate(self.train_iter, 1):

            x, y = self._to_apply_multi(x, y)
            self.optimizer.zero_grad()
            prediction = self.model(x)
            loss = self.criterion(prediction, y)
            loss.backward()
            self.optimizer.step()

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

        if test:
            data = self.test_iter
            if data is None:
                raise TypeError('You should provide test dataset to use test method.')
        else:
            data = self.val_iter
        total_loss, total_accuracy = 0., 0.
        total_batch = len(data)

        self.model.eval()

        for x, y in data:

            x, y = self._to_apply_multi(x, y)
            prediction = self.model(x)
            total_loss += self.criterion(prediction, y).item()
            total_accuracy += torch.eq(torch.argmax(prediction, 1), y).float().mean().item()

        avg_loss = total_loss / total_batch
        avg_accuracy = total_accuracy / total_batch
        self._log_eval(avg_loss, avg_accuracy, test)

        return avg_loss, avg_accuracy

    # Log function: overwrite this to use custom logging hook

    log_function = staticmethod(print)


# Clear typing variables from namespace

del typing, types, _iterator_type, _optional_path_type, _loss_function_type, _t, _v
