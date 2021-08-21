# Easyrun-Pytorch
* One-time Pytorch Trainer Utility for general models
* written by Dong Ha Kim, in Yonsei University.
    
## Available Parameters:
* :param model: (torch.nn.Module) model object to use.
* :param optimizer: (torch.optim.Optimizer) optimizer.
* :param criterion: (torch.nn.Module) loss function or model object.
* :param epoch: (int) total epochs.
* :param train_iter: train data loader, or train dataset.
* :param val_iter: validation data loader, or validation dataset.
* :param test_iter: test data loader, or test dataset.
* :param snapshot_dir: (str) provide if you want to use checkpoint saving and loading.
      in this path name, model's weight parameter at best(least) loss will be temporarily saved.
* :param verbose: (bool) verbosity. with turning it on, you can view learning logs.
      default value is True.
* :param timer: (bool) provide with verbosity, if you want to use time-checking.
      default value is True.
* :param log_interval: (int) provide with verbosity, if you want to set your log interval.
      default value is 20.

## Available Methods:
* (): (call) repeat training and validating for all epochs, followed by testing.
* to(device): apply to(device) in model, criterion, and all tensors.
* train(): run training one time with train dataset.
* evaluate(): run validating one time with validation dataset.
* step(): run training, followed by validating one time.
* run(): repeat training and validating for all epochs.
* test(): run testing one time with test dataset.
* state_dict(): returns state dictionary of trainer class.
* load_state_dict(): loads state dictionary of trainer class.

## Simple Usage:
Trainer(model, 'CrossEntropyLoss', optimizer, num_epochs, train_loader, val_loader, test_loader, verbose=True, timer=True, snapshot_dir='.').to(device)()
