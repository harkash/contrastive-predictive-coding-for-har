import copy
from time import time

import numpy as np
import torch
from torch import optim

from meter import RunningMeter, BestMeter
from utils import compute_best_metrics, update_loss, save_meter


def learn_model(model, data_loaders, dataset_sizes, args):
    best_model_wts = copy.deepcopy(model.state_dict())

    # Tracking meter
    running_meter = RunningMeter(args=args)
    best_meter = BestMeter()

    # Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25,
                                          gamma=0.8)

    for epoch in range(0, args.num_epochs):
        since = time()

        # Training
        model, optimizer = train(model, data_loaders["train"],
                                 optimizer, args, epoch,
                                 dataset_sizes["train"],
                                 running_meter)

        scheduler.step()

        # Evaluating on the validation data
        evaluate(model, data_loaders["val"], args, epoch,
                 phase="val", dataset_size=dataset_sizes["val"],
                 running_meter=running_meter)

        # Evaluating on the test data
        evaluate(model, data_loaders["test"], args, epoch,
                 phase="test", dataset_size=dataset_sizes["test"],
                 running_meter=running_meter)

        # Saving the logs
        save_meter(args, running_meter)

        # Updating the best weights
        if running_meter.loss["val"][-1] < best_meter.loss["val"]:
            print('Updating the best val loss at epoch: {}, since {} < '
                  '{}'.format(epoch, running_meter.loss["val"][-1],
                              best_meter.loss["val"]))
            best_meter = compute_best_metrics(running_meter, best_meter)
            running_meter.update_best_meter(best_meter)
            save_meter(args, running_meter)

            best_model_wts = copy.deepcopy(model.state_dict())

        # Printing the time taken
        time_elapsed = time() - since
        print('Epoch {} completed in {:.0f}m {:.0f}s'
              .format(epoch, time_elapsed // 60, time_elapsed % 60))

    # Printing the best metrics
    best_meter.display()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train(model, data_loader, optimizer, args, epoch, dataset_size,
          running_meter):
    # Setting the model to training mode
    model.train()

    # To track the loss and other metrics
    running_loss = 0.0
    running_corrects = 0.0
    running_corrects_steps = np.zeros(args.num_steps_prediction)

    # Iterating over the data
    for i, (inputs, _) in enumerate(data_loader):
        inputs = inputs.float().to(args.device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            batch_acc, loss, batch_acc_steps = model(inputs)

            loss.backward()
            optimizer.step()

        # Appending predictions and loss
        running_loss += loss.item() * inputs.size(0)
        running_corrects += batch_acc * inputs.size(0)
        running_corrects_steps += batch_acc_steps

    # Statistics
    loss = running_loss / dataset_size
    accuracy = running_corrects / dataset_size
    accuracy_steps = running_corrects_steps / dataset_size

    update_loss(phase="train", running_meter=running_meter, loss=loss,
                accuracy=accuracy, epoch=epoch, accuracy_steps=accuracy_steps)

    return model, optimizer


def evaluate(model, data_loader, args, epoch, phase, dataset_size,
             running_meter):
    model.eval()

    # To track the loss and other metrics
    running_loss = 0.0
    running_corrects = 0.0
    running_corrects_steps = np.zeros(args.num_steps_prediction)

    # Iterating over the data
    for i, (inputs, _) in enumerate(data_loader):
        inputs = inputs.float().to(args.device)

        with torch.set_grad_enabled(False):
            batch_acc, loss, batch_acc_steps = model(inputs)

        # Appending predictions and loss
        running_loss += loss.item() * inputs.size(0)
        running_corrects += batch_acc * inputs.size(0)
        running_corrects_steps += batch_acc_steps

    # Statistics
    loss = running_loss / dataset_size
    accuracy = running_corrects / dataset_size
    accuracy_steps = running_corrects_steps / dataset_size

    update_loss(phase=phase, running_meter=running_meter, loss=loss,
                accuracy=accuracy, epoch=epoch, accuracy_steps=accuracy_steps)

    return
