import pickle
from datetime import date

import numpy as np
import os
import random
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def model_save_name(args, classifier=False):
    """
    To obtain the name of the saved model based on input parameters
    :param args: arguments
    :param classifier: boolean flag for the classifier
    :return: name of the logs/model
    """
    cpc = '{0.dataset}_k_{0.num_steps_prediction}'.format(args)

    # CPC training settings
    training_settings = '_lr_{0.learning_rate}_bs_{0.batch_size}'.format(args)

    # Classifier
    classification = ""
    if classifier:
        if args.saved_model is not None:
            classification += '_saved_model_True'

        classification += "_cls_lr_{0.classifier_lr}_{0.learning_schedule}_" \
                          "cls_bs_{0.classifier_batch_size}".format(args)

    name = cpc + training_settings + classification

    return name


def compute_best_metrics(running_meter, best_meter, classifier=False):
    """
    Computing the best metrics for the pre-training and classification
    :param running_meter: log
    :param best_meter: previous best log
    :param classifier: boolean flag to decide whether to compute by the best
    performance by lowest loss or highest validation F1-score
    :return: updated best meter
    """
    if classifier:
        loc = np.argmax(running_meter.f1_score['val'])
    else:
        min_loss = np.min(running_meter.loss['val'])  # Minimum loss
        loc = np.where(running_meter.loss['val'] == min_loss)[
            0][-1]  # The latest epoch to give the lowest loss

    # Epoch where the best validation loss was obtained
    epoch = running_meter.epochs[loc]

    # Updating the best meter with values based on the epoch
    phases = ['train', 'val', 'test']
    for phase in phases:
        best_meter.update(
            phase,
            running_meter.loss[phase][loc],
            running_meter.accuracy[phase][loc],
            running_meter.f1_score[phase][loc],
            running_meter.f1_score_weighted[phase][loc],
            running_meter.confusion_matrix[phase][loc],
            running_meter.accuracy_steps[phase][loc],
            epoch)

    return best_meter


def update_loss(phase, running_meter, loss, accuracy, epoch, accuracy_steps):
    """
    Updating the pre-training loss in the logs for the epoch
    :param phase: train/val/test phases
    :param running_meter: logs
    :param loss: loss for the specific epoch
    :param accuracy: overall accuracy during pre-training for the epoch
    :param epoch: epoch number
    :param accuracy_steps: accuracy for each step
    :return:
    """
    running_meter.update(phase, loss, accuracy, 0, 0, [], accuracy_steps)

    # printing the metrics
    print("The epoch: {} | phase: {} | loss: {:.4f} | accuracy: {:.4f} | mean "
          "f1-score: {:.4f} | weighted f1-score: {:.4f}"
          .format(epoch, phase, loss, accuracy, 0, 0))

    return


def save_meter(args, running_meter, finetune=False):
    """
    Saving the logs
    :param args: arguments
    :param running_meter: running meter object to save
    :param mlp: if saving during the MLP training, then adds '_eval_log.pkl'
    to the end
    :return: nothing
    """
    name = model_save_name(args, classifier=finetune)
    save_name = name + '_finetune_log.pkl' if finetune else name + '_log.pkl'

    # Creating logs by the date now. To make stuff easier
    folder = os.path.join('saved_logs', date.today().strftime(
        "%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, save_name), 'wb') as f:
        pickle.dump(running_meter, f, pickle.HIGHEST_PROTOCOL)

    return


def save_model(model, args, classifier=False):
    """
    Saves the weights from the model
    :param model: model being trained
    :param args: arguments
    :param classifier: if we are training a classifier
    :return: nothing
    """
    name = model_save_name(args, classifier=classifier)

    # Creating logs by the date now. To make stuff easier
    folder = os.path.join('models', date.today().strftime(
        "%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)

    model_name = os.path.join(folder, name + '.pkl')

    torch.save(model.state_dict(), model_name)

    return


def set_all_seeds(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    return


def compute_classifier_metrics(actual_labels, pred_labels, phase,
                               running_meter, loss, epoch):
    acc = accuracy_score(actual_labels, pred_labels)
    f_score_weighted = f1_score(actual_labels, pred_labels, average='weighted')
    f_score_macro = f1_score(actual_labels, pred_labels, average='macro')
    conf_matrix = confusion_matrix(y_true=actual_labels, y_pred=pred_labels,
                                   normalize="true")
    running_meter.update(phase, loss, acc, f_score_macro, f_score_weighted,
                         conf_matrix, [])

    # printing the metrics
    print("The epoch: {} | phase: {} | loss: {:.4f} | accuracy: {:.4f} | mean "
          "f1-score: {:.4f} | weighted f1-score: {:.4f}"
          .format(epoch, phase, loss, acc, f_score_macro, f_score_weighted))

    return running_meter
