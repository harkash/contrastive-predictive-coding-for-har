import numpy as np


class RunningMeter:
    def __init__(self, args):
        # Tracking at a per epoch level
        self.loss = {'train': [], 'val': [], 'test': []}
        self.accuracy = {'train': [], 'val': [], 'test': []}
        self.f1_score = {'train': [], 'val': [], 'test': []}
        self.f1_score_weighted = {'train': [], 'val': [], 'test': []}
        self.confusion_matrix = {'train': [], 'val': [], 'test': []}
        self.accuracy_steps = {'train': [], 'val': [], 'test': []}

        self.epochs = np.arange(0, args.num_epochs)

        self.best_meter = BestMeter()

        self.args = args

    def update(self, phase, loss, accuracy, f1_score,
               f1_score_weighted, confusion_matrix, accuracy_steps):
        # Update the metrics for every phase
        self.loss[phase].append(loss)
        self.accuracy[phase].append(accuracy)
        self.f1_score[phase].append(f1_score)
        self.f1_score_weighted[phase].append(f1_score_weighted)
        self.confusion_matrix[phase].append(confusion_matrix)
        self.accuracy_steps[phase].append(accuracy_steps)

    def get(self):
        return self.loss, self.accuracy, self.f1_score, self.f1_score_weighted, \
               self.confusion_matrix, self.accuracy_steps, self.epochs

    def update_best_meter(self, best_meter):
        self.best_meter = best_meter


class BestMeter:
    def __init__(self):
        # Storing the best values
        self.loss = {'train': np.inf, 'val': np.inf, 'test': np.inf}
        self.accuracy = {'train': 0.0, 'val': 0.0, 'test': 0.0}
        self.f1_score = {'train': 0.0, 'val': 0.0, 'test': 0.0}
        self.f1_score_weighted = {'train': 0.0, 'val': 0.0, 'test': 0.0}
        self.confusion_matrix = {'train': [], 'val': [], 'test': []}
        self.accuracy_steps = {'train': [], 'val': [], 'test': []}
        self.epoch = 0

    def update(self, phase, loss, accuracy, f1_score,
               f1_score_weighted, confusion_matrix, accuracy_steps, epoch):
        self.loss[phase] = loss
        self.accuracy[phase] = accuracy
        self.f1_score[phase] = f1_score
        self.f1_score_weighted[phase] = f1_score_weighted
        self.confusion_matrix[phase] = confusion_matrix
        self.accuracy_steps[phase] = accuracy_steps
        self.epoch = epoch

    def get(self):
        return self.loss, self.accuracy, self.f1_score, self.f1_score_weighted, \
               self.confusion_matrix, self.epoch, self.accuracy_steps

    def display(self):
        print('The best epoch is {}'.format(self.epoch))
        for phase in ['train', 'val', 'test']:
            print('Phase: {}, loss: {}, accuracy: {}, f1_score: {}, f1_score '
                  'weighted: {}'
                  .format(phase, self.loss[phase], self.accuracy[phase],
                          self.f1_score[phase], self.f1_score_weighted[phase]),
                  self.accuracy_steps)
