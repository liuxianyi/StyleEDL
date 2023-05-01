import math
import torch
import numpy as np


class LDL_measurement(object):
    '''
    compute and stores the average LDL measurement
    '''
    def __init__(self, num_classes):
        self.reset()
        self.num_classes = num_classes
        self.current_size = 1

    def reset(self):
        self.current_result = {
            'klDiv': 0,
            'cosine': 0,
            'intersection': 0,
            'chebyshev': 0,
            'clark': 0,
            'canberra': 0,
            'squareChord': 0,
            'sorensendist': 0
        }
        self.sum = {
            'klDiv': 0,
            'cosine': 0,
            'intersection': 0,
            'chebyshev': 0,
            'clark': 0,
            'canberra': 0,
            'squareChord': 0,
            'sorensendist': 0
        }
        self.count = 0

    def update(self, output, target, n=1):
        self.current_size = n

        self.current_result['klDiv'] = self.KLDiv(output, target)
        self.current_result['cosine'] = self.cosine(output, target)
        self.current_result['intersection'] = self.intersection(output, target)
        self.current_result['chebyshev'] = self.chebyshev(output, target)
        self.current_result['clark'] = self.clark(output, target)
        self.current_result['canberra'] = self.canberra(output, target)
        self.current_result['squareChord'] = self.squareChord(output, target)
        self.current_result['sorensendist'] = self.sorensendist(output, target)

        # all
        self.count += n
        for key in self.sum:
            self.sum[key] += self.current_result[key]

    def average(self):
        avg = {}
        for key in self.sum.keys():
            avg[key] = self.sum[key] / self.count
        return avg

    def value(self):
        current_avg = {}
        for key in self.current_result.keys():
            current_avg[key] = self.current_result[key] / self.current_size
        return current_avg

    def KLDiv(self, output, target):
        distribution_predict = output.cpu().detach().numpy()
        distribution_real = target.cpu().detach().numpy()
        batch_KL = np.nansum(distribution_real *
                             np.log(distribution_real / distribution_predict))
        return batch_KL

    def cosine(self, output, target):
        distribution_predict = output.cpu().detach().numpy()
        distribution_real = target.cpu().detach().numpy()
        return np.sum(
            np.sum(distribution_real * distribution_predict, 1) /
            (np.sqrt(np.sum(distribution_real**2, 1)) *
             np.sqrt(np.sum(distribution_predict**2, 1))))

    def intersection(self, output, target):
        distribution_predict = output.cpu().detach().numpy()
        distribution_real = target.cpu().detach().numpy()

        concat = np.dstack((distribution_predict, distribution_real))
        concat = np.min(concat, -1)

        return np.sum(concat)

    def chebyshev(self, output, target):
        distribution_predict = output.cpu().detach().numpy()
        distribution_real = target.cpu().detach().numpy()
        return np.sum(
            np.max(np.abs(distribution_real - distribution_predict), 1))

    def clark(self, output, target):
        distribution_predict = output.cpu().detach().numpy()
        distribution_real = target.cpu().detach().numpy()
        numerator = (distribution_real - distribution_predict)**2
        denominator = (distribution_real + distribution_predict)**2

        return np.nansum(
            np.sqrt(
                np.nansum(numerator / denominator, axis=1) / self.num_classes))

    def canberra(self, output, target):
        distribution_predict = output.cpu().detach().numpy()
        distribution_real = target.cpu().detach().numpy()
        numerator = np.abs(distribution_real - distribution_predict)
        denominator = distribution_real + distribution_predict
        return np.nansum(numerator / denominator) / self.num_classes

    def squareChord(self, output, target):
        distribution_predict = output.cpu().detach().numpy()
        distribution_real = target.cpu().detach().numpy()
        numerator = (np.sqrt(distribution_real) -
                     np.sqrt(distribution_predict))**2
        denominator = np.nansum(numerator)
        return denominator

    def sorensendist(self, output, target):
        distribution_predict = output.cpu().detach().numpy()
        distribution_real = target.cpu().detach().numpy()
        numerator = np.sum(np.abs(distribution_real - distribution_predict), 1)
        denominator = np.sum(distribution_real + distribution_predict, 1)
        return np.nansum(numerator / denominator)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def average(self):
        return self.avg

    def value(self):
        return self.val

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """
    def __init__(self, difficult_examples=True):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())
        self.filenames = []

    def add(self, output, target, filename):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'

        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'

        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

        self.filenames += filename  # record filenames

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(
                scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= pos_count
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.clone().cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.clone().cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)

    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)
            Nc[k] = np.sum(targets * (scores >= 0))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1
