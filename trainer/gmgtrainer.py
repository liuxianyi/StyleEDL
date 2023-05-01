# -*- encoding: utf-8 -*-
import time
import torch
import torch.nn as nn
from torch.optim import SGD

from loss import AdvDivLoss
from .basetrainer import BaseTrainer


class gmgTrainer(BaseTrainer):
    def __init__(self, opt, logger, writer):
        super().__init__(opt, logger, writer)
        logger.name = __name__
        self.softmax = nn.Softmax(dim=1)

    def set_loss(self):
        self.loss = nn.KLDivLoss(reduction='batchmean')

        self.loss_div_1 = AdvDivLoss(self.opt['parts'])
        self.loss_div_1.to(self.opt['device'])

        self.loss_div_2 = AdvDivLoss(self.opt['parts'])
        self.loss_div_2.to(self.opt['device'])

        self.loss_mse = nn.MSELoss()

    def set_optimizer(self):
        self.optimizer = SGD(self.model.parameters(),
                             lr=self.opt['lr'],
                             momentum=self.opt['momentum'],
                             weight_decay=eval(self.opt['weight_decay']))

    def train_epoch(self, epoch):
        start_time = time.time()
        for i, (inputs, labels, cls) in enumerate(self.dataloader_train):
            inputs = inputs.to(self.opt['device'])
            labels = labels.to(self.opt['device'])
            cls = cls.to(self.opt['device'])
            outputs, gcn, fc1, fc2 = self.model(inputs)

            result2 = outputs[0]
            for j in range(1, self.opt['parts']):
                result2 = outputs[j] + result2
            result2 /= self.opt['parts']

            result = self.opt['mu'] * result2 + (1 - self.opt['mu']) * gcn

            self.meters_dict['ldl'].update(result, labels, inputs.size(0))
            prediction = torch.max(result, 1)[1]
            self.meters_dict['acc'].update(
                sum(prediction == cls) / inputs.size(0))

            loss_dis_1 = self.loss(torch.log(gcn), labels)

            loss_dis_2 = self.loss(torch.log(outputs[0]), labels)
            for j in range(1, self.opt['parts']):
                loss_dis_2 += self.loss(torch.log(outputs[j]), labels)

            loss_dis_2 /= self.opt['parts']

            loss_dis = (loss_dis_1 + loss_dis_2) / 2 # L_pred

            loss_div_1 = self.loss_div_1(fc1) # 
            loss_div_2 = self.loss_div_2(fc2)

            loss_div = (loss_div_1 + loss_div_2) / 2

            loss = loss_dis + loss_div / (loss_div / loss_dis).detach()

            self.meters_dict['loss'].update(loss.item(), inputs.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time = time.time() - start_time
            self.meters_dict['batch_time'].update(batch_time)
            start_time = time.time()

            if i % self.opt['display_interval'] == 0:
                print(
                    '{}, {} Epoch, {} Iter, Loss: {:.4f}, Batch time: {:.4f}, Acc: {:.4f}'
                    .format(time.strftime('%Y-%m-%d %H:%M:%S'), epoch, i,
                            self.meters_dict['loss'].value(),
                            self.meters_dict['batch_time'].value(),
                            self.meters_dict['acc'].value()))
                self.writer.add_scalar('train/acc',
                                       self.meters_dict['acc'].value(),
                                       global_step=self.train_steps)
                self.writer.add_scalar('loss/train_loss',
                                       self.meters_dict['loss'].value(),
                                       global_step=self.train_steps)
            self.train_steps += 1

    def test_epoch(self, epoch):
        start_time = time.time()
        for i, (inputs, labels, cls) in enumerate(self.dataloader_test):
            inputs = inputs.to(self.opt['device'])
            labels = labels.to(self.opt['device'])
            cls = cls.to(self.opt['device'])

            with torch.no_grad():
                inputs = inputs.to(self.opt['device'])
                labels = labels.to(self.opt['device'])
                cls = cls.to(self.opt['device'])

                outputs, gcn, fc1, fc2 = self.model(inputs)

                result2 = outputs[0]
                for j in range(1, self.opt['parts']):
                    result2 = outputs[j] + result2
                result2 /= self.opt['parts']

                result = self.opt['mu'] * result2 + (1 - self.opt['mu']) * gcn

                self.meters_dict['ldl'].update(result, labels, inputs.size(0))
                prediction = torch.max(result, 1)[1]
                self.meters_dict['acc'].update(
                    sum(prediction == cls) / inputs.size(0))

                loss_dis = self.loss(torch.log(outputs[0]), labels)
                loss_dis += self.loss(torch.log(gcn), labels)
                for j in range(1, self.opt['parts']):
                    loss_dis += self.loss(torch.log(outputs[j]), labels)

                loss_dis /= self.opt['parts'] * 2

                loss_div = self.loss_div_1(fc1)
                loss_div2 = self.loss_div_2(fc2)

                loss_divv = (loss_div + loss_div2) / 2

                loss = loss_dis + loss_divv / (loss_divv / loss_dis).detach()

                self.meters_dict['loss'].update(loss.item(), inputs.size(0))

            batch_time = time.time() - start_time
            self.meters_dict['batch_time'].update(batch_time)
            start_time = time.time()
            if i % self.opt['display_interval'] == 0:
                print(
                    '{}, {} Epoch, {} Iter, Loss: {:.4f}, Batch time: {:.4f}, Acc: {:.4f}'
                    .format(time.strftime('%Y-%m-%d %H:%M:%S'), epoch, i,
                            self.meters_dict['loss'].value(),
                            self.meters_dict['batch_time'].value(),
                            self.meters_dict['acc'].value()))

                self.writer.add_scalar('test/acc',
                                       self.meters_dict['acc'].value(),
                                       global_step=self.test_steps)
                self.writer.add_scalar('loss/test_loss',
                                       self.meters_dict['loss'].value(),
                                       global_step=self.test_steps)
            self.test_steps += 1
