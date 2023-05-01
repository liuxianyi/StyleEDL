# -*- encoding: utf-8 -*-
import os
import time

from numpy import inf

from utils import AverageMeter, write_result, LDL_measurement
from dataset import Dataset_LDL
from transforms import get_transforms
from scheduler import get_scheduler
from network import get_model

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader


class BaseTrainer():
    def __init__(self, opt, logger, writer) -> None:
        self.opt = opt
        self.logger = logger
        self.writer = writer
        self.logger.name = __name__

        # data
        self.dataloader_train = None
        self.dataloader_test = None

        # model
        self.model = None

        # optimizer
        self.optimizer = None
        self.scheduler = None

        # loss
        self.loss = None

        # statistical
        self.meters_dict = {}

        # train
        # self.save_mark = inf
        self.start_epoch = 0
        self.train_steps = 0
        self.test_steps = 0

    def set_model(self):
        self.model = get_model(self.opt['model'])(self.opt)

    def set_dataloader(self):
        # train
        transforms_train = get_transforms(self.opt['image_size'],
                                          'train',
                                          self.opt['dataset'],
                                          isNormalize=True)
        dataset_train = Dataset_LDL(
            os.path.join(self.opt['data_path'], self.opt['dataset']), 'train',
            transforms_train)
        self.dataloader_train = DataLoader(dataset_train,
                                           batch_size=self.opt['batch_size'],
                                           num_workers=self.opt['num_workers'],
                                           shuffle=True,
                                           drop_last=True)

        # test
        transforms_test = get_transforms(self.opt['image_size'],
                                         'test',
                                         self.opt['dataset'],
                                         isNormalize=True)
        dataset_test = Dataset_LDL(
            os.path.join(self.opt['data_path'], self.opt['dataset']), 'test',
            transforms_test)
        self.dataloader_test = DataLoader(dataset_test,
                                          batch_size=self.opt['batch_size'],
                                          num_workers=self.opt['num_workers'])

    def set_optimizer(self):
        self.optimizer = SGD(self.model.parameters(),
                             lr=self.opt['lr'],
                             momentum=self.opt['momentum'])

    def set_scheduler(self):
        self.scheduler = get_scheduler(self.opt, self.optimizer)

    def set_loss(self):
        raise NotImplementedError

    def train(self):
        self.model.to(self.opt['device'])
        self.loss.to(self.opt['device'])
        for epoch in range(self.opt['epochs']):
            epoch += self.start_epoch
            # train
            self.model.train()
            for k, v in self.meters_dict.items():
                self.meters_dict[k].reset()
            self.train_epoch(epoch)
            self.writer_train(epoch)

            # learning rate
            if self.scheduler:
                self.scheduler.step()
                self.writer.add_scalar(
                    "train/lr",
                    self.optimizer.state_dict()['param_groups'][0]['lr'],
                    global_step=epoch)
            # test
            self.model.eval()
            for k, v in self.meters_dict.items():
                self.meters_dict[k].reset()
            self.test_epoch(epoch)
            self.writer_test(epoch)

            checkpoint = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_steps': self.train_steps,
                'test_steps': self.test_steps,
            }
            self.save_checkpoint(checkpoint, 'epoch')


    def test(self):
        raise NotImplementedError

    def train_epoch(self, epoch):
        start_time = time.time()
        for i, (inputs, labels, cls) in enumerate(self.dataloader_train):
            inputs = inputs.to(self.opt['device'])
            labels = labels.to(self.opt['device'])
            cls = cls.to(self.opt['device'])

            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)

            self.meters_dict['loss'].update(loss.item(), inputs.size(0))
            self.meters_dict['ldl'].update(outputs, labels, inputs.size(0))

            prediction = torch.max(outputs, 1)[1]
            self.meters_dict['acc'].update(
                sum(prediction == cls) / inputs.size(0))

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
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)

            self.meters_dict['loss'].update(loss.item(), inputs.size(0))
            self.meters_dict['ldl'].update(outputs, labels, inputs.size(0))

            prediction = torch.max(outputs, 1)[1]
            self.meters_dict['acc'].update(
                sum(prediction == cls) / inputs.size(0))

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

    def meters(self):
        self.meters_dict['loss'] = AverageMeter('loss')
        self.meters_dict['ldl'] = LDL_measurement(self.opt['num_classes'])
        self.meters_dict['batch_time'] = AverageMeter('batch_time')
        self.meters_dict['acc'] = AverageMeter('acc')

    def writer_train(self, epoch):
        loss = self.meters_dict['loss'].average()
        ldl = self.meters_dict['ldl'].average()
        acc = self.meters_dict['acc'].average()

        self.logger.info(
            'Train: {epoch}\tLoss: {loss:.4f}\tkldiv: {kldiv:.4f}\tCosine: {Cosine:.4f}\t Cheb: {Cheb:.4f}\t intersection: {intersection:.4f}'
            .format(epoch=epoch,
                    loss=loss,
                    kldiv=ldl['klDiv'],
                    Cosine=ldl['cosine'],
                    Cheb=ldl['chebyshev'],
                    intersection=ldl['intersection']))
        self.logger.info("Acc:{}".format(acc))
        self.writer.add_scalar("train/acc", acc, epoch)

        self.writer.add_scalar("train/Loss", loss, epoch)
        self.writer.add_scalar("train/KLDiv", ldl['klDiv'], epoch)
        self.writer.add_scalar("train/Cosine", ldl['cosine'], epoch)
        self.writer.add_scalar("train/intersection", ldl['intersection'],
                               epoch)
        self.writer.add_scalar("train/chebyshev", ldl['chebyshev'], epoch)
        self.writer.add_scalar("train/clark", ldl['clark'], epoch)
        self.writer.add_scalar("train/canberra", ldl['canberra'], epoch)
        self.writer.add_scalar("train/squareChord", ldl['squareChord'], epoch)
        self.writer.add_scalar("train/sorensendist", ldl['sorensendist'],
                               epoch)
        write_result(self.opt['path'], epoch, acc, ldl, 'train')

    def writer_test(self, epoch):
        loss = self.meters_dict['loss'].average()
        ldl = self.meters_dict['ldl'].average()
        acc = self.meters_dict['acc'].average()

        self.logger.info(
            'Test: {epoch}\tLoss: {loss:.4f}\tkldiv: {kldiv:.4f}\tCosine: {Cosine:.4f}\tCheb: {Cheb:.4f}\t intersection: {intersection:.4f}'
            .format(epoch=epoch,
                    loss=loss,
                    kldiv=ldl['klDiv'],
                    Cosine=ldl['cosine'],
                    Cheb=ldl['chebyshev'],
                    intersection=ldl['intersection']))

        self.logger.info("Acc:{}".format(acc))
        self.writer.add_scalar("test/acc", acc, epoch)

        self.writer.add_scalar("test/Loss", loss, epoch)
        self.writer.add_scalar("test/KLDiv", ldl['klDiv'], epoch)
        self.writer.add_scalar("test/Cosine", ldl['cosine'], epoch)
        self.writer.add_scalar("test/intersection", ldl['intersection'], epoch)
        self.writer.add_scalar("test/chebyshev", ldl['chebyshev'], epoch)
        self.writer.add_scalar("test/clark", ldl['clark'], epoch)
        self.writer.add_scalar("test/canberra", ldl['canberra'], epoch)
        self.writer.add_scalar("test/squareChord", ldl['squareChord'], epoch)
        self.writer.add_scalar("test/sorensendist", ldl['sorensendist'], epoch)

        write_result(self.opt['path'], epoch, acc, ldl, 'test')

    def load_checkpoint(self):
        checkpoint = torch.load(self.opt["resume_path"])
        # epoch
        self.start_epoch = checkpoint['epoch']

        # model
        model_dict = self.model.state_dict()
        for k, v in checkpoint['state_dict'].items():
            if k in model_dict and v.shape == model_dict[k].shape:
                model_dict[k] = v
        self.model.load_state_dict(model_dict)

        # optimizer
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # others
        # self.save_mark = checkpoint['best_score']
        self.train_steps = checkpoint['train_steps']
        self.test_steps = checkpoint['test_steps']

    def save_checkpoint(self, checkpoint, save_name):
        save_path = os.path.join('logs', self.opt['path'], save_name + '.pth')
        torch.save(checkpoint, save_path, _use_new_zipfile_serialization=True)
