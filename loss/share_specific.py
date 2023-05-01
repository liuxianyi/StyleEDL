# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedAndSpecificLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 3)
        self.softmax = nn.Softmax(dim=-1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, shared1, specific1, shared2, specific2, shared3,
                specific3, label):

        orth_1 = torch.bmm(shared1.unsqueeze(1),
                           specific1.unsqueeze(1).transpose(1, 2))
        orth_1 = torch.norm(orth_1)

        orth_2 = torch.bmm(shared2.unsqueeze(1),
                           specific2.unsqueeze(1).transpose(1, 2))
        orth_2 = torch.norm(orth_2)

        orth_3 = torch.bmm(shared3.unsqueeze(1),
                           specific3.unsqueeze(1).transpose(1, 2))
        orth_3 = torch.norm(orth_3)

        shared = torch.cat([shared1, shared2, shared3], dim=0)

        out = self.fc1(shared)
        out = self.fc2(out)
        out = self.softmax(out)
        cls = self.loss(out, label.long())

        return orth_1 + orth_2 + orth_3 + cls
