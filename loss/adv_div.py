import torch.nn as nn
import torch

def Adv_hook(module, grad_in, grad_out):
    return ((grad_in[0] * (-1), grad_in[1]))


class AdvDivLoss(nn.Module):
    """
    Attention AdvDiverse Loss
    x : is the vector
    """
    def __init__(self, parts=4):
        super(AdvDivLoss, self).__init__()
        self.parts = parts
        
        self.fc_pre = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Flatten(1),
                                    nn.Linear(256, 128, bias=False))
        self.fc = nn.Sequential(nn.BatchNorm1d(128), nn.ReLU(),
                                nn.Linear(128, 128), nn.BatchNorm1d(128))
        self.fc_pre.register_backward_hook(Adv_hook)

    def forward(self, x):
        x = nn.functional.normalize(x)
        x = self.fc_pre(x)
        x = self.fc(x)
        x = nn.functional.normalize(x)
        out = 0
        num = int(x.size(0) / self.parts)
        for i in range(self.parts):
            for j in range(self.parts):
                if i < j:
                    out += ((x[i * num:(i + 1) * num, :] -
                             x[j * num:(j + 1) * num, :]).norm(
                                 dim=1, keepdim=True)).mean()
        return out * 2 / (self.parts * (self.parts - 1))


class Circular_structured(nn.Module):
    """
    Circular_structured
    x : is the vector
    """
    def __init__(self):
        super(Circular_structured, self).__init__()
        self.pi = torch.tensor(3.14159).to('cuda:0')
    def cal_pi(self, x):
        x = torch.where((x >= 0.5 * self.pi) & (x < 1.5 * self.pi), torch.ones_like(x), torch.zeros_like(x))
        return x
    def forward(self, x):
        N, C = x.shape
        j = torch.arange(1, C+1).repeat((N, 1)).to('cuda:0')
        thetaj = self.pi * (j * 2 - 1) / 8.0
        rj = torch.ones_like(x)

        theta_ji = thetaj * x

        r_ji = rj * x
        x_ji = r_ji * torch.cos(theta_ji)
        y_ji = r_ji * torch.sin(theta_ji)

        x_i = x_ji.sum(1)
        y_i = y_ji.sum(1)

        r_i = torch.sqrt(x_i ** 2 + y_i ** 2)
        theta_i = torch.arctan(y_i/x_i)

        p_i = torch.where((theta_i < 1.5 * self.pi) & (theta_i >= 0.5 * self.pi), torch.ones_like(theta_i), torch.zeros_like(theta_i))

        e_i = (p_i, theta_i, r_i)
        return e_i


class ProgressiveCircularLoss(nn.Module):
    """
    Circular_structured
    x : is the vector
    """
    def __init__(self, mu = 0.5):
        super(ProgressiveCircularLoss, self).__init__()
        self.mu = mu
        self.klloss = nn.KLDivLoss(reduction='batchmean')
        self.cs = Circular_structured()

    def forward(self, x, y):
        x_ = self.cs(x)
        y_ = self.cs(y)
        L_pc = ((y_[0] - x_[0]) ** 2 + (y_[1] - x_[1]) ** 2) * y_[2]
        l_pc = L_pc.mean()
        l_kl = self.klloss(x, y)
        return self.mu * l_pc + (1 - self.mu) * l_kl

if __name__ == "__main__":
    cs_loss = Circular_structured()
    pc_loss = ProgressiveCircularLoss()
    x = torch.tensor([[-0.2313, -0.1209,  0.1813,  0.0177,  0.2263,  0.3111, -0.2915, -0.2925],
                        [-0.2313, -0.1209,  0.1813,  0.0177,  0.2263,  0.3111, -0.2915, -0.2925]])
    y = torch.tensor([[-1.2313, -0.1209,  0.1813,  0.0377,  0.2263,  0.1111, -0.2915, -0.2125],
                        [-0.5313, -0.3209,  1.1813,  0.1177,  0.2263,  0.3111, -0.1915, -0.2925]])
    loss = cs_loss(x)
    print(loss)

    loss = pc_loss(x, y)
    print(loss)

