import torch
import torch.nn as nn
import torch.nn.functional as F


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


class LossMulti(nn.Module):
    def __init__(self, jaccard_weight=0.5, class_weights=None, num_classes=2, alpha=0.25, gamma=2.0):
        super(LossMulti, self).__init__()
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32))
            if torch.cuda.is_available():
                nll_weight = nll_weight.cuda()
        else:
            nll_weight = torch.tensor([1.0, 2.0])
            if torch.cuda.is_available():
                nll_weight = nll_weight.cuda()

        self.nll_loss = nn.NLLLoss2d(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = 0.3

    def focal_loss(self, outputs, targets):
        if outputs.size(1) == self.num_classes:
            probs = torch.exp(outputs)
        else:
            probs = F.softmax(outputs, dim=1)

        probs_pos = probs[:, 1]

        if len(targets.size()) == 3:
            targets = targets.unsqueeze(1)
        elif len(targets.size()) == 4 and targets.size(1) == 1:
            pass
        else:
            raise ValueError("Targets shape must be [batch, H, W] or [batch, 1, H, W]")

        targets = targets.float()

        bce_loss = -torch.log(probs_pos + 1e-8) * targets.squeeze(1)
        bce_loss = bce_loss.mean()

        pt = probs_pos * targets.squeeze(1) + (1 - probs_pos) * (1 - targets.squeeze(1))
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = (focal_weight * bce_loss).mean()

        return focal_loss

    def dice_loss(self, outputs, targets):
        if outputs.size(1) == self.num_classes:
            probs = torch.exp(outputs)
        else:
            probs = F.softmax(outputs, dim=1)

        probs_pos = probs[:, 1]

        if len(targets.size()) == 3:
            targets = targets.unsqueeze(1)
        elif len(targets.size()) == 4 and targets.size(1) == 1:
            pass
        else:
            raise ValueError("Targets shape must be [batch, H, W] or [batch, 1, H, W]")

        targets = targets.float()

        if len(probs_pos.size()) == 4:
            probs_pos = probs_pos.squeeze(1)

        intersection = (probs_pos * targets.squeeze(1)).sum(dim=(1, 2))
        union = probs_pos.sum(dim=(1, 2)) + targets.squeeze(1).sum(dim=(1, 2))

        dice = (2. * intersection + 1e-8) / (union + 1e-8)
        return 1 - dice.mean()

    def __call__(self, outputs, targets):
        nll_loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()
                union = jaccard_output.sum() + jaccard_target.sum()
                nll_loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight

        focal_loss = self.focal_loss(outputs, targets)

        dice_loss = self.dice_loss(outputs, targets)

        total_loss = nll_loss + focal_loss + self.dice_weight * dice_loss

        return total_loss