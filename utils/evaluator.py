import numpy as np
import torch
from torch.utils.data import DataLoader


def calc_cm_dice_scoress(preds, labels, classes, num_samples=None):

    cm_dice_scoress = {}

    if num_samples:
        samples = np.random.choice(len(preds), num_samples)
        preds, labels = preds[samples], labels[samples]

    for i, first_class in enumerate(classes):
        cm_dice_scoress[first_class] = {}
        labels_class = (labels == i).float()
        for j, second_class in enumerate(classes):
            preds_class = (preds == j).float()
            inter = torch.sum(torch.mul(labels_class, preds_class))
            union = torch.sum(labels_class) + torch.sum(preds_class) + 0.0001
            cm_dice_scoress[first_class][second_class] = (2 * torch.div(inter, union)).item()

    return cm_dice_scoress


def calc_class_dice_scores(preds, labels, classes, num_samples=None):
    class_dice_scores = {}

    if num_samples:
        samples = np.random.choice(len(preds), min(len(preds), num_samples), replace=False)
        preds, labels = preds[samples], labels[samples]

    for i, _class in enumerate(classes):
        preds_class = (preds == i).float()
        labels_class = (labels == i).float()
        inter = torch.sum(torch.mul(labels_class, preds_class))
        union = torch.sum(labels_class) + torch.sum(preds_class) + 0.0001
        class_dice_scores[_class] = (2 * torch.div(inter, union)).item()

    return class_dice_scores


def calc_validation_dice_score(model, dataset, classes, batch_size, device=0):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    volumes_dice_scores = {}

    with torch.no_grad():
        for vol_idx, (vol_id, volume, labels, _) in enumerate(dataloader):
            vol_id, volume, labels = vol_id[0], volume[0], labels[0]

            volume = volume if len(volume.shape) == 4 else volume[:, np.newaxis, :, :]
            volume, labels = volume.type(torch.FloatTensor), labels.type(torch.LongTensor)

            volume_prediction = []
            for i in range(0, len(volume), batch_size):
                batch = volume[i:i + batch_size]
                batch = batch.cuda(device) if torch.cuda.is_available() else batch
                out = model(batch)
                _, output = torch.max(out, dim=1)
                volume_prediction.append(output)
            volume_prediction = torch.cat(volume_prediction)

            volume_dice_scores = calc_class_dice_scores(volume_prediction, labels.cuda(device), classes)

            volumes_dice_scores[vol_id] = volume_dice_scores

            print('Mean dice score of all classes for volume "{}":'.format(vol_id),
                  np.mean(list(volume_dice_scores.values())))

    return volumes_dice_scores