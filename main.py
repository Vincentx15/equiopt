from models import CustomRCPS, EquiNetBinary
from loader import BatchDataset
from torch.utils.data import DataLoader

import numpy as np
import sys
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score


def learn(model,
          train_dataloader,
          test_dataloader,
          n_epochs=15,
          gpu_device=0,
          validate_each=50,
          save_path='toto.ckpt',
          early_stop_threshold=200):
    """
    Run the training
    :param n_epochs: the number of epochs
    :param data: the database for results
    :param gpu_device:
    :param validate_each: print results information each given step
    :param abort: If not None, abort after the given number of step (for testing purpose)
    :param weights: if some weights are given for resuming a training
    :param early_stop_threshold:
    :param blobber_params: useful for model optimization
    :param blob_metrics: Whether to use the usual random validation split or the CATH one along with blobs metrics
    :return:
    """
    np.random.seed(0)
    torch.manual_seed(0)

    gpu_device = torch.device(f'cuda:{gpu_device}' if torch.cuda.is_available() else 'cpu')
    model.to(gpu_device)
    optimizer = optim.Adam(model.parameters())

    # writer = SummaryWriter(logdir)
    best_loss = 0.
    epochs_from_best = 0

    import time
    time_passed = time.perf_counter()
    for epoch in range(n_epochs):
        passed = 0
        for i, (inputs, targets) in enumerate(train_dataloader):
            passed += 1
            inputs, targets = inputs.to(gpu_device), targets.to(gpu_device)
            model.train()
            out = model(inputs)
            loss = torch.nn.BCELoss()(out, targets)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            if not i % validate_each:
                print(
                    f'After {epoch} epochs, and {i}/{len(train_dataloader)} batches,'
                    f' we get a loss of {loss.item():.4} '
                    f' in a runtime of {time.perf_counter() - time_passed:.4}s'
                    f' and a best AuROC of {best_loss:.5}')

                all_preds = list()
                all_targets = list()
                model.eval()
                with torch.no_grad():
                    for i, (inputs, targets) in enumerate(test_dataloader):
                        inputs = inputs.to(gpu_device)
                        out = model(inputs)
                        all_preds.append(out.detach().cpu().numpy())
                        all_targets.append(targets.detach().cpu().numpy())

                        if i > 10:
                            break
                all_targets = np.concatenate(all_targets)
                all_preds = np.concatenate(all_preds)
                validation_loss = roc_auc_score(all_targets, all_preds)

                # Early stopping
                if early_stop_threshold is not None:
                    if validation_loss > best_loss:
                        best_loss = validation_loss
                        epochs_from_best = 0
                        model.to('cpu')
                        torch.save(model.state_dict(), save_path)
                        model.to(gpu_device)
                    else:
                        epochs_from_best += 1
                        if epochs_from_best > early_stop_threshold:
                            print('This model was early stopped')
                            return best_loss
                else:
                    model.to('cpu')
                    torch.save(model.state_dict(), save_path)
                    model.to(gpu_device)
        train_dataloader.dataset.on_epoch_end()
    return best_loss


if __name__ == '__main__':
    pass

    train_dataloader = BatchDataset(TF='CTCF', seq_len=1000, is_aug=False,
                                    batch_size=10).get_loader(num_workers=2)
    test_dataloader = BatchDataset(TF='CTCF', seq_len=1000, is_aug=False,
                                   batch_size=10, split='test').get_loader(num_workers=2)

    model = CustomRCPS()
    # for param in model.parameters():
    #     print(param.shape)
    learn(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
