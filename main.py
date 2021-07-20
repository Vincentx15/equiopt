from models import CustomRCPS, EquiNetBinary, PosthocModel, AverageModel, RegBinary
from loader import BatchDataset
from torch.utils.data import DataLoader

import numpy as np
import sys
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score


def test(model, test_dataloader, gpu_device='cpu', max_validation_steps=None):
    all_preds = list()
    all_targets = list()
    model.eval()
    model.to(gpu_device)
    with torch.no_grad():
        for j, (inputs, targets) in enumerate(test_dataloader):
            inputs = inputs.to(gpu_device)
            out = model(inputs)
            all_preds.append(out.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

            if max_validation_steps is not None and j > max_validation_steps:
                break
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    validation_loss = roc_auc_score(all_targets, all_preds)
    return validation_loss


def learn(model,
          train_dataloader,
          test_dataloader,
          n_epochs=15,
          gpu_device=0,
          validate_each=50,
          save_path='toto.ckpt',
          early_stop_threshold=200,
          max_validation_steps=None):
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
            if passed >= 160 * 50:
                return best_loss
            inputs, targets = inputs.to(gpu_device), targets.to(gpu_device)
            model.train()
            out = model(inputs)
            loss = torch.nn.BCELoss()(out, targets)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            if not i % validate_each:
                validation_loss = test(model=model, test_dataloader=test_dataloader,
                                       gpu_device=gpu_device, max_validation_steps=max_validation_steps)

                print(f'{epoch} epochs {i}/{len(train_dataloader)} batches;'
                      f' loss: {loss.item():.4} '
                      f' in {time.perf_counter() - time_passed:.4}s;'
                      f' current AuROC: {validation_loss:.5}'
                      f' best: {best_loss:.5}')

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
                            model.to('cpu')
                            return best_loss
                else:
                    model.to('cpu')
                    torch.save(model.state_dict(), save_path)
                    model.to(gpu_device)
        train_dataloader.dataset.on_epoch_end()
    model.to('cpu')
    return best_loss


if __name__ == '__main__':
    pass

    # Global options to be argparsed in the end
    num_workers = 20
    TF = 'CTCF'
    max_validation_steps = 20
    early_stop_threshold = 200

    # For TF = MAX expected performance is around 0.98

    # Get the data
    train_dataloader = BatchDataset(TF=TF, seq_len=1000, is_aug=False,
                                    batch_size=100).get_loader(num_workers=num_workers)
    test_dataloader = BatchDataset(TF=TF, seq_len=1000, is_aug=False,
                                   batch_size=100, split='test').get_loader(num_workers=num_workers)

    # Define an equivariant model
    model1 = CustomRCPS(seed=0)
    learn(model=model1, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
          max_validation_steps=max_validation_steps, early_stop_threshold=early_stop_threshold)

    train_dataloader.dataset.reinitialize_seed()
    test_dataloader.dataset.reinitialize_seed()
    # Define a second equivariant models and train it separately. Then average the two outputs
    model2 = CustomRCPS(seed=1)
    learn(model=model2, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
          max_validation_steps=max_validation_steps, early_stop_threshold=early_stop_threshold)
    equi_post_hoc = AverageModel(model1=model1, model2=model2)

    train_dataloader.dataset.reinitialize_seed()
    test_dataloader.dataset.reinitialize_seed()
    # Define a normal model, and post-hoc it
    model_3 = RegBinary(seed=0)
    learn(model=model_3, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
          max_validation_steps=max_validation_steps, early_stop_threshold=early_stop_threshold)
    reg_post_hoc = PosthocModel(model_3)

    # Now let us compare the performance of these models.
    equi_validation = test(model=model1, test_dataloader=test_dataloader)
    equi_posthoc_validation = test(model=equi_post_hoc, test_dataloader=test_dataloader)
    reg_posthoc_validation = test(model=reg_post_hoc, test_dataloader=test_dataloader)
    print(equi_validation, equi_posthoc_validation, reg_posthoc_validation)
    # equi_validation : 0.9354  equi_posthoc_validation : 0.9379 reg_posthoc_validation : 0.9309
