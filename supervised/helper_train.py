import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, CohenKappa, F1Score
from torch.nn.functional import softmax
from models.model import sleep_model
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from torch.cuda.amp import GradScaler


class distill_train(nn.Module):

    def __init__(self, EXPERIMENT_NAME, SAVE_PATH, train_loader, test_loader, wandb_logger, epoch_len, input_channels):
        super(distill_train, self).__init__()

        self.epoch_len = epoch_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = sleep_model(input_channels, self.epoch_len).to(self.device)
        self.exp_name = EXPERIMENT_NAME
        self.save_path = SAVE_PATH

        self.weight_decay = 3e-5
        self.lr = 0.001
        self.num_epochs = 60

        self.best_accuracy = 0
        self.loggr = wandb_logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.lr,
            betas=(0.9, 0.99),
            weight_decay=self.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(self.optimizer,
                                           mode="min",
                                           patience=5,
                                           factor=0.2
                                           )
        
    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader

    def training_step(self, batch, batch_idx):
        X, y = batch
        X, y = X.float().to(self.device), y.long().to(self.device)
        y = y[:, (self.epoch_len // 2)]
        outs = self.model(X)
        loss = self.criterion(outs, y)
        return loss, outs, y

    def testing_step(self, batch, batch_idx):
        X, y = batch
        X, y = X.float().to(self.device), y.long().to(self.device)
        y = y[:, (self.epoch_len // 2)]
        outs = self.model(X)
        loss = self.criterion(outs, y)
        return loss, outs, y  

    def on_train_epoch_end(self, outputs, epoch):
        epoch_preds = torch.vstack([x for x in outputs["preds"]])
        epoch_targets = torch.hstack([x for x in outputs["targets"]])
        epoch_loss = torch.hstack([torch.tensor(x) for x in outputs['loss']]).mean()

        self.scheduler.step(epoch_loss)
        
        class_preds = epoch_preds.argmax(dim=1)
        acc = Accuracy(task="multiclass", num_classes=5).to(self.device)(epoch_preds, epoch_targets)
        f1_score = F1Score(task="multiclass", num_classes=5, average="macro").to(self.device)(epoch_preds, epoch_targets)
        kappa = CohenKappa(task="multiclass", num_classes=5).to(self.device)(epoch_preds, epoch_targets)
        bal_acc = balanced_accuracy_score(epoch_targets.cpu().numpy(), class_preds.cpu().numpy())

        self.loggr.log({
                'F1 train': f1_score,
                'Kappa train': kappa,
                'Bal Acc train': bal_acc,
                'Acc train': acc,
                'Loss train': epoch_loss.item(),
                'Epoch': epoch,
                'LR': self.scheduler.optimizer.param_groups[0]["lr"],
            })

    def on_test_epoch_end(self, outputs, epoch):
        epoch_preds = torch.vstack([x for x in outputs["preds"]])
        epoch_targets = torch.hstack([x for x in outputs["targets"]])
        epoch_loss = torch.hstack([torch.tensor(x) for x in outputs['loss']]).mean()
        
        class_preds = epoch_preds.argmax(dim=1)
        acc = Accuracy(task="multiclass", num_classes=5).to(self.device)(epoch_preds, epoch_targets)
        f1_score = F1Score(task="multiclass", num_classes=5, average="macro").to(self.device)(epoch_preds, epoch_targets)
        kappa = CohenKappa(task="multiclass", num_classes=5).to(self.device)(epoch_preds, epoch_targets)
        bal_acc = balanced_accuracy_score(epoch_targets.cpu().numpy(), class_preds.cpu().numpy())

        self.loggr.log({
                'F1 test': f1_score,
                'Kappa test': kappa,
                'Bal Acc test': bal_acc,
                'Acc test': acc,
                'Loss test': epoch_loss.item(),
                'Epoch': epoch
            })
        return acc

    def fit(self):
        scaler = GradScaler()
        for epoch in range(self.num_epochs):
            
            print('='*50, end = '\n')
            print(f"Epoch: {epoch}")
            print('='*50, end = '\n')

            # Training Loop
            train_outputs = {"loss": [], "preds": [], "targets": []}
            self.model.train()

            for batch_idx, batch in tqdm(enumerate(self.train_loader), desc="Train", total=len(self.train_loader)):
                with torch.cuda.amp.autocast():
                    loss, outs, y = self.training_step(batch, batch_idx)
                    
                self.optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                train_outputs['loss'].append(loss.detach().item())
                train_outputs['preds'].append(softmax(outs.detach(), dim=1))
                train_outputs['targets'].append(y.detach())

            self.on_train_epoch_end(train_outputs, epoch)

            # Testing Loop
            test_outputs = {"loss": [], "preds": [], "targets": []}
            self.model.eval()
            with torch.no_grad():
                for batch_idx, batch in tqdm(enumerate(self.test_loader), desc="Test", total=len(self.test_loader)):
                    loss, outs, y = self.testing_step(batch, batch_idx)
                    
                    test_outputs['loss'].append(loss.item())
                    test_outputs['preds'].append(softmax(outs, dim=1))
                    test_outputs['targets'].append(y)

            acc = self.on_test_epoch_end(test_outputs, epoch)
            if self.best_accuracy < acc:
                checkpoint = {
                        'model': self.model.state_dict(),
                        'epoch': epoch,
                        'accuracy': acc,
                    }
                torch.save(
                    checkpoint,
                    os.path.join(self.save_path, self.exp_name + "_best.pt"),
                    )
                self.loggr.save(os.path.join(self.save_path, self.exp_name + "_best.pt"))
                print(f"Best weights saved with accuracy: {acc*100:.2f} at epoch: {epoch}")