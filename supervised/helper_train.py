import os
import time, math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, CohenKappa, F1Score
from torch.nn.functional import softmax
from models.model import sleep_model, ft_loss
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.model_selection import KFold
from utils.dataloader import TuneDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler


class sleep_pretrain(nn.Module):

    def __init__(self, config, name, dataloader, test_subjects, wandb_logger):
        super(sleep_pretrain, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = sleep_model(config).to(self.device)
        self.config = config
        self.weight_decay = 3e-5
        self.name = name
        self.dataloader = dataloader
        self.loggr = wandb_logger
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(self.optimizer,
                                           mode="min",
                                           patience=5,
                                           factor=0.2)
        self.epochs = self.config.num_epoch

        self.max_f1 = 0
        self.max_kappa = 0
        self.max_bal_acc = 0
        self.max_acc = 0

        self.test_subjects = test_subjects

    def training_step(self, batch, batch_idx):
        X, y = batch
        X, y = X.to(self.device).float(), y.to(self.device).long()
        loss = self.model(X, y)
        return loss

    def training_epoch_end(self, outputs):
        epoch_loss = torch.hstack([torch.tensor(x) for x in outputs["loss"]]).mean()
        self.loggr.log({
            "Epoch Loss": epoch_loss,
            "LR": self.scheduler.optimizer.param_groups[0]["lr"],
            "Epoch": self.current_epoch,
        })
        self.scheduler.step(epoch_loss)
        return epoch_loss

    def on_epoch_end(self):
        checkpoint = {"eeg_model_state_dict": self.model.eeg_encoder.state_dict()}
        torch.save(checkpoint, os.path.join(self.config.exp_path, self.name + ".pt"))
        return None

    def ft_fun(self, test_subjects_train, test_subjects_test):

        train_dl = DataLoader(
            TuneDataset(test_subjects_train),
            batch_size=self.config.eval_batch_size,
            shuffle=True,
        )
        test_dl = DataLoader(
            TuneDataset(test_subjects_test),
            batch_size=self.config.eval_batch_size,
            shuffle=False,
        )

        sleep_eval = sleep_ft(
            self.config.exp_path + "/" + self.name + ".pt",
            self.config,
            train_dl,
            test_dl,
            self.loggr,
        )
        f1, kappa, bal_acc, acc = sleep_eval.fit()

        return f1, kappa, bal_acc, acc

    def do_kfold(self):

        kfold = KFold(n_splits=self.config.splits,
                      shuffle=True,
                      random_state=1234)

        k_acc, k_f1, k_kappa, k_bal_acc = 0, 0, 0, 0
        start = time.time()
        
        i = 0
        for train_idx, test_idx in kfold.split(self.test_subjects):

            test_subjects_train = [self.test_subjects[i] for i in train_idx]
            test_subjects_test = [self.test_subjects[i] for i in test_idx]
            test_subjects_train = [
                rec for sub in test_subjects_train for rec in sub
            ]
            test_subjects_test = [
                rec for sub in test_subjects_test for rec in sub
            ]
            
            i+=1
            print(f'Fold: {i}')
            
            f1, kappa, bal_acc, acc = self.ft_fun(test_subjects_train, test_subjects_test)
            k_f1 += f1
            k_kappa += kappa
            k_bal_acc += bal_acc
            k_acc += acc
      
        pit = time.time() - start
        print(f"Took {int(pit // 60)} min:{int(pit % 60)} secs")

        return (
            k_f1 / self.config.splits,
            k_kappa / self.config.splits,
            k_bal_acc / self.config.splits,
            k_acc / self.config.splits,
        )

    def fit(self):

        epoch_loss = 0
        scaler = GradScaler()
        
        for epoch in range(1, self.epochs+1):
            self.current_epoch = epoch
            outputs = { "loss": [],}

            self.model.train()
            for batch_idx, batch in tqdm(enumerate(self.dataloader), desc="Pretraining", total=len(self.dataloader)):

                with torch.cuda.amp.autocast():
                    loss = self.training_step(batch, batch_idx)

                self.optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                outputs["loss"].append(loss.detach().item())

            epoch_loss = self.training_epoch_end(outputs)
            
            print('='*50, end = '\n')
            print(f"Epoch: {epoch}, Loss: {epoch_loss}")
            print('='*50, end = '\n')
           
            self.on_epoch_end()

            # evaluation step
            if (epoch == 1) or (epoch % 10 == 0):
                f1, kappa, bal_acc, acc = self.do_kfold()
                self.loggr.log({
                    'F1': f1,
                    'Kappa': kappa,
                    'Bal Acc': bal_acc,
                    'Acc': acc,
                    'Epoch': epoch
                })
                print(f'F1: {f1} Kappa: {kappa} B.Acc: {bal_acc} Acc: {acc}')

                checkpoint_epoch = {
                    "eeg_model_state_dict": self.model.eeg_encoder.state_dict(),
                    "pretrain_epoch": epoch,
                    "f1": f1,
                }
                torch.save(
                    checkpoint_epoch,
                    os.path.join(self.config.exp_path, self.name + f"__{epoch}.pt"),
                )
                self.loggr.save(
                        os.path.join(self.config.exp_path, self.name + f"__{epoch}.pt"))

                if self.max_f1 < f1:
                    checkpoint = {
                        'eeg_model_state_dict': self.model.eeg_encoder.state_dict(),
                        'best_pretrain_epoch': epoch,
                        'f1': f1
                    }
                    torch.save(
                        checkpoint,
                        os.path.join(self.config.exp_path,
                                     self.name + "_best.pt"),
                    )
                    self.loggr.save(
                        os.path.join(self.config.exp_path, self.name + f'_best.pt'))

                    full_checkpoint = {
                        "model_state_dict": self.model.state_dict(),
                        "epoch": self.current_epoch,
                    }
                    torch.save(
                        full_checkpoint,
                        os.path.join(self.config.exp_path, self.name + "_full_best.pt"),
                    )
                    self.loggr.save(
                        os.path.join(self.config.exp_path, self.name + "_full_best.pt"))
                    self.max_f1 = f1

                    

class sleep_ft(nn.Module):

    def __init__(self, checkpoint_path, config, train_dl, valid_dl, wandb_logger):
        super(sleep_ft, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ft_loss(checkpoint_path, config, self.device).to(self.device)
        self.config = config
        self.weight_decay = 3e-5
        self.loggr = wandb_logger
        self.criterion = nn.CrossEntropyLoss()
        self.train_ft_dl = train_dl
        self.valid_ft_dl = valid_dl
        self.eval_es = self.config.eval_early_stopping

        self.best_loss = torch.tensor(math.inf).to(self.device)
        self.counter = torch.tensor(0).to(self.device)
        self.max_f1 = torch.tensor(0).to(self.device)
        self.max_acc = torch.tensor(0).to(self.device)
        self.max_bal_acc = torch.tensor(0)
        self.max_kappa = torch.tensor(0).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.weight_decay,
        )
        self.ft_epoch = self.config.num_ft_epoch

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.valid_dl

    def training_step(self, batch, batch_idx):
        data, y = batch
        data, y = data.float().to(self.device), y.long().to(self.device)
        outs = self.model(data)
        loss = self.criterion(outs, y)
        return loss

    def validation_step(self, batch, batch_idx):
        data, y = batch
        data, y = data.float().to(self.device), y.to(self.device)
        outs = self.model(data)
        loss = self.criterion(outs, y)
 
        return {
            "loss": loss.detach(),
            "preds": softmax(outs.detach(), dim=1),
            "targets": y.detach()
        }   

    def validation_epoch_end(self, outputs):

        epoch_preds = torch.vstack([x for x in outputs["preds"]])
        epoch_targets = torch.hstack([x for x in outputs["targets"]])
        epoch_loss = torch.hstack([torch.tensor(x) for x in outputs['loss']]).mean()
        
        class_preds = epoch_preds.argmax(dim=1)
        acc = Accuracy(task="multiclass", num_classes=5).to(self.device)(epoch_preds, epoch_targets)
        f1_score = F1Score(task="multiclass", num_classes=5, average="macro").to(self.device)(epoch_preds, epoch_targets)
        kappa = CohenKappa(task="multiclass", num_classes=5).to(self.device)(epoch_preds, epoch_targets)
        bal_acc = balanced_accuracy_score(epoch_targets.cpu().numpy(), class_preds.cpu().numpy())

        if f1_score > self.max_f1:
            # self.loggr.log({'Pretrain Epoch' : self.loggr.plot.confusion_matrix(probs=None,title=f'Pretrain Epoch :{self.pret_epoch+1}',
            #            y_true= epoch_targets.cpu().numpy(), preds= class_preds.numpy(),
            #            class_names= ['Wake', 'N1', 'N2', 'N3', 'REM'])})
            self.max_f1 = f1_score
            self.max_kappa = kappa
            self.max_bal_acc = bal_acc
            self.max_acc = acc

        return epoch_loss

    def on_train_end(self):
        return self.max_f1, self.max_kappa, self.max_bal_acc, self.max_acc

    def fit(self):

        for ep in tqdm(range(self.ft_epoch), desc="Linear Evaluation"):

            # Training Loop
            self.model.train()
            ft_outputs = {"loss": [], "acc": [], "preds": [], "targets": []}

            for batch_idx, batch in enumerate(self.train_ft_dl):
                loss = self.training_step(batch, batch_idx)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Validation Loop
            self.model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.valid_ft_dl):
                    dct = self.validation_step(batch, batch_idx)
                    loss, preds, targets = (
                        dct["loss"],
                        dct["preds"],
                        dct["targets"],
                    )
                    ft_outputs["loss"].append(loss.item())
                    ft_outputs["preds"].append(preds)
                    ft_outputs["targets"].append(targets)

            val_loss = self.validation_epoch_end(ft_outputs)

            if val_loss + 0.001 < self.best_loss:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1

            if self.counter == self.eval_es:
                print(f'Early stopped at {ep} epoch')
                break

        return self.on_train_end()