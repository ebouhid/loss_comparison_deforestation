import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from segmentation_models_pytorch.utils.losses import BCELoss
import torch
import torchmetrics
import numpy as np
import hashlib
from focalloss import FocalLoss
from tverskyloss import BinaryTverskyLoss

class DeepLabV3Plus_FocalLoss_GammaDot5_4ch(pl.LightningModule):
    def __init__(self,
                 encoder_name='resnet101',
                 lr=1e-3,
                 encoder_weights='imagenet'):
        super().__init__()

        # Defining model
        self.model = smp.DeepLabV3Plus(in_channels=4,
                                 classes=1,
                                 activation=None,
                                 encoder_name=encoder_name,
                                 encoder_weights=encoder_weights)

        self.loss = FocalLoss(alpha=1, gamma=0.5, reduction='mean')
        self.lr = lr

        # Defining metrics
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.train_precision = torchmetrics.Precision(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')
        self.train_recall = torchmetrics.Recall(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')
        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.train_iou = torchmetrics.JaccardIndex(task='binary')
        self.val_iou = torchmetrics.JaccardIndex(task='binary')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(),
                                     lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=10,
                                                    gamma=0.9,
                                                    verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss and training metrics
        loss = self.loss(outputs, targets)
        train_accuracy = np.float64(self.train_accuracy(outputs, targets))
        train_precision = np.float64(self.train_precision(outputs, targets))
        train_recall = np.float64(self.train_recall(outputs, targets))
        train_f1 = np.float64(self.train_f1(outputs, targets))
        train_iou = np.float64(self.train_iou(outputs, targets))

        # Log metrics
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_accuracy', train_accuracy, on_epoch=True)
        self.log('train_precision', train_precision, on_epoch=True)
        self.log('train_recall', train_recall, on_epoch=True)
        self.log('train_f1', train_f1, on_epoch=True)
        self.log('train_iou', train_iou, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss and validation metrics
        loss = self.loss(outputs, targets)
        val_accuracy = np.float64(self.val_accuracy(outputs, targets))
        val_precision = np.float64(self.val_precision(outputs, targets))
        val_recall = np.float64(self.val_recall(outputs, targets))
        val_f1 = np.float64(self.val_f1(outputs, targets))
        val_iou = np.float64(self.val_iou(outputs, targets))

        # Log metrics
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_accuracy', val_accuracy, on_epoch=True)
        self.log('val_precision', val_precision, on_epoch=True)
        self.log('val_recall', val_recall, on_epoch=True)
        self.log('val_f1', val_f1, on_epoch=True)
        self.log('val_iou', val_iou, on_epoch=True)


class DeepLabV3Plus_FocalLoss_Gamma2_4ch(pl.LightningModule):
    def __init__(self,
                 encoder_name='resnet101',
                 lr=1e-3,
                 encoder_weights='imagenet'):
        super().__init__()

        # Defining model
        
        self.model = smp.DeepLabV3Plus(in_channels=4,
                                 classes=1,
                                 activation=None,
                                 encoder_name=encoder_name,
                                 encoder_weights=encoder_weights)

        self.loss = FocalLoss(alpha=1, gamma=2, reduction='mean')
        self.lr = lr

        # Defining metrics
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.train_precision = torchmetrics.Precision(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')
        self.train_recall = torchmetrics.Recall(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')
        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.train_iou = torchmetrics.JaccardIndex(task='binary')
        self.val_iou = torchmetrics.JaccardIndex(task='binary')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(),
                                     lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=10,
                                                    gamma=0.9,
                                                    verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss and training metrics
        loss = self.loss(outputs, targets)
        train_accuracy = np.float64(self.train_accuracy(outputs, targets))
        train_precision = np.float64(self.train_precision(outputs, targets))
        train_recall = np.float64(self.train_recall(outputs, targets))
        train_f1 = np.float64(self.train_f1(outputs, targets))
        train_iou = np.float64(self.train_iou(outputs, targets))

        # Log metrics
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_accuracy', train_accuracy, on_epoch=True)
        self.log('train_precision', train_precision, on_epoch=True)
        self.log('train_recall', train_recall, on_epoch=True)
        self.log('train_f1', train_f1, on_epoch=True)
        self.log('train_iou', train_iou, on_epoch=True)

        return loss
    
class DeepLabV3Plus_Baseline_4ch(pl.LightningModule):
    def __init__(self,
                 encoder_name='resnet101',
                 lr=1e-3,
                 encoder_weights='imagenet'):
        super().__init__()

        # Defining model
        self.model = smp.DeepLabV3Plus(in_channels=4,
                                 classes=1,
                                 activation=None,
                                 encoder_name=encoder_name,
                                 encoder_weights=encoder_weights)

        self.loss = torch.nn.BCEWithLogitsLoss()
        self.lr = lr

        # Defining metrics
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.train_precision = torchmetrics.Precision(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')
        self.train_recall = torchmetrics.Recall(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')
        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.train_iou = torchmetrics.JaccardIndex(task='binary')
        self.val_iou = torchmetrics.JaccardIndex(task='binary')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(),
                                     lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=10,
                                                    gamma=0.9,
                                                    verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss and training metrics
        loss = self.loss(outputs, targets)
        train_accuracy = np.float64(self.train_accuracy(outputs, targets))
        train_precision = np.float64(self.train_precision(outputs, targets))
        train_recall = np.float64(self.train_recall(outputs, targets))
        train_f1 = np.float64(self.train_f1(outputs, targets))
        train_iou = np.float64(self.train_iou(outputs, targets))

        # Log metrics
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_accuracy', train_accuracy, on_epoch=True)
        self.log('train_precision', train_precision, on_epoch=True)
        self.log('train_recall', train_recall, on_epoch=True)
        self.log('train_f1', train_f1, on_epoch=True)
        self.log('train_iou', train_iou, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss and validation metrics
        loss = self.loss(outputs, targets)
        val_accuracy = np.float64(self.val_accuracy(outputs, targets))
        val_precision = np.float64(self.val_precision(outputs, targets))
        val_recall = np.float64(self.val_recall(outputs, targets))
        val_f1 = np.float64(self.val_f1(outputs, targets))
        val_iou = np.float64(self.val_iou(outputs, targets))

        # Log metrics
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_accuracy', val_accuracy, on_epoch=True)
        self.log('val_precision', val_precision, on_epoch=True)
        self.log('val_recall', val_recall, on_epoch=True)
        self.log('val_f1', val_f1, on_epoch=True)
        self.log('val_iou', val_iou, on_epoch=True)

class DeepLabV3Plus_TverskyLoss_4ch(pl.LightningModule):
    def __init__(self,
                 encoder_name='resnet101',
                 lr=1e-3,
                 encoder_weights='imagenet'):
        super().__init__()

        # Defining model
        self.model = smp.DeepLabV3Plus(in_channels=4,
                                 classes=1,
                                 activation=None,
                                 encoder_name=encoder_name,
                                 encoder_weights=encoder_weights)

        self.loss = BinaryTverskyLoss(alpha=0.3, beta=0.7, smooth=1e-5)
        self.lr = lr

        # Defining metrics
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.train_precision = torchmetrics.Precision(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')
        self.train_recall = torchmetrics.Recall(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')
        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.train_iou = torchmetrics.JaccardIndex(task='binary')
        self.val_iou = torchmetrics.JaccardIndex(task='binary')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(),
                                     lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=10,
                                                    gamma=0.9,
                                                    verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss and training metrics
        loss = self.loss(outputs, targets)
        train_accuracy = np.float64(self.train_accuracy(outputs, targets))
        train_precision = np.float64(self.train_precision(outputs, targets))
        train_recall = np.float64(self.train_recall(outputs, targets))
        train_f1 = np.float64(self.train_f1(outputs, targets))
        train_iou = np.float64(self.train_iou(outputs, targets))

        # Log metrics
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_accuracy', train_accuracy, on_epoch=True)
        self.log('train_precision', train_precision, on_epoch=True)
        self.log('train_recall', train_recall, on_epoch=True)
        self.log('train_f1', train_f1, on_epoch=True)
        self.log('train_iou', train_iou, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss and validation metrics
        loss = self.loss(outputs, targets)
        val_accuracy = np.float64(self.val_accuracy(outputs, targets))
        val_precision = np.float64(self.val_precision(outputs, targets))
        val_recall = np.float64(self.val_recall(outputs, targets))
        val_f1 = np.float64(self.val_f1(outputs, targets))
        val_iou = np.float64(self.val_iou(outputs, targets))

        # Log metrics
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_accuracy', val_accuracy, on_epoch=True)
        self.log('val_precision', val_precision, on_epoch=True)
        self.log('val_recall', val_recall, on_epoch=True)
        self.log('val_f1', val_f1, on_epoch=True)
        self.log('val_iou', val_iou, on_epoch=True)  

class DeepLabV3Plus_FocalLoss_GammaDot5_8ch(pl.LightningModule):
    def __init__(self,
                 encoder_name='resnet101',
                 lr=1e-3,
                 encoder_weights='imagenet'):
        super().__init__()

        # Defining model
        self.model = smp.DeepLabV3Plus(in_channels=8,
                                 classes=1,
                                 activation=None,
                                 encoder_name=encoder_name,
                                 encoder_weights=encoder_weights)

        self.loss = FocalLoss(alpha=1, gamma=0.5, reduction='mean')
        self.lr = lr

        # Defining metrics
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.train_precision = torchmetrics.Precision(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')
        self.train_recall = torchmetrics.Recall(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')
        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.train_iou = torchmetrics.JaccardIndex(task='binary')
        self.val_iou = torchmetrics.JaccardIndex(task='binary')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(),
                                     lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=10,
                                                    gamma=0.9,
                                                    verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss and training metrics
        loss = self.loss(outputs, targets)
        train_accuracy = np.float64(self.train_accuracy(outputs, targets))
        train_precision = np.float64(self.train_precision(outputs, targets))
        train_recall = np.float64(self.train_recall(outputs, targets))
        train_f1 = np.float64(self.train_f1(outputs, targets))
        train_iou = np.float64(self.train_iou(outputs, targets))

        # Log metrics
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_accuracy', train_accuracy, on_epoch=True)
        self.log('train_precision', train_precision, on_epoch=True)
        self.log('train_recall', train_recall, on_epoch=True)
        self.log('train_f1', train_f1, on_epoch=True)
        self.log('train_iou', train_iou, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss and validation metrics
        loss = self.loss(outputs, targets)
        val_accuracy = np.float64(self.val_accuracy(outputs, targets))
        val_precision = np.float64(self.val_precision(outputs, targets))
        val_recall = np.float64(self.val_recall(outputs, targets))
        val_f1 = np.float64(self.val_f1(outputs, targets))
        val_iou = np.float64(self.val_iou(outputs, targets))

        # Log metrics
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_accuracy', val_accuracy, on_epoch=True)
        self.log('val_precision', val_precision, on_epoch=True)
        self.log('val_recall', val_recall, on_epoch=True)
        self.log('val_f1', val_f1, on_epoch=True)
        self.log('val_iou', val_iou, on_epoch=True)


class DeepLabV3Plus_FocalLoss_Gamma2_8ch(pl.LightningModule):
    def __init__(self,
                 encoder_name='resnet101',
                 lr=1e-3,
                 encoder_weights='imagenet'):
        super().__init__()

        # Defining model
        
        self.model = smp.DeepLabV3Plus(in_channels=8,
                                 classes=1,
                                 activation=None,
                                 encoder_name=encoder_name,
                                 encoder_weights=encoder_weights)

        self.loss = FocalLoss(alpha=1, gamma=2, reduction='mean')
        self.lr = lr

        # Defining metrics
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.train_precision = torchmetrics.Precision(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')
        self.train_recall = torchmetrics.Recall(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')
        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.train_iou = torchmetrics.JaccardIndex(task='binary')
        self.val_iou = torchmetrics.JaccardIndex(task='binary')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(),
                                     lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=10,
                                                    gamma=0.9,
                                                    verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss and training metrics
        loss = self.loss(outputs, targets)
        train_accuracy = np.float64(self.train_accuracy(outputs, targets))
        train_precision = np.float64(self.train_precision(outputs, targets))
        train_recall = np.float64(self.train_recall(outputs, targets))
        train_f1 = np.float64(self.train_f1(outputs, targets))
        train_iou = np.float64(self.train_iou(outputs, targets))

        # Log metrics
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_accuracy', train_accuracy, on_epoch=True)
        self.log('train_precision', train_precision, on_epoch=True)
        self.log('train_recall', train_recall, on_epoch=True)
        self.log('train_f1', train_f1, on_epoch=True)
        self.log('train_iou', train_iou, on_epoch=True)

        return loss
    
class DeepLabV3Plus_Baseline_8ch(pl.LightningModule):
    def __init__(self,
                 encoder_name='resnet101',
                 lr=1e-3,
                 encoder_weights='imagenet'):
        super().__init__()

        # Defining model
        self.model = smp.DeepLabV3Plus(in_channels=8,
                                 classes=1,
                                 activation=None,
                                 encoder_name=encoder_name,
                                 encoder_weights=encoder_weights)

        self.loss = torch.nn.BCEWithLogitsLoss()
        self.lr = lr

        # Defining metrics
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.train_precision = torchmetrics.Precision(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')
        self.train_recall = torchmetrics.Recall(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')
        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.train_iou = torchmetrics.JaccardIndex(task='binary')
        self.val_iou = torchmetrics.JaccardIndex(task='binary')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(),
                                     lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=10,
                                                    gamma=0.9,
                                                    verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss and training metrics
        loss = self.loss(outputs, targets)
        train_accuracy = np.float64(self.train_accuracy(outputs, targets))
        train_precision = np.float64(self.train_precision(outputs, targets))
        train_recall = np.float64(self.train_recall(outputs, targets))
        train_f1 = np.float64(self.train_f1(outputs, targets))
        train_iou = np.float64(self.train_iou(outputs, targets))

        # Log metrics
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_accuracy', train_accuracy, on_epoch=True)
        self.log('train_precision', train_precision, on_epoch=True)
        self.log('train_recall', train_recall, on_epoch=True)
        self.log('train_f1', train_f1, on_epoch=True)
        self.log('train_iou', train_iou, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss and validation metrics
        loss = self.loss(outputs, targets)
        val_accuracy = np.float64(self.val_accuracy(outputs, targets))
        val_precision = np.float64(self.val_precision(outputs, targets))
        val_recall = np.float64(self.val_recall(outputs, targets))
        val_f1 = np.float64(self.val_f1(outputs, targets))
        val_iou = np.float64(self.val_iou(outputs, targets))

        # Log metrics
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_accuracy', val_accuracy, on_epoch=True)
        self.log('val_precision', val_precision, on_epoch=True)
        self.log('val_recall', val_recall, on_epoch=True)
        self.log('val_f1', val_f1, on_epoch=True)
        self.log('val_iou', val_iou, on_epoch=True)

class DeepLabV3Plus_TverskyLoss_8ch(pl.LightningModule):
    def __init__(self,
                 encoder_name='resnet101',
                 lr=1e-3,
                 encoder_weights='imagenet'):
        super().__init__()

        # Defining model
        self.model = smp.DeepLabV3Plus(in_channels=8,
                                 classes=1,
                                 activation=None,
                                 encoder_name=encoder_name,
                                 encoder_weights=encoder_weights)

        self.loss = BinaryTverskyLoss(alpha=0.3, beta=0.7, smooth=1e-5)
        self.lr = lr

        # Defining metrics
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.train_precision = torchmetrics.Precision(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')
        self.train_recall = torchmetrics.Recall(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')
        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.train_iou = torchmetrics.JaccardIndex(task='binary')
        self.val_iou = torchmetrics.JaccardIndex(task='binary')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(),
                                     lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=10,
                                                    gamma=0.9,
                                                    verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss and training metrics
        loss = self.loss(outputs, targets)
        train_accuracy = np.float64(self.train_accuracy(outputs, targets))
        train_precision = np.float64(self.train_precision(outputs, targets))
        train_recall = np.float64(self.train_recall(outputs, targets))
        train_f1 = np.float64(self.train_f1(outputs, targets))
        train_iou = np.float64(self.train_iou(outputs, targets))

        # Log metrics
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_accuracy', train_accuracy, on_epoch=True)
        self.log('train_precision', train_precision, on_epoch=True)
        self.log('train_recall', train_recall, on_epoch=True)
        self.log('train_f1', train_f1, on_epoch=True)
        self.log('train_iou', train_iou, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss and validation metrics
        loss = self.loss(outputs, targets)
        val_accuracy = np.float64(self.val_accuracy(outputs, targets))
        val_precision = np.float64(self.val_precision(outputs, targets))
        val_recall = np.float64(self.val_recall(outputs, targets))
        val_f1 = np.float64(self.val_f1(outputs, targets))
        val_iou = np.float64(self.val_iou(outputs, targets))

        # Log metrics
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_accuracy', val_accuracy, on_epoch=True)
        self.log('val_precision', val_precision, on_epoch=True)
        self.log('val_recall', val_recall, on_epoch=True)
        self.log('val_f1', val_f1, on_epoch=True)
        self.log('val_iou', val_iou, on_epoch=True)