import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torch
import torchmetrics
import numpy as np
import hashlib
from focalloss import FocalLoss
from tverskyloss import BinaryTverskyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

class DeforestationDetectionModel(pl.LightningModule):
    def __init__(self, in_channels, encoder_name='resnet101', lr=1e-3, encoder_weights='imagenet'):
        super().__init__()

        # Defining model
        self.model = smp.DeepLabV3Plus(in_channels=in_channels,
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, patience=5, factor=0.9, mode='min', verbose=True),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1,
            'strict': True,
            'name': 'lr_scheduler'
        }
        
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

        return val_f1
    
    def on_save_checkpoint(self, checkpoint):
        return super().on_save_checkpoint(checkpoint) # For now
