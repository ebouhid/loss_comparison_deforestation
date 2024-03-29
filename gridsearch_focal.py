import torch
from data.dataset import XinguDataset
import models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import mlflow
import albumentations as A
from focalloss import FocalLoss
from itertools import product

# Set experiment name
INFO = 'test'
mlflow.set_experiment(INFO)
# Instantiating logger
mlflow.pytorch.autolog()

# Set hyperparameters
MODEL_NAME = 'DeepLabV3Plus'
BATCH_SIZE = 32
NUM_EPOCHS = 100
PATCH_SIZE = 256
STRIDE_SIZE = 64
NUM_CLASSES = 1
DATASET_DIR = './data/scenes_allbands_ndvi'
GT_DIR = './data/truth_masks'
COMPOSITION = [4, 3, 1, 6]
compname = ''.join([str(i) for i in COMPOSITION]) if COMPOSITION != range(1, 9) else "All+NDVI"

# Set regions
train_regions = [2, 4, 6, 7, 8, 9, 10]  # Do not use region 5 anywhere
test_regions = [1, 3]

alphas = [0.25, 0.5, 0.75]
gammas = [0.1, 0.25, 0.5, 1., 2., 5.]

for alpha, gamma in product(alphas, gammas):
    loss = FocalLoss(alpha=alpha, gamma=gamma, debug=True)

    model = models.DeforestationDetectionModel(in_channels=len(COMPOSITION), composition_name=compname, lr=1e-5, loss=loss, debug=True)

    aug = A.Sequential([
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
        ], p=0.8),
        A.Normalize(mean=(0.485, 0.456, 0.406, 0.5), std=(0.229, 0.224, 0.225, 0.5), max_pixel_value=1.0)])

    # Instantiating datasets
    train_ds = XinguDataset(DATASET_DIR,
                            GT_DIR,
                            COMPOSITION,
                            train_regions,
                            patch_size=PATCH_SIZE,
                            stride_size=STRIDE_SIZE,
                            reflect_pad=True,
                            transforms=aug)
    test_ds = XinguDataset(DATASET_DIR,
                            GT_DIR,
                            COMPOSITION,
                            test_regions,
                            patch_size=PATCH_SIZE,
                            stride_size=PATCH_SIZE,
                            reflect_pad=True,
                            transforms=False)

    # Instantiating dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_ds,
                                                batch_size=BATCH_SIZE,
                                                shuffle=False,
                                                num_workers=8)


    mlflow.log_params({
        'model_name': MODEL_NAME,
        'loss': model.loss,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'patch_size': PATCH_SIZE,
        'stride_size': STRIDE_SIZE,
        'num_classes': NUM_CLASSES,
        'dataset_dir': DATASET_DIR,
        'gt_dir': GT_DIR,
        'composition': compname,
        'train_regions': train_regions,
        'test_regions': test_regions,
        'train_size': len(train_ds),
        'test_size': len(test_ds),
        'alpha': alpha,
        'gamma': gamma
    })

    # Instantiating checkpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath='./models/', filename=f'{INFO}-{MODEL_NAME}-{compname}', monitor='val_iou', save_top_k=1, mode='max', verbose=True)

    # Instantiating trainer
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS,
                            callbacks=[checkpoint_callback], accelerator="gpu", devices=-1)

    # Training
    trainer.fit(model, train_loader, test_loader)

    mlflow.end_run()

# End of gridsearch
print(f"{10*'#'}Gridsearch finished!{10*'#'}")
