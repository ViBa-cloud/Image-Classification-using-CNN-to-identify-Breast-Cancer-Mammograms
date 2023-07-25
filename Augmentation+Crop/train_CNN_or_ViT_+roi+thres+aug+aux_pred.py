import gc
import os

# import cv2
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from PIL import Image
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from timm import create_model, list_models
from timm.data import create_transform
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

import random

import torchvision

MAX_EVAL_BATCHES = 400
PREDICT_MAX_BATCHES = 1e9
N_FOLDS = 5
FOLDS = np.array(os.environ.get('FOLDS', '0,1,2,3,4').split(',')).astype(int)

#!!!!!!!!!!!!!!!!!!!! change follow until MODEL_PREFIX !!!!!!!!!!!!!!!!!!!!!!!!#
MODEL_NAME = 'deit3_small_patch16_384_in21ft1k'

MODEL_SIZE_X = 384
MODEL_SIZE_Y = 384

MODEL_MEAN = 0.2179
MODEL_STD = 0.0529

MODEL_PREFIX = "vit_s_8e_4lr_ser_norm"

# seresnext50_32x4d
# x,y = (1024,512)
# mean=0.2179, std=0.0529

# deit3_base_patch16_384
# deit3_small_patch16_384_in21ft1k
#  'mean': (0.485, 0.456, 0.406),
#  'std': (0.229, 0.224, 0.225),

# eva02_large_patch14_448.mim_m38m_ft_in22k_in1k
# mean=(0.48145466, 0.4578275, 0.40821073)
# std=(0.26862954, 0.26130258, 0.27577711)


CATEGORY_AUX_TARGETS = ['site_id', 'laterality', 'view', 'implant', 'biopsy', 'invasive', 'BIRADS', 'density', 'difficult_negative_case', 'machine_id', 'age']
TARGET = 'cancer'
ALL_FEAT = [TARGET] + CATEGORY_AUX_TARGETS

print('Running locally')
print('Model : ', MODEL_NAME)
print('no freeze 8 epoch 4 lr sernext norm')

#TRAIN_IMAGES_PATH = '/scratch/eecs545w23_class_root/eecs545w23_class/yngmkim/png_bcd_roi_1024x/train_images'
TRAIN_IMAGES_PATH = '/scratch/eecs545w23_class_root/eecs545w23_class/yngmkim/png_bcd_roi_1024x_split/train_images'
MODELS_PATH = '/scratch/eecs545w23_class_root/eecs545w23_class/yngmkim/vit_frez_model'

print('Model path: ', MODELS_PATH)
print('Model prefix: ', MODEL_PREFIX)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#WANDB_SWEEP = False
#TRAIN = True
#CV = True

# set the seed
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)

RUN_NAME = "_split_0418_2"

# parameters adapt from others work
class Config:
    # These are optimal parameters collected from https://wandb.ai/vslaykovsky/rsna-breast-cancer-sweeps/sweeps/k281hlr9?workspace=user-vslaykovsky
    ONE_CYCLE = True
    ONE_CYCLE_PCT_START = 0.1
    ADAMW = False
    ADAMW_DECAY = 0.024
    # ONE_CYCLE_MAX_LR = float(os.environ.get('LR', '0.0008'))
    # ONE_CYCLE_MAX_LR = float(os.environ.get('LR', '0.0004'))
    ONE_CYCLE_MAX_LR = float(os.environ.get('LR', '0.0004'))
    EPOCHS = int(os.environ.get('EPOCHS', 8))
    MODEL_TYPE = os.environ.get('MODEL', MODEL_NAME)
    DROPOUT = float(os.environ.get('DROPOUT', 0.1))
    AUG = os.environ.get('AUG', 'true').lower() == 'true'
    AUX_LOSS_WEIGHT = 94
    POSITIVE_TARGET_WEIGHT=20 # make a reason why weight 20?
    # BATCH_SIZE = 32
    BATCH_SIZE = 16
    AUTO_AUG_M = 10
    AUTO_AUG_N = 2


class BreastCancerDataSet(torch.utils.data.Dataset):
    def __init__(self, df, path, transforms=None):
        super().__init__()
        self.df = df
        self.path = path
        self.transforms = transforms

    def __getitem__(self, i):

        path = f'{self.path}/{self.df.iloc[i].patient_id}/{self.df.iloc[i].image_id}.png'
        try:
            img = Image.open(path).convert('RGB')
        except Exception as ex:
            print(path, ex)
            return None

        if self.transforms is not None:
            img = self.transforms(img)


        if TARGET in self.df.columns:
            cancer_target = torch.as_tensor(self.df.iloc[i].cancer)
            cat_aux_targets = torch.as_tensor(self.df.iloc[i][CATEGORY_AUX_TARGETS])
            return img, cancer_target, cat_aux_targets

        return img

    def __len__(self):
        return len(self.df)
    
    
class BreastCancerModel(torch.nn.Module):
    def __init__(self, aux_classes, model_type=Config.MODEL_TYPE, dropout=0.):
        super().__init__()
        self.model = create_model(model_type, pretrained=True, num_classes=0, drop_rate=dropout)

        #freeze
        #for param in self.model.parameters():
        #    param.requires_grad = False     

            
        self.backbone_dim = self.model(torch.randn(1, 3, MODEL_SIZE_X, MODEL_SIZE_Y)).shape[-1]

        self.nn_cancer = torch.nn.Sequential(
            torch.nn.Linear(self.backbone_dim, 1),
        )
        self.nn_aux = torch.nn.ModuleList([
            torch.nn.Linear(self.backbone_dim, n) for n in aux_classes
        ])

    def forward(self, x):
        # returns logits
        x = self.model(x)

        cancer = self.nn_cancer(x).squeeze()
        aux = []
        for nn in self.nn_aux:
            aux.append(nn(x).squeeze())
        return cancer, aux

    def predict(self, x):
        cancer, aux = self.forward(x)
        sigaux = []
        for a in aux:
            sigaux.append(torch.softmax(a, dim=-1))
        return torch.sigmoid(cancer), sigaux
    

def get_transforms(aug=False):
    def transforms(img):
        img = img.convert('RGB').resize((MODEL_SIZE_Y, MODEL_SIZE_X))# convert to RGB to use pretrained model
        if aug:
            tfm = [
                torchvision.transforms.RandomHorizontalFlip(0.5),
                torchvision.transforms.RandomRotation(degrees=(-5, 5)), 
                #torchvision.transforms.RandomAffine(degrees=(0, 0), translate=(0.05, 0.05)),
                #torchvision.transforms.RandomResizedCrop((MODEL_SIZE_X, MODEL_SIZE_Y), scale=(0.8, 1), ratio=(0.45, 0.55)) 
            ]
        else:
            tfm = [
                #torchvision.transforms.RandomHorizontalFlip(0.5),
                #torchvision.transforms.Resize((MODEL_SIZE_X, MODEL_SIZE_Y))
            ]
        img = torchvision.transforms.Compose(tfm + [            
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=MODEL_MEAN, std=MODEL_STD),
            #torchvision.transforms.Normalize(mean=0.2179, std=0.0529),            
        ])(img)
        return img

    return lambda img: transforms(img)


def save_model(name, model, thres, model_type):
    torch.save({'model': model.state_dict(), 'threshold': thres, 'model_type': model_type}, f'{name}')
    
def load_model(name, dir='.', model=None):
    data = torch.load(os.path.join(dir, f'{name}'), map_location=DEVICE)
    if model is None:
        model = BreastCancerModel(AUX_TARGET_NCLASSES, data['model_type'])
    model.load_state_dict(data['model'])
    return model, data['threshold'], data['model_type']  


def pfbeta(labels, predictions, beta=1.):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / max(y_true_count, 1)  # avoid / 0
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0
    
def precision_recall(labels, predictions):
    y_true_count = 0
    ctp = 0
    cfp = 0
    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / max(y_true_count, 1)  # avoid / 0
    return c_precision, c_recall
    

def optimal_f1(labels, predictions):
    thres = np.linspace(0, 1, 101)
    f1s = [pfbeta(labels, predictions > thr) for thr in thres]
    idx = np.argmax(f1s)
    return f1s[idx], thres[idx]

def evaluate_model(model: BreastCancerModel, ds, max_batches=PREDICT_MAX_BATCHES, shuffle=False, config=Config):
    torch.manual_seed(42)
    model = model.to(DEVICE)
    dl_test = torch.utils.data.DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=shuffle, pin_memory=False)
    pred_cancer = []
    with torch.no_grad():
        
        model.eval()
        cancer_losses = []
        aux_losses = []
        losses = []
        targets = []
        with tqdm(dl_test, desc='Eval', mininterval=30) as progress:
            for i, (X, y_cancer, y_aux) in enumerate(progress):
                with autocast(enabled=True):
                    y_aux = y_aux.to(DEVICE)
                    X = X.to(DEVICE)
                    y_cancer_pred, aux_pred = model.forward(X)

                    cancer_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        y_cancer_pred, 
                        y_cancer.to(float).to(DEVICE),
                        pos_weight=torch.tensor([config.POSITIVE_TARGET_WEIGHT]).to(DEVICE)
                    ).item()
                    aux_loss = torch.mean(torch.stack([torch.nn.functional.cross_entropy(aux_pred[i], y_aux[:, i]) for i in range(y_aux.shape[-1])])).item()
                    pred_cancer.append(torch.sigmoid(y_cancer_pred))
                    cancer_losses.append(cancer_loss)
                    aux_losses.append(aux_loss)
                    losses.append(cancer_loss + config.AUX_LOSS_WEIGHT * aux_loss)
                    targets.append(y_cancer.cpu().numpy())
                if i >= max_batches:
                    break
        targets = np.concatenate(targets)
        pred = torch.concat(pred_cancer).cpu().numpy()
        pf1, thres = optimal_f1(targets, pred)
        precision, recall = precision_recall(targets, pred)
        return np.mean(cancer_losses), (pf1, thres), pred, np.mean(losses), np.mean(aux_losses), precision, recall
    
    
def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()
    

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or np.any([v in name.lower()  for v in skip_list]):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def train_model(ds_train, ds_eval, logger, name, config=Config, do_save_model=True):
    torch.manual_seed(42)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)

    model = BreastCancerModel(AUX_TARGET_NCLASSES, config.MODEL_TYPE, config.DROPOUT).to(DEVICE)

    if config.ADAMW:
        optim = torch.optim.AdamW(add_weight_decay(model, weight_decay=config.ADAMW_DECAY, skip_list=['bias']), lr=config.ONE_CYCLE_MAX_LR, betas=(0.9, 0.999), weight_decay=config.ADAMW_DECAY)
    else:
        optim = torch.optim.Adam(model.parameters())


    scheduler = None
    if config.ONE_CYCLE:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=config.ONE_CYCLE_MAX_LR, epochs=config.EPOCHS,
                                                        steps_per_epoch=len(dl_train),
                                                        pct_start=config.ONE_CYCLE_PCT_START)
        
    

    scaler = GradScaler()
    best_eval_score = 0
    for epoch in tqdm(range(config.EPOCHS), desc='Epoch'):

        model.train()
        with tqdm(dl_train, desc='Train', mininterval=30) as train_progress:
            for batch_idx, (X, y_cancer, y_aux) in enumerate(train_progress):
                y_aux = y_aux.to(DEVICE)

                optim.zero_grad()
                # Using mixed precision training
                with autocast():
                    y_cancer_pred, aux_pred = model.forward(X.to(DEVICE))
                    cancer_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        y_cancer_pred,
                        y_cancer.to(float).to(DEVICE),
                        pos_weight=torch.tensor([config.POSITIVE_TARGET_WEIGHT]).to(DEVICE)
                    )
                    aux_loss = torch.mean(torch.stack([torch.nn.functional.cross_entropy(aux_pred[i], y_aux[:, i]) for i in range(y_aux.shape[-1])]))
                    loss = cancer_loss + config.AUX_LOSS_WEIGHT * aux_loss
                    if np.isinf(loss.item()) or np.isnan(loss.item()):
                        print(f'Bad loss, skipping the batch {batch_idx}')
                        del loss, cancer_loss, y_cancer_pred
                        gc_collect()
                        continue

                # scaler is needed to prevent "gradient underflow"
                scaler.scale(loss).backward()
                scaler.step(optim)
                if scheduler is not None:
                    scheduler.step()
                    
                scaler.update()

                lr = scheduler.get_last_lr()[0] if scheduler else config.ONE_CYCLE_MAX_LR


        if ds_eval is not None and MAX_EVAL_BATCHES > 0:
            cancer_loss, (f1, thres), _, loss, aux_loss, precision, recall = evaluate_model(
                model, ds_eval, max_batches=MAX_EVAL_BATCHES, shuffle=False, config=config)

            if f1 > best_eval_score:
                best_eval_score = f1
                if do_save_model:
                    save_model(name, model, thres, config.MODEL_TYPE)
                    
            print({'eval_cancer_loss': cancer_loss, 'eval_f1': f1, 'max_eval_f1': best_eval_score,
                   'eval_f1_thres': thres, 'eval_loss': loss, 'eval_aux_loss': aux_loss, 
                   'precision': precision,'recall': recall,'epoch': epoch})

    return model



if __name__ == "__main__":
    
    # adjust dataframe of image data
    df_train = pd.read_csv('/scratch/eecs545w23_class_root/eecs545w23_class/yngmkim/train_data_split.csv')
    
    split = StratifiedGroupKFold(N_FOLDS)
    for k, (_, test_idx) in enumerate(split.split(df_train, df_train.cancer, groups=df_train.patient_id)):
        df_train.loc[test_idx, 'split'] = k
    df_train.split = df_train.split.astype(int)
    
    df_train.age.fillna(df_train.age.mean(), inplace=True)
    df_train['age'] = pd.qcut(df_train.age, 10, labels=range(10), retbins=False).astype(int)
    
    df_train[CATEGORY_AUX_TARGETS] = df_train[CATEGORY_AUX_TARGETS].apply(LabelEncoder().fit_transform)
    
    # make train dataset
    ds_train = BreastCancerDataSet(df_train, TRAIN_IMAGES_PATH, get_transforms(aug=True))
    
    # because AUX_TARGET_NCLASSES are different among splits define manually
    AUX_TARGET_NCLASSES = [2, 2, 6, 2, 2, 2, 4, 5, 2, 10, 10]
    
    for fold in FOLDS:
        gc_collect()
        ds_train = BreastCancerDataSet(df_train.query('split != @fold'), TRAIN_IMAGES_PATH, get_transforms(aug=Config.AUG))
        ds_eval = BreastCancerDataSet(df_train.query('split == @fold'), TRAIN_IMAGES_PATH, get_transforms(aug=False))
        train_model(ds_train, ds_eval, None, f'{MODELS_PATH}/{MODEL_PREFIX}-model-f{fold}')