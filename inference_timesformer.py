import os
import torch.nn as nn
import torch.nn.functional as F
from timesformer_pytorch import TimeSformer
import torch
from warmup_scheduler import GradualWarmupScheduler
import wandb
import random
import gc
import pytorch_lightning as pl
import scipy.stats as st
from torch.utils.data import DataLoader
import numpy as np
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tap import Tap
import glob

class InferenceArgumentParser(Tap):
    segment_id: list[str] =['20230925002745']
    segment_path:str='./eval_scrolls'
    model_path:str= 'outputs/vesuvius/pretraining_all/vesuvius-models/valid_20230827161847_0_fr_i3depoch=7.ckpt'
    out_path:str=""
    stride: int = 2
    start_idx:int=15
    workers: int = 4
    batch_size: int = 512
    size:int=64
    reverse:int=0
    device:str='cuda'
    format='tif'
args = InferenceArgumentParser().parse_args()
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    # comp_dir_path = './'
    comp_dir_path = './'
    comp_folder_name = './'
    comp_dataset_path = f'./'
    
    exp_name = 'pretraining_all'
    # ============== model cfg =============
    in_chans = 26 # 65
    encoder_depth=5
    # ============== training cfg =============
    size = 64
    tile_size = 64
    stride = tile_size // 3

    train_batch_size = 256 # 32
    valid_batch_size = 256
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    epochs = 50 # 30

    # adamW warmupあり
    warmup_factor = 10
    # lr = 1e-4 / warmup_factor
    lr = 1e-4 / warmup_factor
    min_lr = 1e-6
    num_workers = 16
    seed = 42
    # ============== augmentation =============
    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]
def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False

def cfg_init(cfg, mode='val'):
    set_seed(cfg.seed)
cfg_init(CFG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def read_image_mask(fragment_id,start_idx=18,end_idx=38,rotation=0):
    images = []
    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start_idx, end_idx)
    for i in idxs:
        image = cv2.imread(f"{args.segment_path}/{fragment_id}/layers/{i:02}.{args.format}", 0)
        pad0 = (256 - image.shape[0] % 256)
        pad1 = (256 - image.shape[1] % 256)
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        image=np.clip(image,0,200)
        images.append(image)
    images = np.stack(images, axis=2)
    if args.reverse != 0 or fragment_id in ['20230701020044','verso','20230901184804','20230901234823','20230531193658','20231007101615','20231005123333','20231011144857','20230522215721', '20230919113918', '20230625171244','20231022170900','20231012173610','20231016151000']:
        print("Reverse Segment")
        images=images[:,:,::-1]

    fragment_mask=None
    wildcard_path_mask = f'{args.segment_path}/{fragment_id}/*_mask.png'
    if os.path.exists(f'{args.segment_path}/{fragment_id}/{fragment_id}_mask.png'):
        fragment_mask=cv2.imread(CFG.comp_dataset_path + f"{args.segment_path}/{fragment_id}/{fragment_id}_mask.png", 0)
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    elif len(glob.glob(wildcard_path_mask)) > 0:
        # any *mask.png exists
        mask_path = glob.glob(wildcard_path_mask)[0]
        fragment_mask = cv2.imread(mask_path, 0)
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    else:
        # White mask
        fragment_mask = np.ones_like(images[:,:,0]) * 255

    return images,fragment_mask

def get_img_splits(fragment_id,s,e,rotation=0):
    images = []
    xyxys = []
    image,fragment_mask = read_image_mask(fragment_id,s,e,rotation)
    x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
    y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))
    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size
            if not np.any(fragment_mask[y1:y2, x1:x2]==0):
                images.append(image[y1:y2, x1:x2])
                xyxys.append([x1, y1, x2, y2])
    test_dataset = CustomDatasetTest(images,np.stack(xyxys), CFG,transform=A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.Normalize(
            mean= [0] * CFG.in_chans,
            std= [1] * CFG.in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]))

    test_loader = DataLoader(test_dataset,
                              batch_size=CFG.valid_batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False,
                              )
    return test_loader, np.stack(xyxys),(image.shape[0],image.shape[1]),fragment_mask

def get_transforms(data, cfg):
    if data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug

class CustomDatasetTest(Dataset):
    def __init__(self, images,xyxys, cfg, transform=None):
        self.images = images
        self.xyxys=xyxys
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        xy=self.xyxys[idx]
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)
        return image,xy
    
class RegressionPLModel(pl.LightningModule):
    def __init__(self,pred_shape,size=64,enc='',with_norm=False):
        super(RegressionPLModel, self).__init__()
        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
        self.backbone=TimeSformer(
                dim = 512,
                image_size = 64,
                patch_size = 16,
                num_frames = 30,
                num_classes = 16,
                channels=1,
                depth = 8,
                heads = 6,
                dim_head =  64,
                attn_dropout = 0.1,
                ff_dropout = 0.1
            )
        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)

    def forward(self, x):
        if x.ndim==4:
            x=x[:,None]
        if self.hparams.with_norm:
            x=self.normalization(x)
        x = self.backbone(torch.permute(x, (0, 2, 1,3,4)))
        x=x.view(-1,1,4,4)        
        return x
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        if torch.isnan(loss1):
            print("Loss nan encountered")
        self.log("train/Arcface_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}

    def validation_step(self, batch, batch_idx):
        x,y,xyxys= batch
        batch_size = x.size(0)
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            self.mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=16,mode='bilinear').squeeze(0).squeeze(0).numpy()
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/MSE_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}
    
    def configure_optimizers(self):

        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=CFG.lr)
    
        scheduler = get_scheduler(CFG, optimizer)
        return [optimizer],[scheduler]

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 10, eta_min=1e-6)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)

def predict_fn(test_loader, model, device, test_xyxys,pred_shape):
    mask_pred = np.zeros(pred_shape)
    mask_count = np.zeros(pred_shape)
    kernel=gkern(CFG.size,1)
    kernel=kernel/kernel.max()
    model.eval()

    for step, (images,xys) in tqdm(enumerate(test_loader),total=len(test_loader)):
        images = images.to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                y_preds = model(images)
        y_preds = torch.sigmoid(y_preds).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xys):
            mask_pred[y1:y2, x1:x2] += np.multiply(F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=16,mode='bilinear').squeeze(0).squeeze(0).numpy(),kernel)
            mask_count[y1:y2, x1:x2] += np.ones((CFG.size, CFG.size))

    mask_pred /= mask_count
    return mask_pred
import gc

if __name__ == "__main__":
    model=RegressionPLModel.load_from_checkpoint(args.model_path,strict=False)
    model.cuda()
    model.eval()
    wandb.init(
        project="Vesuvius", 
        name=f"ALL_scrolls_tta", 
        )
    for fragment_id in args.segment_id:
        if os.path.exists(f"{args.segment_path}/{fragment_id}/layers/00.{args.format}"):
            preds=[]
            for r in [0]:
                for i in [17]:
                    start_f=i
                    end_f=start_f+CFG.in_chans
                    test_loader,test_xyxz,test_shape,fragment_mask=get_img_splits(fragment_id,start_f,end_f,r)
                    mask_pred= predict_fn(test_loader, model, device, test_xyxz,test_shape)
                    mask_pred=np.clip(np.nan_to_num(mask_pred),a_min=0,a_max=1)
                    mask_pred/=mask_pred.max()

                    preds.append(mask_pred)

            img=wandb.Image(
            preds[0], 
            caption=f"{fragment_id}"
            )
            wandb.log({'predictions':img})
            gc.collect()

            if len(args.out_path) > 0:
                # CV2 image
                image_cv = (mask_pred * 255).astype(np.uint8)
                try:
                    os.makedirs(args.out_path,exist_ok=True)
                except:
                    pass
                cv2.imwrite(os.path.join(args.out_path, f"{fragment_id}_prediction.png"), image_cv)

    del mask_pred,test_loader,model
    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish()
