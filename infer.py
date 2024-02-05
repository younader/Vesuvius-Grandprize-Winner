import wandb
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import gc
import pytorch_lightning as pl
import numpy as np
import scipy.stats as st
from torch.utils.data import DataLoader, Dataset
import os
import gc
import random
import numpy as np
from tqdm.auto import tqdm
import segmentation_models_pytorch as smp
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from i3dallnl import InceptionI3d
from PIL import Image
import re

class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    comp_dir_path = './'
    comp_folder_name = './'
    # comp_dataset_path = f'{comp_dir_path}datasets/{comp_folder_name}/'
    comp_dataset_path = f'./'
    
    exp_name = 'pretraining_all'

    in_chans = 33 # 65

    valid_id = 2

    print_freq = 50
    num_workers = 32

    size = 64
    tile_size = 64
    stride = tile_size // 2
    valid_batch_size = 32

    outputs_path = f'/home/ubuntu/Vesuvius-GrandPrize/outputs/{comp_name}/{exp_name}/'

    model_dir = outputs_path + \
        f'{comp_name}-models/'

    figures_dir = outputs_path + 'figures/'

    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'

def set_seed(seed=42, cudnn_deterministic=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False

set_seed()

# 2d gaussian kernel
def gkern(kernlen=21, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_image_mask(fragment_id, start_idx=18, end_idx=38):
    images = []

    mid = 65 // 2
    idxs = range(start_idx, end_idx)
    
    for i in idxs:
        p = f"/home/ubuntu/scroll_data/scroll_inkdetection/dataset_flat/raw_fragments/{fragment_id}/layers/{i:02}.tif"
        # print(os.path.exists(p))
        image = cv2.imread(p, 0)

        pad0 = (256 - image.shape[0] % 256)
        pad1 = (256 - image.shape[1] % 256)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        image = np.clip(image, 0, 200)

        images.append(image)

    images = np.stack(images, axis=2)
    if str(fragment_id) in ['20230701020044','verso','20230901184804','20230901234823','20230531193658','20231007101615','20231005123333','20231011144857','20230522215721', '20230919113918', '20230625171244','20231022170900','20231012173610','20231016151000','20231224042141', '20231222233538']:        
        images = images[:,:,::-1]

    fragment_mask = np.zeros(images[0].shape)
    if os.path.exists(f'/home/ubuntu/scroll_data/scroll_inkdetection/dataset_flat/raw_fragments/{fragment_id}/{re.sub("[^0-9]", "", fragment_id)}_mask.png'):
        fragment_mask=cv2.imread(f"/home/ubuntu/scroll_data/scroll_inkdetection/dataset_flat/raw_fragments/{fragment_id}/{re.sub('[^0-9]', '', fragment_id)}_mask.png", 0)
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)

    if os.path.exists(f'/home/ubuntu/scroll_data/scroll_inkdetection/dataset_flat/raw_fragments/{fragment_id}/out_mask.png'):
        fragment_mask=cv2.imread(f"/home/ubuntu/scroll_data/scroll_inkdetection/dataset_flat/raw_fragments/{fragment_id}/out_mask.png", 0)
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
        
    return images,fragment_mask

def get_img_splits(fragment_id, start, end, rotation=0):
    images = []
    xyxys = []
    image, fragment_mask = read_image_mask(fragment_id, start, end)
    x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
    y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))
    for y1 in tqdm(y1_list):
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size
            if not np.any(fragment_mask[y1:y2, x1:x2]==0):
                images.append(image[y1:y2, x1:x2])
                xyxys.append([x1, y1, x2, y2])

    test_dataset = CustomDatasetTest(images, np.stack(xyxys), CFG, transform=A.Compose([
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
                                num_workers=CFG.num_workers,
                                pin_memory=True,
                                drop_last=False,
    )
    
    return test_loader, np.stack(xyxys),(image.shape[0],image.shape[1]), fragment_mask

class CustomDatasetTest(Dataset):
    def __init__(self, images,xyxys, cfg, transform=None):
        self.images = images
        self.xyxys = xyxys
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        xy = self.xyxys[idx]
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)

        return image,xy

class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i]+encoder_dims[i-1], encoder_dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i-1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            f_down = self.convs[i - 1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask

class RegressionPLModel(pl.LightningModule):
    def __init__(self, pred_shape, size=224, enc='', with_norm=False):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.15)
        self.loss_func = lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
        
        self.backbone = InceptionI3d(in_channels=1, num_classes=512, non_local=True)
        self.decoder = Decoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1,1,20,256,256))], upscale=1)

        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)

    def forward(self, x):
        if x.ndim==4:
            x = x[:,None]
        if self.hparams.with_norm:
            x = self.normalization(x)
        feat_maps = self.backbone(x)
        feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        
        return pred_mask
    
    # Other functions are not overidden as they are unused in inference
   
def TTA(x:torch.Tensor,model:nn.Module):
    shape=x.shape
    x = [x,*[torch.rot90(x,k=i,dims=(-2,-1)) for i in range(1,4)],]
    x = torch.cat(x,dim=0)
    x = model(x)
    x = x.reshape(4,shape[0],CFG.size//4,CFG.size//4)
    
    x = [torch.rot90(x[i],k=-i,dims=(-2,-1)) for i in range(4)]
    x = torch.stack(x,dim=0)
    return x.mean(0)

def predict_fn(test_loader, model, device, test_xyxys, pred_shape):
    mask_pred = np.zeros(pred_shape)
    mask_count = np.zeros(pred_shape)
    kernel = gkern(CFG.size, 1)
    kernel = kernel / kernel.max()
    model.eval()

    for _, (images, xys) in tqdm(enumerate(test_loader),total=len(test_loader)):
        images = images.to(device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                y_preds = model(images)
            # y_preds =TTA(images,model)

        y_preds = torch.sigmoid(y_preds).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xys):
            mask_pred[y1:y2, x1:x2] += np.multiply(F.interpolate(y_preds[i].unsqueeze(0).float(), scale_factor=4, mode='bilinear').squeeze(0).squeeze(0).numpy(),kernel)
            mask_count[y1:y2, x1:x2] += np.ones((CFG.size, CFG.size))

    mask_pred /= mask_count
    return mask_pred, mask_count > 0 # Return mask of what parts of the image are not part of the fragment

def run_on_fragment(fragment_id, model_name="wild12_64_20230820203112_0_fr_i3depoch=3.ckpt", use_wandb=True):
    model = RegressionPLModel.load_from_checkpoint(CFG.model_dir + model_name,strict=False)
    model.cuda()
    model.eval()
    if use_wandb:
        wandb.login()
        wandb.init(
            # Set the project where this run will be logged
            project="vesivus", 
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"ALL_scrolls_tta_{model_name}",
        )

    # for fragment_id in ['20230901184804','20230901234823','20230902141231','20230903193206','20230528112855','20230519215753','20230525200512','20230528112855','20230531121653','20230601204340','20230609123853','20230620230617','20230620230619','20230828154913','20230902141231','20230711201157']:
    preds=[]
    
    start_f = 15
    end_f = start_f + CFG.in_chans

    test_loader, test_xyxz, test_shape, _ = get_img_splits(fragment_id, start_f, end_f, 0)
    
    mask_pred, mask = predict_fn(test_loader, model, device, test_xyxz, test_shape)
    mask_pred = np.clip(np.nan_to_num(mask_pred), a_min=0, a_max=1)
    mask_pred /= mask_pred.max()
    
    preds.append(mask_pred)

    img = wandb.Image(
        preds[0], 
        caption=f"{fragment_id}"
    )

    if use_wandb:
        wandb.log({'predictions':img})

    # memory cleanup
    gc.collect()
    del mask_pred, test_loader, model
    torch.cuda.empty_cache()
    gc.collect()
    
    if use_wandb:
        wandb.finish()

    return preds[0], mask

image_path = lambda id: f"Vesuvius-GrandPrize/outputs/vesuvius/pretraining_all/figures/{id}.png"

if __name__ == '__main__':
    # If running as a standalone script, run inference on a fragment
    fragment_image, _ = run_on_fragment("working_4039_4898_8091")

    # Save that fragment's image generation
    fragment_image = Image.fromarray((fragment_image * 255).astype(np.uint8))
    fragment_image.save(image_path("working_4039_4898_8091"))
