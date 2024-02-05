import os
import cv2
import numpy as np
def read_image_mask(fragment_id,start_idx=15,end_idx=45):

    images = []

    # idxs = range(65)
    mid = 65 // 2

    idxs = range(start_idx, end_idx)

    for i in idxs:
        
        image = cv2.imread( f"train_scrolls/{fragment_id}/layers/{i:02}.tif", 0)

        pad0 = (256 - image.shape[0] % 256)
        pad1 = (256 - image.shape[1] % 256)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        # image = ndimage.median_filter(image, size=5)
        
        # image = cv2.resize(image, (image.shape[1]//2,image.shape[0]//2), interpolation = cv2.INTER_AREA)
        image=np.clip(image,0,200)
        images.append(image)
    images = np.stack(images, axis=2)
    if fragment_id in ['20230701020044','verso','20230901184804','20230901234823','20230531193658','20231007101615','20231005123333','20231011144857','20230522215721', '20230919113918', '20230625171244','20231022170900','20231012173610','20231016151000']:
        images=images[:,:,::-1]
    if fragment_id in ['20231022170901','20231022170900']:
        mask = cv2.imread( f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.tiff", 0)
    else:
        mask = cv2.imread(f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.png", 0)

    fragment_mask=cv2.imread( f"train_scrolls/{fragment_id}/{fragment_id}_mask.png", 0)
    fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    mask = mask.astype('float32')
    mask/=255
    return images, mask,fragment_mask

def run_sanity_checks():
    for fragment_id in ['20231210121321','20231106155350','20231005123336','20230820203112','20230620230619','20230826170124','20230702185753','20230522215721','20230531193658','20230520175435','20230903193206','20230902141231','20231007101615','20230929220924','recto','verso','20231016151000','20231012184423','20231031143850']:  
        print(fragment_id)
        assert os.path.exists(f'train_scrolls/{fragment_id}/layers/00.tif')
        assert os.path.exists(f'train_scrolls/{fragment_id}/{fragment_id}_inklabels.png')
        assert os.path.exists(f'train_scrolls/{fragment_id}/{fragment_id}_mask.png')
    assert os.path.exists(f'train_scrolls/20231022170901/layers/00.tif')
    assert os.path.exists(f'train_scrolls/20231022170901/20231022170901_inklabels.tiff')
    assert os.path.exists(f'train_scrolls/20231022170901/20231022170901_mask.png')
def prepare_data():
    for l in os.listdir('all_labels/'):
        if '.png' in l:
            f_id=l[:-14]
            if os.path.exists(f'train_scrolls/{f_id}'):
                img=cv2.imread(f'all_labels/{f_id}_inklabels.png', 0)
                cv2.imwrite(f"train_scrolls/{f_id}/{f_id}_inklabels.png", img) 
        if '.tiff' in l:
            f_id=l[:-15]
            if os.path.exists(f'train_scrolls/{f_id}'):
                img=cv2.imread(f'all_labels/{f_id}_inklabels.tiff', 0)
                cv2.imwrite(f"train_scrolls/{f_id}/{f_id}_inklabels.tiff", img) 
if __name__ == "__main__":
    prepare_data()
    run_sanity_checks()