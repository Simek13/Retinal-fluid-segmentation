from datasets.dataset import Dataset
from os.path import splitext, join
from os import listdir
import numpy as np
from PIL import Image


class ImbalancedDataset(Dataset):
    def __init__(self, root_path, mask_ext='.png'):
        super().__init__(root_path, mask_ext)

    def load_data(self):
        images_path = self.root_path + "/images"
        masks_path = self.root_path + "/masks"
        image_files = [d for d in listdir(images_path)]

        # full_data = []
        balanced_data = []
        for im in image_files:
            name = splitext(im)[0]
            mask_name = name + self.mask_ext
            mask = np.array(Image.open(join(masks_path, mask_name)), dtype=np.uint8)
            if np.count_nonzero(mask) != 0:
                balanced_data.append((im, name + self.mask_ext))
            # full_data.append((im, name + self.mask_ext))

        # return full_data, balanced_data
        return balanced_data
