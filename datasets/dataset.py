from os.path import splitext
from os import listdir


class Dataset:

    def __init__(self, root_path, img_ext, mask_ext):
        self.root_path = root_path
        self.img_ext = img_ext
        self.mask_ext = mask_ext

    def load_data(self):
        images_path = self.root_path + "/images"
        image_files = [d for d in listdir(images_path)]

        data = []
        for im in image_files:
            name = splitext(im)[0]
            data.append((name + self.img_ext, name + self.mask_ext))

        return data
