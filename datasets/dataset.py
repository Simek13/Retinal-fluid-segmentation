from os.path import splitext, join
from os import listdir


class Dataset:

    def __init__(self, root_path, mask_ext='.png'):
        self.root_path = root_path
        self.mask_ext = mask_ext

    def load_data(self):
        images_path = self.root_path + "/images"
        image_files = [d for d in listdir(images_path)]

        data = []
        for im in image_files:
            name = splitext(im)[0]
            data.append((im, name + self.mask_ext))

        return data

    def load_volumes(self):
        train_path = self.root_path + '/train'
        val_path = self.root_path + '/val'
        test_path = self.root_path + '/test'

        train_data = []
        val_data = []
        test_data = []

        for v in listdir(train_path):
            for im in listdir(join(train_path, v, 'images')):
                name = splitext(im)[0]
                train_data.append((v, im, name + self.mask_ext))
        for v in listdir(val_path):
            for im in listdir(join(val_path, v, 'images')):
                name = splitext(im)[0]
                val_data.append((v, im, name + self.mask_ext))
        for v in listdir(test_path):
            for im in listdir(join(test_path, v, 'images')):
                name = splitext(im)[0]
                test_data.append((v, im, name + self.mask_ext))
        return train_data, val_data, test_data
