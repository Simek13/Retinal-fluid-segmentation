from os import listdir, makedirs
from os.path import join, splitext, dirname
from shutil import copyfile


def image_to_mask(image_name):
    fname, ext = splitext(image_name)
    return fname + '_mask' + ext


def process(root_path):
    image_path = join(root_path, 'images')
    mask_path = join(root_path, 'masks')
    try:
        makedirs(image_path)
    except:
        pass
    try:
        makedirs(mask_path)
    except:
        pass

    for d in listdir(root_path):
        if d != 'images' and d != 'masks':
            for file in listdir(join(root_path, d, 'images')):
                copyfile(join(root_path, d, 'images', file), join(image_path, file))
                copyfile(join(root_path, d, 'masks', image_to_mask(file)), join(mask_path, file))


if __name__ == '__main__':
    process('./train')


