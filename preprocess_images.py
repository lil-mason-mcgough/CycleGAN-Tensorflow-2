import numpy as np
import cv2

OUTPUT_EXT = '.jpg'

def argmax(mylist):
    return max(range(len(mylist)), key=lambda i: mylist[i])

def argmin(mylist):
    return min(range(len(mylist)), key=lambda i: mylist[i])

def imread(img_path):
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def resize(img, dims):
    img = cv2.resize(img, dims[::-1], interpolation=cv2.INTER_AREA)
    return img

def imwrite(img, img_path):
    cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])

def dice_image(img, interval, axis=0):
    assert img.shape[axis] >= interval, 'interval ({}) must not be larger '\
        'than selected axis size ({})'.format(interval, img.shape[axis])

    n_crops = img.shape[axis] // interval
    img_crops = []
    for i in range(n_crops):
        lower = i * interval
        upper = (i + 1) * interval
        img_crop = img.take(indices=range(lower, upper), axis=axis)
        img_crops.append(img_crop)
    # crop remainder if not covered yet
    if img.shape[axis] % interval:
        lower = img.shape[axis] - interval
        upper = img.shape[axis]
        img_crop = img.take(indices=range(lower, upper), axis=axis)
        img_crops.append(img_crop)

    return img_crops


if __name__ == '__main__':
    import os, glob
    import argparse
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x: x

    parser = argparse.ArgumentParser(description='Load, resize, dice, and '\
        'write a set of images.')
    parser.add_argument('input_pattern', type=str, help='glob-style pattern for '\
        'selecting input images, wrapped in quotes.')
    parser.add_argument('output_dir', type=str, help='Directory to save outputs to.')
    parser.add_argument('-r', '--recursive', action='store_true',
        help='If true, searches <input_pattern> recursively.')
    parser.add_argument('-w', '--width', type=int, help='Output width to save '\
        'image. Currently resizes preserving aspect ratio and then dice crops image.')
    args = parser.parse_args()

    img_paths = glob.glob(args.input_pattern, recursive=args.recursive)
    for img_path in tqdm(img_paths):
        # load
        img = imread(img_path)

        # resize
        size_orig = img.shape[0:2]
        short_dim = argmin(size_orig)
        resize_ratio = args.width / size_orig[short_dim]
        size_resize = map(lambda x: resize_ratio * x, size_orig)
        size_resize = tuple(map(int, map(round, size_resize)))
        img = resize(img, size_resize)

        # dice
        long_dim = 1 - short_dim
        img_crops = dice_image(img, args.width, axis=long_dim)

        # write
        for i, img_crop in enumerate(img_crops):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            output_img_path = os.path.join(args.output_dir,
                '{}_{:02d}{}'.format(img_name, i, OUTPUT_EXT))
            imwrite(img_crop, output_img_path)
