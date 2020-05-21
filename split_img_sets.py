import numpy as np

def random_indices(array):
    idxs = np.arange(len(array), dtype=np.int64)
    np.random.shuffle(idxs)
    return idxs

def split_indices(idxs, train_split=0.8):
    n_idxs = len(idxs)
    n_training = int(round(n_idxs * train_split))
    train_idxs = all_idxs[:n_training]
    test_idxs = all_idxs[n_training:]
    return train_idxs, test_idxs

if __name__ == '__main__':
    import os, glob, shutil
    import argparse
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x: x

    parser = argparse.ArgumentParser(description='Randomly split set of '\
        'images into training and testing sets')
    parser.add_argument('input_pattern', type=str, help='glob-style pattern for '\
        'selecting input images, wrapped in quotes.')
    parser.add_argument('output_train_dir', type=str, help='Directory to '\
        'save training data to.')
    parser.add_argument('output_test_dir', type=str, help='Directory to '\
        'save testing data to.')
    parser.add_argument('-r', '--recursive', action='store_true',
        help='If true, searches <input_pattern> recursively.')
    parser.add_argument('-t', '--train_split', type=float, default=0.8, 
        help='The proportion of images to use as training. The rest will be '\
            'testing.')
    args = parser.parse_args()

    img_paths = glob.glob(args.input_pattern, recursive=args.recursive)
    all_idxs = random_indices(img_paths)
    train_idxs, test_idxs = split_indices(all_idxs, train_split=args.train_split)
    img_paths = np.array(img_paths)
    train_paths, test_paths = img_paths[train_idxs], img_paths[test_idxs]

    data_output_dir = args.output_train_dir
    print('Saving training images to {}'.format(data_output_dir))
    for img_path in tqdm(train_paths):
        img_name = os.path.basename(img_path)
        output_img_path = os.path.join(data_output_dir, img_name)
        shutil.copyfile(img_path, output_img_path)
    data_output_dir = args.output_test_dir
    print('Saving testing images to {}'.format(data_output_dir))
    for img_path in tqdm(test_paths):
        img_name = os.path.basename(img_path)
        output_img_path = os.path.join(data_output_dir, img_name)
        shutil.copyfile(img_path, output_img_path)
