import os

ROOT_DATASET = ""

def return_ucf101(modality):
    filename_categories = 101

    root_data = "/"
    filename_imglist_train = "dataset/ucf101/ucf101_train_split_1_rawframes.txt"
    filename_imglist_val = "dataset/ucf101/ucf101_val_split_1_rawframes.txt"
    prefix = 'img_{:05d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    filename_categories = 51

    root_data = "/"
    filename_imglist_train = "hmdb51/hmdb51_train_split_1_rawframes.txt"
    filename_imglist_val = "hmdb51/hmdb51_val_split_1_rawframes.txt"
    prefix = 'img_{:05d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality, root_dataset):
    dict_single = {'ucf101': return_ucf101, 'hmdb51': return_hmdb51}
    global ROOT_DATASET
    ROOT_DATASET = root_dataset
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
