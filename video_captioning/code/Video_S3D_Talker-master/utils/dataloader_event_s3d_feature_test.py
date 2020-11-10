
'''
    Author: Zhiyuan Jacob Fang
    Dataset script to load and collate batch files.
'''
import os
import cv2
import json
import pickle
import torch
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset


class YC2Dataset(Dataset):

    # Auxiliary functions
    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.word_to_ix

    def get_ix_to_word(self):
        return self.ix_to_word

    def __init__(self, opt, mode='training'):
        super(YC2Dataset, self).__init__()
        self.opt = opt
        self.mode = mode

        # Load the caption annotations
        with open(os.path.join(opt['ann_path'], opt['cap_file'])) as json_file:
            self.caption_hub = json.load(json_file)['database']

        # Read the training/testing split
        SPLITS = ['train', 'val']

        # sample_list: [[train_list], [val_list]]
        self.splits = []
        for split in SPLITS:
            with open(os.path.join(opt['ann_path'], 'new_splits/{param}_list.txt'.format(param=split))) as f:
                self.splits.append(f.readlines())

        # Load the vocabulary dictionary
        vocab = json.load(open(opt['vocab_file']))
        self.word_to_ix = vocab['word_to_ix']
        self.ix_to_word = vocab['ix_to_word']

        # TODO: create video_feat by combining seg
        # Load video features
        self.video_feat = []
        for split in SPLITS:
            v = os.path.join(opt['feat_path'], 'new_splits/{param}_frame_feat_csv'.format(param=split))
            for directory in os.listdir(v):
                curr_video = []
                for dirs in os.listdir(os.path.join(v, directory)):
                    for feat in os.listdir(os.path.join(v, directory, dirs)):
                        if feat.endswith(".csv"):
                            torch.cat(curr_video, feat)
                self.video_feat.append(curr_video);

        # # Load GPT language features
        # self.language_feat_hub = pickle.load(open(opt['language_feat_path'], 'rb'), encoding='iso-8859-1')
        #
        # # S3D feature hub
        # self.s3d_feat_hub = pickle.load(open(opt['s3d_feat_path'], 'rb'), encoding='iso-8859-1')

    def __getitem__(self, video_id=False):

        if self.mode == 'training':
            video_id = self.splits[0][video_id].replace('\n', '')
        elif self.mode == 'validation':
            video_id = self.splits[1][video_id].replace('\n', '')

        seg_idx = video_id.split('_')[-1]
        video_id = video_id[ : -len(seg_idx)-1]
        # video_s3d_feat = self.s3d_feat_hub[video_id + seg_idx]

        # TODO: create video_feat by combining seg
        video_feat = self.video_feat

        # Load the caption and time boundaries
        data = {}
        annotation = self.caption_hub[video_id]
        duration = annotation['duration']
        recipe = annotation['recipe_type']
        video_url = annotation['video_url']
        segment_length = len(annotation['annotations'])

        caption = self.caption_hub[video_id]['annotations'][int(seg_idx)]['sentence']
        segment = self.caption_hub[video_id]['annotations'][int(seg_idx)]['segment']
        # cap_features = self.language_feat_hub[video_id+seg_idx]

        # TODO: Get video_features -> concatenate features for each video ID
        data['arr_length'] = len(video_feat)
        data['video_feat'] = torch.tensor(video_feat)

        # data['video_feat'] = torch.tensor(video_s3d_feat)
        data['segment'] = segment
        data['caption'] = caption
        # data['cap_feat'] = cap_features
        # data['arr_length'] = video_s3d_feat.shape[0]
        data['duration'] = duration
        data['video_id'] = video_id+str(seg_idx)
        return data

    def __len__(self):
        if self.mode == 'training':
            length = len(self.splits[0])
        elif self.mode == 'validation':
            length = len(self.splits[1])
        elif self.mode == 'testing':
            length = len(self.splits[2])
        return length


# batch collate function
def yc2_collate_fn(batch_lst):
    '''
    :param batch_lst: Raw instance level annotation from YC2 Dataset class.
    :return: batch annotations that include: features, captions, segments, length of features and length of videos.
    '''
    batch_lens = [_['arr_length'] for _ in batch_lst]
    max_length = max(batch_lens)
    batch_feat = torch.zeros((len(batch_lst), max_length, 1024))
    captions = []
    segments = []
    durations = []
    video_ids = []
    language_feat = torch.zeros((len(batch_lst), 768))

    for batch_id, batch_data in enumerate(batch_lst):
        batch_feat[batch_id][: batch_data['arr_length']] = batch_data['video_feat']
        captions.append(batch_data['caption'])
        segments.append(batch_data['segment'])
        durations.append(batch_data['duration'])
        video_ids.append(batch_data['video_id'])
        language_feat[batch_id] = torch.tensor(batch_data['cap_feat'])
    return batch_feat, language_feat, captions, segments, torch.tensor(batch_lens), durations, video_ids