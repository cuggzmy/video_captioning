import json
import torch
import random
import numpy as np
from utils.utils import *
from utils.opts_C3D_videos import *
from torch.utils.data import DataLoader
from model.Model_S3D_videos import Model
from model.transformer.Constants import *
from nltk.translate.bleu_score import corpus_bleu
from model.transformer.c3d_Translator import translate_batch
from utils.dataloader_event_s3d_feature_test import YC2Dataset, yc2_collate_fn

import sys
sys.path.append("utils/pycocoevalcap/")

from bleu.bleu import Bleu
from cider.cider import Cider
from meteor.meteor import Meteor
from utils.rouge import Rouge


def pos_emb_generation(visual_feats):
    '''
        Generate the position embedding input for Transformers.
    '''
    seq = list(range(1, visual_feats.shape[1] + 1))
    src_pos = torch.tensor([seq] * visual_feats.shape[0]).cuda()
    return src_pos


def list_to_sentence(list):
    sentence = ''
    for element in list:
        sentence += ' ' + element
    return sentence


def test(loader, model, opt, vocab):
    bleu_scores = []
    write_to_txt = []

    res = {}
    gts = {}

    for batch_id, (video_input, language_feat, captions, time_seg, batch_lens, duration, video_id) in enumerate(loader):

        # Convert the textual input to numeric labels
        cap_gts, cap_mask = convert_caption_labels(captions, loader.dataset.get_vocab(), opt['max_length'])

        video_input = video_input.cuda()
        cap_gts = torch.tensor(cap_gts).cuda().long()
        # cap_mask = cap_mask.cuda()

        with torch.no_grad():
            # Beam Search Starts From Here
            batch_hyp = translate_batch(model, video_input, opt)

        # Stack all GTs captions
        references = [[cap.split(' ')] for cap in captions]

        # Stack all Predicted Captions
        hypotheses = []
        for predict in zip(batch_hyp):
            predict = predict[0]
            _ = []
            if EOS in predict[0]:
                sep_id = predict[0].index(EOS)
            else:
                sep_id = -1
            for word in predict[0][0: sep_id]:
                _.append(vocab[str(word)])
            hypotheses.append(_)

        # Stack all predictions for the Gougue/Meteour Scores
        res[batch_id] = [list_to_sentence(hypotheses[0])]
        gts[batch_id] = [list_to_sentence(references[0][0])]
        print(batch_id)
    avg_bleu_score, bleu_scores = Bleu(4).compute_score(gts, res)
    avg_cider_score, cider_scores = Cider().compute_score(gts, res)
    avg_meteor_score, meteor_scores = Meteor().compute_score(gts, res)
    avg_rouge_score, rouge_scores = Rouge().compute_score(gts, res)
    print('C, M, R, B:', avg_cider_score, avg_meteor_score, avg_rouge_score, avg_bleu_score)

def main(opt):
    dataset = YC2Dataset(opt, 'validation')
    dataloader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=False, collate_fn=yc2_collate_fn)

    model = Model(
        dataset.get_vocab_size(),
        cap_max_seq=opt['max_length'],
        tgt_emb_prj_weight_sharing=True,
        d_k=opt['dim_head'],
        d_v=opt['dim_head'],
        d_model=opt['dim_model'],
        d_word_vec=opt['dim_word'],
        d_inner=opt['dim_inner'],
        n_layers=opt['num_layer'],
        n_head=opt['num_head'],
        dropout=0.1,
        c3d_path=opt['c3d_path'])


    model =  nn.DataParallel(model)
    state_dict = torch.load(opt['load_checkpoint'])
    model.load_state_dict(state_dict)

    model.eval().cuda()
    test(dataloader, model, opt, dataloader.dataset.get_ix_to_word())


if __name__ == '__main__':
    opt = parse_opt()
    opt = vars(opt)
    opt['batch_size'] = 1
    main(opt)