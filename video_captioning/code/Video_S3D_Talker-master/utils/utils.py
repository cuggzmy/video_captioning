import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import model.transformer.Constants as Constants


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        # self.loss_fn = nn.NLLLoss(reduce=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, target, mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        # truncate to the same size
        batch_size = target.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss_fn(logits, target)
        output = torch.sum(loss * mask) / batch_size
        return output


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        # Action verb, increase the weight of these generation words
        loss_weight = torch.ones_like(gold).float()
        for i in range(0, len(gold), 47):
            loss_weight[i] = 10
        # loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')
        loss = (F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='none') * loss_weight).sum()

    return loss


def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    return loss, n_correct


def pos_emb_generation(word_labels):
    '''
        Generate the position embedding input for Transformers.
    '''

    seq = list(range(1, word_labels.shape[1] + 1))
    tgt_pos = torch.tensor([seq] * word_labels.shape[0]).cuda()
    binary_mask = (word_labels != 0).long()

    return tgt_pos*binary_mask


def show_prediction(seq_probs, labels, vocab):
    '''
        :return: predicted words and GT words.
    '''
    # Print out the predicted sentences and GT
    _ = seq_probs.view(labels.shape[0], labels.shape[1], -1)[0]
    pred_idx = torch.argmax(_, 1)
    print(' \n')
    print([vocab[str(widx.cpu().numpy())] for widx in pred_idx if widx != 0])
    print([vocab[str(word.cpu().numpy())] for word in labels[0] if word != 0])


def preprocess_cap(captions):
    '''
    Clear the special token.
    '''
    return [cap.replace('.', ' ').replace(',', ' ').replace(')', '').\
        replace('(', '').replace(':', '').replace('\\', '').replace('=', '').replace('/', ' ') for cap in captions]


def convert_caption_labels(captions, vocab, max_len):
    '''
    captions: list of captions;
    vocab: vocabulary of token to numeric labels.
    return: labels.
    '''
    captions = preprocess_cap(captions)
    cap_mask = np.zeros((len(captions), max_len))
    cap_gts = np.zeros((len(captions), max_len))

    for batch_idx, cap in enumerate(captions):
        cap = '<sos> ' + cap + ' <eos>'
        for j, w in enumerate(cap.split(' ')):
            cap_gts[batch_idx][j] = vocab.get(w, '1')

        non_zero = (cap_gts[batch_idx] == 0).nonzero()
        if len(non_zero[0]) != 0: cap_mask[batch_idx][:int(non_zero[0][0])] = 1
        else: cap_mask[batch_idx] += 1
    return cap_gts, cap_mask


def clip_sampler(data):
    '''
    :param data: Packed batch data, see Dataloader Collate Function for details.
    :return: Randomly sampled video clip- feature, caption, and feature max-/length of them.
    '''
    video_feat, captions, time_segs, feat_lengths, duration = data

    # Randomly sample video index
    rand_idx = [random.randint(0, len(_) - 1) for _ in time_segs]

    s_time_segs = [time_segs[i][rand_idx[i]] for i in range(len(time_segs))]
    s_captions = [captions[i][rand_idx[i]] for i in range(len(time_segs))]

    # Retrieve out the raw visual representations
    feature_idx = [[round((seg[0] / dur)*f_len), round((seg[1] / dur)*f_len)]
                   for seg, dur, f_len in zip(s_time_segs, duration, feat_lengths)]
    s_feat_lengths = [seg[1]-seg[0] for seg in feature_idx]
    feat_max_length = max(s_feat_lengths)
    batch_feat = torch.zeros((video_feat.shape[0], feat_max_length, 2048))

    # Insert the new features
    for batch_id in range(video_feat.shape[0]):
        start = feature_idx[batch_id][0]
        end = feature_idx[batch_id][1]
        batch_feat[batch_id][0:end-start] = video_feat[batch_id][start:end]
    return batch_feat, s_captions, s_feat_lengths, s_time_segs, duration

def generate_bce_labels(pos_num, neg_num):
    nce_labels = torch.zeros((pos_num + neg_num, 1)).cuda()
    nce_labels[0: pos_num] = 1
    nce_labels[neg_num:] = 0
    return nce_labels

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


