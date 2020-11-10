import os
import random
import numpy as np
from utils.utils import *
import torch.optim as optim
from utils.opts_C3D_videos import *
from torch.utils.data import DataLoader
from model.Model_S3D_videos import Model
from model.transformer.Optim import ScheduledOptim
from utils.dataloader_event_s3d_feature_test import YC2Dataset, yc2_collate_fn

def train(loader, model, optimizer, opt):
    model.train()

    for epoch in range(opt['epochs']):
        for (video_input, language_feat, captions, time_seg, batch_lens, duration, video_id) in loader:
            torch.cuda.synchronize()

            # Convert the textual input to numeric labels
            cap_gts, cap_mask = convert_caption_labels(captions, loader.dataset.get_vocab(), opt['max_length'])

            video_input = video_input.cuda()
            cap_gts = torch.tensor(cap_gts).cuda().long()
            # cap_mask = cap_mask.cuda()
            cap_pos = pos_emb_generation(cap_gts)

            optimizer.zero_grad()

            cap_probs = model(video_input, batch_lens, cap_gts, cap_pos)

            cap_loss, cap_n_correct = cal_performance(cap_probs, cap_gts[:, 1:], smoothing=False)
            loss = cap_loss

            show_prediction(cap_probs, cap_gts[:, :-1], loader.dataset.get_ix_to_word())

            loss.backward()
            optimizer.step_and_update_lr()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1)

            # update parameters
            cap_train_loss = cap_loss.item()
            torch.cuda.synchronize()

            non_pad_mask = cap_gts[:, 1:].ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            print('(epoch %d), cap_train_loss = %.6f, current step = %d, current lr = %.3E, cap_acc = %.3f,'
                  % (epoch, cap_train_loss/n_word, optimizer.n_current_steps,
                     optimizer._optimizer.param_groups[0]['lr'], cap_n_correct/n_word))

        if epoch % opt['save_checkpoint_every'] == 0:
            model_path = os.path.join(opt['checkpoint_path'], 'Model_c3d_2tsfm_%d.pth' % epoch)
            model_info_path = os.path.join(opt['checkpoint_path'], 'model_score3.txt')
            torch.save(model.state_dict(), model_path)

            print('model saved to %s' % model_path)
            with open(model_info_path, 'a') as f:
                f.write('model_%d, cap_loss: %.6f' % (epoch, cap_train_loss))


def main(opt):
    # mode = training|validation|testing
    dataset = YC2Dataset(opt, 'training')
    dataloader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=True, collate_fn=yc2_collate_fn)

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
        c3d_path = opt['c3d_path'])

    model = nn.DataParallel(model).cuda()

    # if opt['load_checkpoint']:
    #     state_dict = torch.load(opt['load_checkpoint'])
    #     model.load_state_dict(state_dict)

    optimizer = ScheduledOptim(optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                          betas=(0.9, 0.98), eps=1e-09), 512, opt['warm_up_steps'])

    train(dataloader, model, optimizer, opt)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    opt = parse_opt()
    opt = vars(opt)
    main(opt)