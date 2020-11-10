import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    # ------------------------------------------------ Data Path -------------------------------------------------------

    parser.add_argument(
        '--ann_path',
        type=str,
        default='./data/annotations/',
        help='Path of YouCook2 Dataset Annotations.')

    parser.add_argument(
        '--feat_path',
        type=str,
        default='./data/features/feat_csv/',
        help='Path of YouCook2 Dataset Features.')

    # parser.add_argument(
    #     '--c3d_feat_path',
    #     type=str,
    #     default='./data/c3d_features/',
    #     help='Path of YouCook2 Dataset event video S3D Features.')
    #
    # parser.add_argument(
    #     '--s3d_feat_path',
    #     default='./data/howto100m_s3d_feature_hub.pkl',
    #     type=str,
    #     help='Path of YouCook2 Dataset event video C3D Features.')
    #
    # parser.add_argument(
    #     '--event_video_path',
    #     type=str,
    #     default='./data/224reduced_event_videos',
    #     help='Path of YouCook2 Dataset raw event video.')

    parser.add_argument(
        '--cap_file',
        type=str,
        # default='youcookii_annotations_trainval.json',
        default='pseudo_captions.json',
        help='Caption files.')

    parser.add_argument(
        '--vocab_file',
        type=str,
        default='./data/annotations/yc2_vocab.pkl',
        help='Vocabulary diction files.')

    # parser.add_argument(
    #     '--c3d_path',
    #     type=str,
    #     # default='/home/ubuntu/ECCV20_Video_Pretraining/pre-trained/c3d-pretrained.pth',
    #     default='./pre-trained/s3d_nce_pretrained.pth',
    #     help='Pre-trained C3D weights.')
    #
    # parser.add_argument(
    #     '--language_feat_path',
    #     type=str,
    #     default='./data/language_features.pkl',
    #     help='Extracted language features.')

    # ------------------------------------------------ Model Setting ---------------------------------------------------

    parser.add_argument(
        '--max_length',
        type=int,
        default=46+2,
        help='Caption files.')

    parser.add_argument(
        '--feature_mode',
        type=str,
        default='ResNet',
        help='Model of feature: ResNet|BN-Inception.')

    parser.add_argument(
        '--input_dropout_p',
        type=float,
        default=0.4,
        help='strength of dropout in the Language Model RNN')

    parser.add_argument(
        '--dim_word',
        type=int,
        default=768,
        help='the encoding size of each token in the vocabulary, and the video.')

    parser.add_argument(
        '--dim_model',
        type=int,
        default=768,
        help='size of the rnn hidden layer')

    parser.add_argument(
        '--dim_language',
        type=int,
        default=768,
        help='dim of language feature from GPT')

    # 12-12 8 6 4
    parser.add_argument(
        '--num_head',
        type=int,
        default=8,
        help='Numbers of head in transformers.')

    parser.add_argument(
        '--num_layer',
        type=int,
        default=2,
        help='Numbers of layers in transformers.')

    parser.add_argument(
        '--dim_head',
        type=int,
        default=48,
        help='Dimension of the attention head.')

    parser.add_argument(
        '--dim_inner',
        type=int,
        default=1024,
        help='Dimension of inner feature in Encoder/Decoder.')

    # Optimization: General
    parser.add_argument(
        '--epochs',
        type=int,
        default=2000,
        help='number of epochs')

    parser.add_argument(
        '--warm_up_steps',
        type=int,
        default=500,
        help='Warm up steps.')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=42,
        help='mini-batch size')

    # -----------------------------------------------Checkpoint Setting-------------------------------------------------

    parser.add_argument(
        '--save_checkpoint_every',
        type=int,
        default=50,
        help='how often to save a model checkpoint (in epoch)?')

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='save',
        help='directory to store check pointed models')

    parser.add_argument(
        '--load_checkpoint',
        type=str,
        default='./save/Model_c3d_2tsfm_800.pth',
        help='directory to load check pointed models')

    parser.add_argument(
        '--gpu', type=str, default='0', help='gpu device number')

    args = parser.parse_args()

    return args
