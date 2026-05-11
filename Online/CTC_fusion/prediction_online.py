from logging import Logger
import warnings, wandb
import pickle
from google.protobuf.reflection import ParseMessage
from collections import defaultdict
from modelling.model import build_model
from modelling.recognition import ctc_decode_func
from utils.optimizer import build_optimizer, build_scheduler
warnings.filterwarnings("ignore")
import argparse
import numpy as np
import os, sys
import shutil
import queue
sys.path.append(os.getcwd())#slt dir
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP, distributed
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from utils.misc import (
    DATASETS,
    get_logger,
    load_config,
    log_cfg,
    load_checkpoint,
    make_model_dir,
    make_logger, make_writer,
    set_seed,
    symlink_update,
    is_main_process, init_DDP, move_to_device,
    neq_load_customized
)
from dataset.Dataloader import build_dataloader
from utils.progressbar import ProgressBar
from utils.metrics import bleu, rouge, wer_list, wer_list_per_sen
from utils.phoenix_cleanup import clean_phoenix_2014_trans, clean_phoenix_2014
from copy import deepcopy
from itertools import groupby
import math
import ctcdecode
from ctcdecode import CTCBeamDecoder, OnlineCTCBeamDecoder


def pad_tensor(x, pad_left, pad_right):
    assert x.ndim in [2, 3, 4]
    if pad_left > 0:
        if x.ndim == 4:
            pad = x[0].repeat(pad_left, 1, 1, 1)
        elif x.ndim == 3:
            pad = x[0].repeat(pad_left, 1, 1)
        elif x.ndim == 2:
            pad = x[0].repeat(pad_left, 1)
        x = torch.cat([pad, x], dim=0)

    if pad_right > 0:
        if x.ndim == 4:
            pad = x[-1].repeat(pad_right, 1, 1, 1)
        elif x.ndim == 3:
            pad = x[-1].repeat(pad_right, 1, 1)
        elif x.ndim == 2:
            pad = x[-1].repeat(pad_right, 1)
        x = torch.cat([x, pad], dim=0)
    return x


def _frame_motion_curve(keypoint, confidence_threshold=0.2, ema_decay=0.6):
    if keypoint.shape[0] <= 1:
        return [0.0] * max(int(keypoint.shape[0]), 1)

    coords = keypoint[..., :2].float()
    if keypoint.shape[-1] > 2:
        confidence = keypoint[..., 2].float()
    else:
        confidence = torch.ones(keypoint.shape[:2], device=keypoint.device, dtype=torch.float32)

    displacement = torch.linalg.norm(coords[1:] - coords[:-1], dim=-1)
    valid = (confidence[1:] > confidence_threshold) & (confidence[:-1] > confidence_threshold)

    smoothed_motion = []
    previous_motion = 0.0
    for frame_idx in range(displacement.shape[0]):
        current_motion = displacement[frame_idx][valid[frame_idx]]
        motion_value = current_motion.mean().item() if current_motion.numel() > 0 else previous_motion
        previous_motion = motion_value if frame_idx == 0 else ema_decay * previous_motion + (1 - ema_decay) * motion_value
        smoothed_motion.append(previous_motion)

    return [smoothed_motion[0] if smoothed_motion else 0.0] + smoothed_motion


def _adaptive_window_centers(
    keypoint,
    total_frames,
    base_stride,
    min_stride,
    max_stride,
    confidence_threshold,
    ema_decay,
    quantile_low,
    quantile_high,
):
    motion_curve = _frame_motion_curve(
        keypoint,
        confidence_threshold=confidence_threshold,
        ema_decay=ema_decay,
    )
    positive_motion = torch.tensor([value for value in motion_curve if value > 0], dtype=torch.float32)

    fallback_stride = int(max(min_stride, min(max_stride, base_stride)))
    if positive_motion.numel() == 0:
        low_motion = 0.0
        high_motion = 0.0
    else:
        low_motion = torch.quantile(positive_motion, quantile_low).item()
        high_motion = torch.quantile(positive_motion, quantile_high).item()

    centers = [0]
    current_center = 0
    while current_center < total_frames - 1:
        motion_value = motion_curve[min(current_center, len(motion_curve) - 1)]
        if high_motion <= low_motion + 1e-6:
            next_stride = fallback_stride
        else:
            normalized = (motion_value - low_motion) / (high_motion - low_motion)
            normalized = max(0.0, min(1.0, normalized))
            mapped_stride = max_stride - normalized * (max_stride - min_stride)
            next_stride = int(round(mapped_stride))
            next_stride = max(min_stride, min(max_stride, next_stride))

        next_center = min(current_center + next_stride, total_frames - 1)
        if next_center == current_center:
            break
        centers.append(next_center)
        current_center = next_center

    return centers


def sliding_windows(video, keypoint, win_size=16, stride=1, save_fea=False, adaptive_cfg=None):
    B, T = video.shape[:2]
    assert B==1
    video = video.squeeze(0)
    keypoint = keypoint.squeeze(0)

    adaptive_cfg = adaptive_cfg or {}
    adaptive_enabled = adaptive_cfg.get('enabled', False)

    if adaptive_enabled:
        min_stride = int(adaptive_cfg.get('min_stride', 1))
        max_stride = int(adaptive_cfg.get('max_stride', max(stride, 1)))
        max_stride = max(min_stride, max_stride)
        centers = _adaptive_window_centers(
            keypoint=keypoint,
            total_frames=T,
            base_stride=stride,
            min_stride=min_stride,
            max_stride=max_stride,
            confidence_threshold=float(adaptive_cfg.get('confidence_threshold', 0.2)),
            ema_decay=float(adaptive_cfg.get('ema_decay', 0.6)),
            quantile_low=float(adaptive_cfg.get('quantile_low', 0.3)),
            quantile_high=float(adaptive_cfg.get('quantile_high', 0.8)),
        )
        pad_left = win_size // 2
        pad_right = win_size - pad_left
        video = pad_tensor(video, pad_left, pad_right)
        keypoint = pad_tensor(keypoint, pad_left, pad_right)

        num_clips = len(centers)
        video_s = torch.zeros(num_clips, win_size, video.shape[1], video.shape[2], video.shape[3]).to(video.device)
        keypoint_s = torch.zeros(num_clips, win_size, keypoint.shape[1], keypoint.shape[2]).to(video.device)
        for i, center in enumerate(centers):
            video_s[i, ...] = video[center:center+win_size, ...]
            keypoint_s[i, ...] = keypoint[center:center+win_size, ...]

        del video; del keypoint
        return video_s, keypoint_s

    num_clips = math.ceil(T/stride)
    num_clips = max(num_clips, 1)
    final_frames = (num_clips-1) * stride + win_size
    pad_left = (final_frames - T) // 2
    pad_right = final_frames - T - pad_left
    
    #pad
    video = pad_tensor(video, pad_left, pad_right)
    keypoint = pad_tensor(keypoint, pad_left, pad_right)
    
    #make slided inputs
    video_s = torch.zeros(num_clips, win_size, video.shape[1], video.shape[2], video.shape[3]).to(video.device)  #N,W,C,H,W
    keypoint_s = torch.zeros(num_clips, win_size, keypoint.shape[1], keypoint.shape[2]).to(video.device)  #N,W,K,3
    for i in range(num_clips):
        st = i * stride
        video_s[i, ...] = video[st:st+win_size, ...]
        keypoint_s[i, ...] = keypoint[st:st+win_size, ...]
    
    del video; del keypoint
    return video_s, keypoint_s


def evaluation(model, val_dataloader, cfg, 
        tb_writer=None, wandb_run=None,
        epoch=None, global_step=None,
        generate_cfg={}, save_dir=None,
        do_translation=True, do_recognition=True, external_logits=None,
    winsize=16, stride=16, adaptive_cfg=None):  #to-do output_dir
    
    logger = get_logger()
    logger.info(generate_cfg)
    print()
    if external_logits is not None:
        with open(external_logits, 'rb') as f:
            external_logits = pickle.load(f)
    if os.environ.get('enable_pbar', '1')=='1':
        pbar = ProgressBar(n_total=len(val_dataloader), desc='Validation')
    else:
        pbar = None
    if epoch!=None:
        logger.info('Evaluation epoch={} validation examples #={}'.format(epoch, len(val_dataloader.dataset)))
    elif global_step!=None:
        logger.info('Evaluation global step={} validation examples #={}'.format(global_step, len(val_dataloader.dataset)))

    voc_size = len(model.gloss_tokenizer)
    ctc_decoder_vocab = [chr(x) for x in range(20000, 20000+voc_size)]
    ctc_decoder = OnlineCTCBeamDecoder(ctc_decoder_vocab,
                                beam_width=generate_cfg['recognition']['beam_size'],
                                blank_id=0,
                                num_processes=5,
                                log_probs_input=False)
    
    model.eval()
    dataset2results = defaultdict(lambda: defaultdict(dict))
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            #forward -- loss
            batch = move_to_device(batch, cfg['device'])
            datasetname = batch['datasetname']

            video_s, keypoint_s = sliding_windows(
                batch['recognition_inputs']['sgn_videos'],
                batch['recognition_inputs']['sgn_keypoints'],
                winsize,
                stride,
                adaptive_cfg=adaptive_cfg,
            )
            num_windows = video_s.shape[0]
            batch['recognition_inputs']['sgn_videos'] = video_s
            batch['recognition_inputs']['sgn_keypoints'] = keypoint_s
            batch['recognition_inputs']['sgn_lengths'] = torch.tensor([winsize]*num_windows).long().to(cfg['device'])
            eff_length = min(batch['recognition_inputs']['gls_lengths'].item(), winsize)
            batch['recognition_inputs']['gls_lengths'] = torch.tensor([eff_length]*num_windows).to(cfg['device'])
            batch['recognition_inputs']['gloss_labels'] = torch.cat([batch['recognition_inputs']['gloss_labels'][:eff_length]]*num_windows, dim=0)
            # print(video_s.shape, keypoint_s.shape)

            forward_output = model(is_train=False, **batch)

            #rgb/keypoint/fuse/ensemble_last_logits
            for k, gls_logits in forward_output.items():
                if not 'gloss_logits' in k or gls_logits==None:
                    continue

                logits_name = k.replace('gloss_logits','')
                if logits_name in ['rgb_','keypoint_','fuse_','ensemble_last_','ensemble_early_','']:
                    if logits_name=='ensemble_early_':
                        input_lengths = forward_output['aux_lengths']['rgb'][-1]
                    else:
                        input_lengths = forward_output['input_lengths']

                    gls_prob = gls_logits.softmax(dim=-1)
                    state = ctcdecode.DecoderState(ctc_decoder)
                    for i in range(num_windows):
                        win_prob = gls_prob[i].unsqueeze(0)  #1,T,N
                        if i < num_windows-1:
                            beam_results, beam_scores, timesteps, out_seq_len = ctc_decoder.decode(win_prob, [state], [False])
                        else:
                            beam_results, beam_scores, timesteps, out_seq_len = ctc_decoder.decode(win_prob, [state], [True])

                    ctc_decode_output = [[x[0] for x in groupby(beam_results[0][0][:out_seq_len[0][0]].tolist())]]
                    batch_pred_gls = model.gloss_tokenizer.convert_ids_to_tokens(ctc_decode_output, datasetname)    

                    # print(batch_pred_gls)
                    for name, gls_hyp, gls_ref in zip(batch['name'], batch_pred_gls, batch['gloss']):
                        dataset2results[datasetname][name][f'{logits_name}gls_hyp'] = \
                            ' '.join(gls_hyp).upper() if model.gloss_tokenizer.lower_case \
                                else ' '.join(gls_hyp)
                        dataset2results[datasetname][name]['gls_ref'] = gls_ref.upper() if model.gloss_tokenizer.lower_case \
                                else gls_ref
                else:
                    #print(logits_name)
                    raise ValueError

            #misc
            if pbar:
                pbar(step)
        print()

    #evaluation (Recognition:WER,  Translation:B/M)
    dataset2evaluation_results = {}
    for datasetname, results in dataset2results.items():
        evaluation_results = {}
        name = list(results.keys())[0]
        if do_recognition:
            evaluation_results['wer'] = 200
            for hyp_name in results[name].keys():
                if not 'gls_hyp' in hyp_name:
                    continue
                k = hyp_name.replace('gls_hyp','')
                if datasetname in ['phoenix', 'phoenix2014tsi']:
                    gls_ref = [clean_phoenix_2014_trans(results[n]['gls_ref']) for n in results]
                    gls_hyp = [clean_phoenix_2014_trans(results[n][hyp_name]) for n in results] 
                elif datasetname in ['phoenix2014', 'phoenix2014si', 'phoenixcomb']:
                    gls_ref = [clean_phoenix_2014(results[n]['gls_ref']) for n in results]
                    gls_hyp = [clean_phoenix_2014(results[n][hyp_name]) for n in results] 
                elif datasetname in ['csl','cslr','wlasl2000','tvb']:
                    gls_ref = [results[n]['gls_ref'] for n in results]
                    gls_hyp = [results[n][hyp_name] for n in results]
                # print(gls_hyp, gls_ref)
                wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)
                wer_results_per_sen = wer_list_per_sen(hypotheses=gls_hyp, references=gls_ref)
                evaluation_results[k+'wer_list'] = wer_results
                logger.info('{}-{}WER: {:.2f}'.format(datasetname, k, wer_results['wer']))
                logger.info('{}-{}WER_per_sen: {:.2f}'.format(datasetname, k, wer_results_per_sen['wer']))
                evaluation_results['wer'] = min(wer_results['wer'], evaluation_results['wer'])

        #save
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f'{datasetname}_results.pkl'),'wb') as f:
                pickle.dump(results, f)
            with open(os.path.join(save_dir, f'{datasetname}_evaluation_results.pkl'),'wb') as f:
                pickle.dump(evaluation_results, f)
        dataset2evaluation_results[datasetname] = evaluation_results
    return dataset2evaluation_results
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser("SLT baseline Testing")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--save_subdir",
        default='prediction_online',
        type=str
    )
    parser.add_argument(
        '--ckpt_name',
        default='best.ckpt',
        type=str
    )
    parser.add_argument(
        '--external_logits',
        default=None,
        type=str
    )
    parser.add_argument(
        '--winsize',
        default=16,
        type=int
    )
    parser.add_argument(
        '--stride',
        default=16,
        type=int
    )
    parser.add_argument(
        '--adaptive_stride',
        action='store_true'
    )
    parser.add_argument(
        '--adaptive_min_stride',
        default=1,
        type=int
    )
    parser.add_argument(
        '--adaptive_max_stride',
        default=4,
        type=int
    )
    parser.add_argument(
        '--adaptive_confidence_threshold',
        default=0.2,
        type=float
    )
    parser.add_argument(
        '--adaptive_ema_decay',
        default=0.6,
        type=float
    )
    parser.add_argument(
        '--adaptive_quantile_low',
        default=0.3,
        type=float
    )
    parser.add_argument(
        '--adaptive_quantile_high',
        default=0.8,
        type=float
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_dir = cfg['training']['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    global logger
    logger = make_logger(model_dir=model_dir, log_file='prediction.log')
    cfg['device'] = torch.device('cuda')
    model = build_model(cfg)
    do_translation, do_recognition = cfg['task']!='S2G', cfg['task']!='G2T' #(and recognition loss>0 if S2T)
    #load model

    adaptive_cfg = {
        'enabled': args.adaptive_stride,
        'min_stride': args.adaptive_min_stride,
        'max_stride': args.adaptive_max_stride,
        'confidence_threshold': args.adaptive_confidence_threshold,
        'ema_decay': args.adaptive_ema_decay,
        'quantile_low': args.adaptive_quantile_low,
        'quantile_high': args.adaptive_quantile_high,
    }

    #per-dataset
    for datasetname in cfg['datanames']:
        logger.info('Evaluate '+datasetname)
        load_model_path = os.path.join(model_dir,'ckpts',args.ckpt_name)
        if not os.path.isfile(load_model_path):
            load_model_path = os.path.join(model_dir,'ckpts',f'{datasetname}_{args.ckpt_name}')
        if not os.path.isfile(load_model_path):
            load_model_path = os.path.join(model_dir,'ckpts',f'phoenixcomb_{args.ckpt_name}')
        if os.path.isfile(load_model_path):
            state_dict = torch.load(load_model_path, map_location='cuda')
            neq_load_customized(model, state_dict['model_state'], verbose=True)
            epoch, global_step = state_dict.get('epoch',0), state_dict.get('global_step',0)
            logger.info('Load model ckpt from '+load_model_path)
        else:
            logger.info(f'{load_model_path} does not exist')
            epoch, global_step = 0, 0
        cfg_ = deepcopy(cfg)
        cfg_['datanames'] = [datasetname]
        cfg_['data'] = {k:v for k,v in cfg['data'].items() if not k in cfg['datanames'] or k==datasetname}
        for split in ['test', 'dev']:
            logger.info('Evaluate on {} set'.format(split))
            dataloader, sampler = build_dataloader(cfg_, split, model.text_tokenizer, model.gloss_tokenizer)
            run_name = f'{args.save_subdir}_winsize_{args.winsize}_stride_{args.stride}'
            if adaptive_cfg['enabled']:
                run_name += f'_adaptive_{adaptive_cfg["min_stride"]}_{adaptive_cfg["max_stride"]}'
            evaluation(model=model, val_dataloader=dataloader, cfg=cfg_, 
                    epoch=epoch, global_step=global_step, 
                    generate_cfg=cfg_['testing']['cfg'],
                    save_dir=os.path.join(model_dir, run_name, split),
                    do_translation=do_translation, do_recognition=do_recognition, external_logits=args.external_logits,
                    winsize=args.winsize, stride=args.stride, adaptive_cfg=adaptive_cfg)

