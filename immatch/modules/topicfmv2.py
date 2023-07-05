from argparse import Namespace
import torch
import numpy as np
import cv2

from third_party.topicfmv2.src.models import TopicFM as TopicFMv2_
from third_party.topicfmv2.src import get_model_cfg
from .base import Matching
from immatch.utils.data_io import load_gray_scale_tensor_cv


class TopicFMv2(Matching):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)

        self.imsize = args.imsize
        self.scale_type = max if args.dim_resized == 'max' else min
        self.match_threshold = args.match_threshold
        self.no_match_upscale = args.no_match_upscale
        self.max_n_matches = args.max_n_matches if args.max_n_matches > 0 else None

        # Load model
        model_variant = args.variant
        conf = dict(get_model_cfg())
        conf['match_coarse']['thr'] = self.match_threshold
        conf['match_coarse']['border_rm'] = args.match_border_rm
        coarse_model_cfg = args.coarse_model_cfg[model_variant]
        for k, v in coarse_model_cfg.items():
            conf["coarse"][k] = v
        if (model_variant == "plus") and "n_sampling_topics" in args:
            conf['coarse']['n_samples'] = args.n_sampling_topics

        print(conf)
        self.model = TopicFMv2_(config=conf)
        ckpt_path = args.ckpt[model_variant]
        ckpt_dict = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt_dict['state_dict'])
        self.model = self.model.eval().to(self.device)

        # Name the method
        # self.ckpt_name = ckpt_path.split('/')[-1].split('.')[0]
        self.name = f'TopicFM_{model_variant}'
        if self.no_match_upscale:
            self.name += '_noms'
        print(f'Initialize {self.name}')

    def load_im(self, im_path):
        return load_gray_scale_tensor_cv(
            im_path, self.device, imsize=self.imsize, dfactor=16, scale_type=self.scale_type
        )

    def match_inputs_(self, gray1, gray2):
        batch = {'image0': gray1, 'image1': gray2}
        with torch.no_grad():
            self.model(batch)
        if self.max_n_matches is not None:
            sorted_ids = torch.argsort(batch["mconf"], descending=True)[:self.max_n_matches]
            kpts1 = batch['mkpts0_f'][sorted_ids, :].cpu().numpy()
            kpts2 = batch['mkpts1_f'][sorted_ids, :].cpu().numpy()
            scores = batch['mconf'][sorted_ids].cpu().numpy()
        else:
            kpts1 = batch['mkpts0_f'].cpu().numpy()
            kpts2 = batch['mkpts1_f'].cpu().numpy()
            scores = batch['mconf'].cpu().numpy()
        matches = np.concatenate([kpts1, kpts2], axis=1)
        return matches, kpts1, kpts2, scores

    def match_pairs(self, im1_path, im2_path):
        gray1, sc1 = self.load_im(im1_path)
        gray2, sc2 = self.load_im(im2_path)

        upscale = np.array([sc1 + sc2])
        matches, kpts1, kpts2, scores = self.match_inputs_(gray1, gray2)

        if self.no_match_upscale:
            return matches, kpts1, kpts2, scores, upscale.squeeze(0)

        # Upscale matches &  kpts
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2
        return matches, kpts1, kpts2, scores
