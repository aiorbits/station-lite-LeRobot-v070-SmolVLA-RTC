#!/usr/bin/env python3
"""Paired offline plain/RTC replay of a saved Station Lite handoff; no robot I/O."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.rtc import RTCConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

TASK = 'Pick up the block and place it in the target zone.'


def metrics(a, prefix, delay, horizon):
    xyz = a[:, 7:10]
    played = xyz[delay:]
    step = np.linalg.norm(np.diff(played, axis=0), axis=1) * 1000
    overlap = np.linalg.norm(xyz[:horizon] - prefix[:horizon, 7:10], axis=1) * 1000
    dz = np.diff(played[:, 2]) * 1000
    return dict(
        prefix_xyz_mean_mm=float(overlap.mean()),
        committed_xyz_max_mm=float(np.linalg.norm(xyz[:delay] - prefix[:delay, 7:10], axis=1).max()*1000),
        first_available_target_error_mm=float(np.linalg.norm(xyz[delay]-prefix[delay, 7:10])*1000),
        handoff_step_mm=float(np.linalg.norm(xyz[delay]-prefix[delay-1, 7:10])*1000),
        future_max_step_mm=float(step.max()), future_steps_over_15mm=int((step>15).sum()),
        future_large_z_reversals=int(((dz[:-1]*dz[1:]<0)&(abs(dz[:-1])>15)&(abs(dz[1:])>15)).sum()),
        future_path_mm=float(step.sum()), future_net_mm=float(np.linalg.norm(played[-1]-played[0])*1000),
        gripper_min_mm=float(a[:,13].min()*1000), gripper_max_mm=float(a[:,13].max()*1000),
    )


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--snapshots',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--seeds',nargs='+',type=int,default=[43,44,45])
    p.add_argument('--phase-indices',nargs='+',type=int,default=[10])
    p.add_argument('--delay',type=int,default=2)
    args=p.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=True)
    snapshot=json.loads((args.snapshots/'loop_01_state.json').read_text())
    old=np.asarray(snapshot['previous_action_segment'],dtype=np.float32)
    joints=np.asarray(snapshot['previous_right_waypoints'])
    nearest=int(np.argmin(np.max(abs(joints-np.asarray(snapshot['state'])[7:13]),axis=1)))
    obs={'observation.state':torch.tensor(snapshot['state'],dtype=torch.float32),'task':TASK}
    for camera in ('cam_high','cam_left_wrist','cam_right_wrist'):
        with Image.open(args.snapshots/f'loop_01_{camera}.jpg') as im:
            rgb=np.asarray(im.convert('RGB'),dtype=np.uint8).copy()
        obs[f'observation.images.{camera}']=torch.from_numpy(rgb).permute(2,0,1).float()/255
    cfg=SmolVLAConfig.from_pretrained(str(args.checkpoint))
    cfg.device=args.device
    cfg.rtc_config=None
    policy=SmolVLAPolicy.from_pretrained(str(args.checkpoint),config=cfg).to(args.device).eval()
    pre,post=make_pre_post_processors(cfg,pretrained_path=str(args.checkpoint),
        preprocessor_overrides={'device_processor':{'device':args.device}})
    batch=pre(obs)
    configs=[('plain',0,0),('rtc_h8_w5',8,5),('rtc_h8_w10',8,10),
             ('rtc_h12_w5',12,5),('rtc_h12_w10',12,10),('rtc_h16_w10',16,10)]
    rows=[]
    metadata={'device':args.device,'checkpoint':str(args.checkpoint.resolve()),
        'snapshots':str(args.snapshots.resolve()),'nearest_saved_joint_index':nearest,
        'delay_steps':args.delay,'seeds':args.seeds,'phase_indices':args.phase_indices,
        'timing_note':('GPU policy+postprocess latency; excludes preprocessing, network and IK.' if args.device=='cuda'
                       else 'CPU replay timings are not deployment latency; delay is a replay assumption.'),
        'scope':'One recorded handoff, paired identical noise; no IK or robot execution.'}
    # Warmup plain inference, excluded from timing.
    policy.predict_action_chunk(batch)
    for phase in args.phase_indices:
        prefix=old[phase:]
        if not 1<=args.delay<8 or len(prefix)<16:
            raise ValueError('Insufficient recorded prefix for this comparison')
        norm_prefix=pre({**obs,'action':torch.from_numpy(prefix)})['action']
        roundtrip=post(norm_prefix).detach().cpu().numpy().reshape(-1,14)
        np.testing.assert_allclose(roundtrip,prefix,atol=1e-6,rtol=1e-5)
        np.save(args.output/f'phase{phase}_prefix.npy',prefix)
        for seed in args.seeds:
            torch.manual_seed(seed)
            noise=torch.randn((1,cfg.chunk_size,cfg.max_action_dim),device=args.device)
            plain=None
            for name,h,w in configs:
                policy.config.rtc_config=(None if not h else RTCConfig(enabled=True,mode='guided',
                    execution_horizon=h,max_guidance_weight=w,prefix_attention_schedule=RTCAttentionSchedule.EXP))
                policy.init_rtc_processor()
                policy.reset()
                if args.device=='cuda': torch.cuda.synchronize()
                start=time.perf_counter()
                pred=policy.predict_action_chunk(batch,noise=noise.clone(),
                    prev_chunk_left_over=norm_prefix if h else None,
                    inference_delay=args.delay,execution_horizon=h or 10)
                a=post(pred).detach().cpu().numpy()[0]
                if args.device=='cuda': torch.cuda.synchronize()
                elapsed=(time.perf_counter()-start)*1000
                if a.shape!=(50,14) or not np.isfinite(a).all(): raise ValueError('Invalid prediction')
                if plain is None: plain=a.copy()
                row=dict(phase=phase,seed=seed,method=name,horizon=h,weight=w,latency_ms=elapsed,
                    **metrics(a,prefix,args.delay,10),
                    max_xyz_change_from_plain_mm=float(np.linalg.norm(a[:,7:10]-plain[:,7:10],axis=1).max()*1000))
                np.save(args.output/f'phase{phase}_seed{seed}_{name}.npy',a)
                rows.append(row)
                (args.output/'results.json').write_text(json.dumps({'metadata':metadata,'rows':rows},indent=2)+'\n')
                print(json.dumps(row),flush=True)
            # RTC with no prefix must reproduce ordinary inference with identical noise.
            control=post(policy.predict_action_chunk(batch,noise=noise.clone(),prev_chunk_left_over=None)).detach().cpu().numpy()[0]
            np.testing.assert_allclose(control,plain,atol=1e-6,rtol=1e-5)
    metadata['checks_passed']=['prefix normalization roundtrip','finite 50x14 outputs','no-prefix RTC equals plain for each seed']
    (args.output/'results.json').write_text(json.dumps({'metadata':metadata,'rows':rows},indent=2)+'\n')


if __name__=='__main__': main()
