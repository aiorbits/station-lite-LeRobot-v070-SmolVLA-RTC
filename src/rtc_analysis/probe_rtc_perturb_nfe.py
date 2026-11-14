#!/usr/bin/env python3
"""Controlled single-handoff prefix perturbation and Euler NFE diagnostics."""
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

TASK='Pick up the block and place it in the target zone.'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--snapshots',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    args.output.mkdir(parents=True,exist_ok=True)
    snapshot=json.loads((args.snapshots/'loop_01_state.json').read_text())
    obs={'observation.state':torch.tensor(snapshot['state'],dtype=torch.float32),'task':TASK}
    for camera in ('cam_high','cam_left_wrist','cam_right_wrist'):
        with Image.open(args.snapshots/f'loop_01_{camera}.jpg') as im:
            rgb=np.asarray(im.convert('RGB'),dtype=np.uint8).copy()
        obs['observation.images.'+camera]=torch.from_numpy(rgb).permute(2,0,1).float()/255
    cfg=SmolVLAConfig.from_pretrained(str(args.checkpoint))
    if cfg.use_delta_joint_actions_aloha or cfg.adapt_to_pi_aloha:raise ValueError('Requires absolute pose14 checkpoint')
    cfg.device='cuda';cfg.rtc_config=RTCConfig(enabled=True,mode='guided',execution_horizon=8,
       max_guidance_weight=10,prefix_attention_schedule=RTCAttentionSchedule.EXP)
    policy=SmolVLAPolicy.from_pretrained(str(args.checkpoint),config=cfg).to('cuda').eval()
    pre,post=make_pre_post_processors(cfg,pretrained_path=str(args.checkpoint))
    batch=pre(obs)
    old=np.asarray(snapshot['previous_action_segment'],dtype=np.float32)[10:]
    if old.shape!=(20,14):raise ValueError(old.shape)
    variants={'baseline':old}
    for shape in ('impulse','step'):
      for position in (0,4,7):
        x=old.copy()
        if shape=='impulse':x[position,9]+=.03
        else:x[position:,9]+=.03
        variants[f'{shape}_{position}']=x
    normal={name:pre({**obs,'action':torch.from_numpy(x)})['action'] for name,x in variants.items()}
    rows=[];raw={}
    def predict(seed,method,nfe,prefix):
        cfg.num_steps=nfe
        cfg.rtc_config=None if method=='plain' else RTCConfig(enabled=True,mode='guided',execution_horizon=8,
           max_guidance_weight=10,prefix_attention_schedule=RTCAttentionSchedule.EXP)
        policy.init_rtc_processor();policy.reset()
        torch.manual_seed(seed)
        noise=torch.randn(1,cfg.chunk_size,cfg.max_action_dim,device='cuda')
        torch.cuda.synchronize();start=time.perf_counter()
        normalized=policy.predict_action_chunk(batch,noise=noise,
            prev_chunk_left_over=prefix if method=='rtc' else None,inference_delay=2,execution_horizon=8)
        decoded=post(normalized)[0].detach().cpu().numpy()
        torch.cuda.synchronize()
        return decoded,(time.perf_counter()-start)*1000
    for seed in (43,44,45):
        baseline,base_ms=predict(seed,'rtc',10,normal['baseline'])
        raw[(seed,'rtc',10)]=baseline
        for label,prefix in normal.items():
            if label=='baseline':candidate,ms=baseline,base_ms
            else:candidate,ms=predict(seed,'rtc',10,prefix)
            delta=np.linalg.norm(candidate[:,7:10]-baseline[:,7:10],axis=1)*1000
            rows.append(dict(experiment='prefix_perturbation',seed=seed,label=label,
                prefix_change_type=label.split('_')[0],prefix_change_position=(int(label.split('_')[1]) if '_' in label else None),
                latency_ms=ms,delta_overlap_max_mm=float(delta[:8].max()),
                delta_nonoverlap_max_mm=float(delta[8:].max()),
                delta_nonoverlap_mean_mm=float(delta[8:].mean()),
                future_max_step_mm=float((np.linalg.norm(np.diff(candidate[2:,7:10],axis=0),axis=1)*1000).max())))
            np.save(args.output/f'seed{seed}_{label}_rtc_nfe10.npy',candidate)
        for method in ('plain','rtc'):
            for nfe in (10,20,40,80):
                if method=='rtc' and nfe==10:out,ms=baseline,base_ms
                else:out,ms=predict(seed,method,nfe,normal['baseline'])
                raw[(seed,method,nfe)]=out
                np.save(args.output/f'seed{seed}_{method}_nfe{nfe}.npy',out)
                step=np.linalg.norm(np.diff(out[2:,7:10],axis=0),axis=1)*1000
                rows.append(dict(experiment='nfe',seed=seed,method=method,nfe=nfe,
                    latency_ms=ms,future_max_step_mm=float(step.max()),
                    future_peak_index=int(np.argmax(step)+2),
                    future_steps_over_15mm=int((step>15).sum())))
        (args.output/'results_partial.json').write_text(json.dumps(rows,indent=2)+'\n')
    for row in rows:
        if row['experiment']!='nfe':continue
        output=raw[(row['seed'],row['method'],row['nfe'])]
        ref=raw[(row['seed'],row['method'],80)]
        dev=np.linalg.norm(output[:,7:10]-ref[:,7:10],axis=1)*1000
        row['vs_80_nfe_max_xyz_mm']=float(dev.max())
        row['vs_80_nfe_mean_xyz_mm']=float(dev.mean())
    (args.output/'results.json').write_text(json.dumps(dict(metadata=dict(
        checkpoint=str(args.checkpoint),observation='recorded live loop_01',
        phase_index=10,inference_delay=2,rtc_horizon=8,rtc_weight=10,
        impulse_or_step_offset_m=0.03,seed_count=3,
        nfe_reference=80,nfe_reference_is_convergence_proof=False,
        scope='single-handoff sensitivity; no cascading, IK, or robot control'),rows=rows),indent=2)+'\n')
    print('completed',len(rows),'rows',flush=True)

if __name__=='__main__':main()
