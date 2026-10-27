from pathlib import Path
import numpy as np, pandas as pd, yaml
ROOT = Path('runs/controlled_evaluation')
meta = pd.read_parquet(ROOT/'processed/metadata.parquet')
print('site,fold,n,safe_rate,trainfit_low,trainfit_high,raw_IF_q01,q50,q99,clip_low_pct,clip_high_pct,mean_valid_coverage')
for site, g in meta.groupby('site_id', sort=True):
    fold = int(g.fold.iloc[0])
    cfg = yaml.safe_load((ROOT/'configs'/f'fold_{fold}_seed_42.yaml').read_text())
    lo, hi = (float(cfg['input_normalization'][k]) for k in ('if_low','if_high'))
    rng = np.random.default_rng(20261019 + fold)
    sampled=[]; low=high=valid_n=0
    for _, row in g.iterrows():
        tile = np.load(row.shard_path, mmap_mode='r')[int(row.shard_idx)]
        image=np.asarray(tile[0],dtype=np.float32).ravel(); valid=np.asarray(tile[2]>0).ravel(); v=image[valid]
        valid_n += len(v); low += int(np.count_nonzero(v <= lo)); high += int(np.count_nonzero(v >= hi))
        ids=rng.choice(len(v),min(64,len(v)),replace=False); sampled.append(v[ids])
    vals=np.concatenate(sampled); q=np.quantile(vals,[.01,.5,.99])
    print(f'{site},{fold},{len(g)},{g.safe_label.mean():.6f},{lo:.8f},{hi:.8f},{q[0]:.8f},{q[1]:.8f},{q[2]:.8f},{100*low/valid_n:.4f},{100*high/valid_n:.4f},{g.valid_coverage.mean():.6f}')
print('\nmodel,site,n,safe_prevalence,pred_safe_coverage,TP,FP,TN,FN,risk_min,risk_q01,risk_q50,risk_q99,risk_max')
for model, folder in [('CNN','model1_resnet18'),('CNN_QCNN','model2_qcnn')]:
    for fold in range(5):
        d=pd.read_parquet(ROOT/'runs'/folder/f'fold_{fold}_seed_42/test_predictions.parquet')
        y=d.true_safe.to_numpy(dtype=int); slope=d.pred_mean_slope_5m.to_numpy(float); hazard=d.pred_hazard_fraction.to_numpy(float)
        pred=((slope < 8.) & (hazard < .05)).astype(int)
        assert np.array_equal(pred,d.pred_safe.to_numpy(int))
        risk=np.maximum(slope/8.,hazard/.05)
        cm=[int(((pred==1)&(y==1)).sum()),int(((pred==1)&(y==0)).sum()),int(((pred==0)&(y==0)).sum()),int(((pred==0)&(y==1)).sum())]
        q=np.quantile(risk,[0,.01,.5,.99,1]); site=meta.loc[meta.fold==fold,'site_id'].iloc[0]
        print(f'{model},{site},{len(d)},{y.mean():.6f},{pred.mean():.6f},{cm[0]},{cm[1]},{cm[2]},{cm[3]},'+','.join(f'{v:.6f}' for v in q))
