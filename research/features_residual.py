"""Amounts relative to the type x direction average: an estimate of each customer's amount level."""
import pandas as pd, numpy as np
def extra5(tx, sig, stats=None):
    t=tx.merge(sig[['signal_id','signal_sanasi']],on='signal_id')
    t['dd']=(t.signal_sanasi-t.tranzaksiya_vaqti).dt.total_seconds()/86400
    t['td']=t.tranzaksiya_turi+'_'+t.kirim_chiqim
    t['hb']=(t.dd>1)
    if stats is None:
        h=t[t.hb]; stats=h.groupby('td').miqdor_indeksi.agg(['mean','std','median'])
    t=t.join(stats,on='td'); t['r']=t.miqdor_indeksi-t['mean']; t['z']=t.r/t['std']
    t['nf']=t.miqdor_indeksi>-2.9   # exclude floor
    F={}
    h=t[t.hb & t.nf]; g=h.groupby('signal_id')
    F['res_mean']=g.r.mean(); F['res_med']=g.r.median(); F['z_mean']=g.z.mean(); F['z_std']=g.z.std()
    F['res_trim']=g.r.apply(lambda v: v.clip(v.quantile(.1),v.quantile(.9)).mean())
    for q in [.1,.25,.75,.9]: F[f'z_q{q}']=g.z.quantile(q)
    for w in [7,30,90]:
        s=h[h.dd<=w].groupby('signal_id'); F[f'z_mean_{w}']=s.z.mean()
    F['z_mean_old']=h[h.dd>90].groupby('signal_id').z.mean()
    for tt in ['bank_otkazmasi','karta','naqd']:
        s=h[h.tranzaksiya_turi==tt].groupby('signal_id'); F[f'z_mean_{tt}']=s.z.mean(); F[f'z_n_{tt}']=s.size()
    for dr in ['kirim','chiqim']:
        F[f'z_mean_{dr}']=h[h.kirim_chiqim==dr].groupby('signal_id').z.mean()
    F['z_in_minus_out']=F['z_mean_kirim']-F['z_mean_chiqim']
    b=t[(t.dd>0)&(t.dd<=1)&t.nf].groupby('signal_id'); F['z_mean_burst']=b.z.mean()
    F['z_burst_minus_hist']=F['z_mean_burst']-F['z_mean']
    # shrunken per-customer estimate (empirical Bayes) of z mean
    n=g.size(); F['z_shrunk']=F['z_mean']*n/(n+20)
    return pd.DataFrame(F).reindex(sig.signal_id), stats
