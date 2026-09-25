"""Windowed per-type amount quantiles, robust spread and money volume (history only)."""
import pandas as pd, numpy as np
def extra3(tx, sig):
    t=tx.merge(sig[['signal_id','signal_sanasi']],on='signal_id')
    t['dd']=(t.signal_sanasi-t.tranzaksiya_vaqti).dt.total_seconds()/86400
    t=t[t.dd>1]  # history only, burst removed
    t['td']=t.tranzaksiya_turi+'_'+t.kirim_chiqim
    a=t.miqdor_indeksi
    out=[]
    for w in [30,90]:
        s=t[t.dd<=w]
        q=s.groupby(['signal_id','td']).miqdor_indeksi.quantile([.25,.5,.75]).unstack([1,2]); q.columns=[f'w{w}_q_{x}_{p}' for x,p in q.columns]; out.append(q)
    s=t[t.dd>90]
    q=s.groupby(['signal_id','td']).miqdor_indeksi.quantile([.5]).unstack([1,2]); q.columns=[f'old_q_{x}_{p}' for x,p in q.columns]; out.append(q)
    # share of transactions per type above/below global cut points (computed on each type's pooled distribution)
    for tt,cuts in {'bank_otkazmasi':[-1.0,0.0,1.0,2.0],'karta':[-1.5,-0.5,0.5]}.items():
        s=t[t.tranzaksiya_turi==tt]; g=s.groupby('signal_id').miqdor_indeksi
        for c in cuts: out.append(g.apply(lambda v,c=c:(v>c).mean()).rename(f'frac_{tt}_gt_{c}'))
    # trimmed means / robust stats per type
    g=t.groupby(['signal_id','tranzaksiya_turi']).miqdor_indeksi
    iqr=(g.quantile(.75)-g.quantile(.25)).unstack().add_prefix('iqr_'); out.append(iqr)
    mad=g.apply(lambda v:(v-v.median()).abs().median()).unstack().add_prefix('mad_'); out.append(mad)
    # counts per type x direction per month (volume level)
    n=t.groupby(['signal_id','td']).size().unstack().div(179).add_prefix('rate_'); out.append(n)
    # log-sum of exp amounts per td (volume in money terms)
    t['e']=np.exp(a)
    v=np.log1p(t.groupby(['signal_id','td']).e.sum().unstack()).add_prefix('logvol_'); out.append(v)
    return pd.concat(out,axis=1).reindex(sig.signal_id)
