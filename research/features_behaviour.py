"""Floor/cap hits, burst fine structure, weekly rhythm, hour entropy, in/out transitions."""
import pandas as pd, numpy as np
def extra4(tx, sig):
    t=tx.merge(sig[['signal_id','signal_sanasi']],on='signal_id').sort_values(['signal_id','tranzaksiya_vaqti'])
    t['dd']=(t.signal_sanasi-t.tranzaksiya_vaqti).dt.total_seconds()/86400
    a=t.miqdor_indeksi; sid=t.signal_id
    F={}
    # floor / cap hits
    F['n_floor']=(a<-2.9).groupby(sid).sum(); F['frac_floor']=(a<-2.9).groupby(sid).mean()
    F['n_cap']=(a.round(4).isin([4.1835,6.6917,6.4304,4.8628,4.563])).groupby(sid).sum()
    F['n_low_bank_in']=((a<-2.5)&(t.tranzaksiya_turi=='bank_otkazmasi')).groupby(sid).sum()
    # burst fine structure
    for lab,w in [('1m',1/1440),('10m',10/1440),('1h',1/24)]:
        b=t[(t.dd>0)&(t.dd<=w)]; g=b.groupby('signal_id')
        F[f'b{lab}_n']=g.size(); F[f'b{lab}_amt']=g.miqdor_indeksi.mean(); F[f'b{lab}_out']=(b.kirim_chiqim=='chiqim').groupby(b.signal_id).mean()
    b=t[(t.dd>0)&(t.dd<=1/24)].copy(); b['gap']=b.groupby('signal_id').tranzaksiya_vaqti.diff().dt.total_seconds()
    g=b.groupby('signal_id').gap; F['b_gap_mean']=g.mean(); F['b_gap_cv']=g.std()/g.mean(); F['b_gap_min']=g.min()
    F['b_first_min']=b.groupby('signal_id').dd.max()*1440
    F['b_amt_slope']=b.groupby('signal_id').apply(lambda d: np.polyfit(np.arange(len(d)),d.miqdor_indeksi.values,1)[0] if len(d)>2 else np.nan)
    F['b_ntypes']=b.groupby('signal_id').tranzaksiya_turi.nunique()
    # history (exclude last day)
    h=t[t.dd>1].copy(); hs=h.signal_id
    hr=h.tranzaksiya_vaqti.dt.hour; ph=pd.crosstab(hs,hr,normalize='index'); F['h_hour_ent']=-(ph*np.log(ph+1e-12)).sum(1); F['h_hour_maxshare']=ph.max(1)
    dw=pd.crosstab(hs,h.tranzaksiya_vaqti.dt.dayofweek,normalize='index'); F['h_dow_ent']=-(dw*np.log(dw+1e-12)).sum(1)
    dc=h.groupby([hs,np.floor(h.dd).astype(int)]).size().unstack(fill_value=0).reindex(columns=range(1,180),fill_value=0)
    v=dc.values.astype(float); vc=v-v.mean(1,keepdims=True); den=(vc**2).sum(1)+1e-9
    for lag in [1,7,30]: F[f'ac_{lag}']=pd.Series((vc[:,lag:]*vc[:,:-lag]).sum(1)/den,index=dc.index)
    F['day_fano']=pd.Series(v.var(1)/(v.mean(1)+1e-9),index=dc.index)
    # direction transitions
    d=(h.kirim_chiqim=='chiqim').astype(int); pd_=d.groupby(hs).shift()
    F['p_out_after_in']=((d==1)&(pd_==0)).groupby(hs).sum()/((pd_==0).groupby(hs).sum()+1e-9)
    F['p_out_after_out']=((d==1)&(pd_==1)).groupby(hs).sum()/((pd_==1).groupby(hs).sum()+1e-9)
    same=(h.tranzaksiya_turi==h.groupby(hs).tranzaksiya_turi.shift()); F['p_same_type']=same.groupby(hs).mean()
    # amount autocorrelation / consecutive amount diff
    pa=h.groupby(hs).miqdor_indeksi.shift(); F['amt_absdiff']=(h.miqdor_indeksi-pa).abs().groupby(hs).mean()
    F['amt_lag_corr']=pd.DataFrame({'a':h.miqdor_indeksi,'b':pa,'s':hs}).dropna().groupby('s').apply(lambda x: x.a.corr(x.b))
    # amount time-slope per type (history)
    for tt in ['karta','bank_otkazmasi']:
        s=h[h.tranzaksiya_turi==tt]
        F[f'slope_{tt}']=s.groupby('signal_id').apply(lambda x: np.polyfit(-x.dd.values,x.miqdor_indeksi.values,1)[0] if len(x)>5 else np.nan)
    return pd.DataFrame(F).reindex(sig.signal_id)
