"""Feature code copied from escalation_model.ipynb (section 3)."""
import numpy as np
import pandas as pd

WINDOWS = [1, 3, 7, 14, 30, 60, 90, 180]


def build_features(tx: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a signal's transaction history into one row of features."""
    t = tx.merge(sig[["signal_id", "signal_sanasi"]], on="signal_id")
    t["dd"] = (t.signal_sanasi - t.tranzaksiya_vaqti).dt.total_seconds() / 86400
    t["amt"] = t.miqdor_indeksi
    t["out"] = (t.kirim_chiqim == "chiqim").astype(np.int8)
    t["hour"] = t.tranzaksiya_vaqti.dt.hour
    t["night"] = t.hour.isin([0, 1, 2, 3, 4, 5]).astype(np.int8)
    t["wkend"] = (t.tranzaksiya_vaqti.dt.dayofweek >= 5).astype(np.int8)
    t["big"] = (t.amt > 2).astype(np.int8)
    t["signed"] = np.where(t.out == 1, -1, 1) * np.exp(t.amt)
    t["eamt"] = np.exp(t.amt)
    t = t.sort_values(["signal_id", "tranzaksiya_vaqti"])
    g = t.groupby("signal_id")
    F = pd.DataFrame(index=sig.signal_id)

    # overall
    F["n_all"] = g.size()
    F["n_after"] = t[t.dd <= 0].groupby("signal_id").size()
    F["span_days"] = (g.tranzaksiya_vaqti.max() - g.tranzaksiya_vaqti.min()).dt.total_seconds() / 86400
    F["first_dd"] = g.dd.max()
    F["last_dd"] = g.dd.min()
    for c in ["amt"]:
        a = g[c].agg(["mean", "std", "min", "max", "median", "skew"])
        a.columns = [f"{c}_{x}" for x in a.columns]
        F = F.join(a)
    q = g.amt.quantile([0.1, 0.9]).unstack()
    F["amt_q10"], F["amt_q90"] = q[0.1], q[0.9]
    F["out_frac"] = g.out.mean()
    F["night_frac"] = g.night.mean()
    F["wkend_frac"] = g.wkend.mean()
    F["big_frac"] = g.big.mean()
    F["hour_mean"] = g.hour.mean()
    F["hour_std"] = g.hour.std()
    F["net_flow"] = g.signed.sum()
    F["eamt_sum"] = g.eamt.sum()

    # type x direction
    t["td"] = t.tranzaksiya_turi + "_" + t.kirim_chiqim
    ct = t.pivot_table(index="signal_id", columns="td", values="amt", aggfunc=["size", "mean", "max"])
    ct.columns = [f"td_{a}_{b}" for a, b in ct.columns]
    F = F.join(ct)
    for c in [c for c in F.columns if c.startswith("td_size_")]:
        F[c.replace("size", "frac")] = F[c].fillna(0) / F.n_all

    # time windows
    for w in WINDOWS:
        s = t[(t.dd > 0) & (t.dd <= w)]
        gs = s.groupby("signal_id")
        F[f"n_{w}"] = gs.size()
        F[f"amt_mean_{w}"] = gs.amt.mean()
        F[f"amt_max_{w}"] = gs.amt.max()
        F[f"amt_std_{w}"] = gs.amt.std()
        F[f"eamt_{w}"] = gs.eamt.sum()
        F[f"net_{w}"] = gs.signed.sum()
        F[f"out_frac_{w}"] = gs.out.mean()
        F[f"n_types_{w}"] = gs.tranzaksiya_turi.nunique()
        for tt in ["naqd", "xalqaro", "bank_otkazmasi", "karta"]:
            F[f"n_{tt}_{w}"] = s[s.tranzaksiya_turi == tt].groupby("signal_id").size()
        F[f"n_{w}"] = F[f"n_{w}"].fillna(0)
    # ratios recent vs baseline (per-day rate)
    base_rate = (F.n_180 - F.n_30) / 150
    base_amt = t[(t.dd > 30)].groupby("signal_id").amt.mean()
    for w in [1, 3, 7, 14, 30]:
        F[f"rate_ratio_{w}"] = (F[f"n_{w}"] / w) / (base_rate + 1e-3)
        F[f"amt_diff_{w}"] = F[f"amt_mean_{w}"] - base_amt
    F["burst_n"] = t[(t.dd > 0) & (t.dd <= 1 / 24)].groupby("signal_id").size()
    F["burst_amt"] = t[(t.dd > 0) & (t.dd <= 1 / 24)].groupby("signal_id").amt.mean()

    # gaps between consecutive transactions
    t["gap"] = t.groupby("signal_id").tranzaksiya_vaqti.diff().dt.total_seconds() / 3600
    gg = t.groupby("signal_id").gap
    F["gap_mean"] = gg.mean()
    F["gap_med"] = gg.median()
    F["gap_min"] = gg.min()
    F["gap_max"] = gg.max()
    F["gap_std"] = gg.std()
    F["gap_lt1m"] = (t.gap < 1 / 60).groupby(t.signal_id).mean()

    # daily activity profile
    t["day"] = np.floor(t.dd).astype(int)
    d = t[t.dd > 1].groupby(["signal_id", "day"]).agg(n=("amt", "size"), e=("eamt", "sum"))
    dg = d.groupby("signal_id")
    F["active_days"] = dg.size()
    F["daily_n_mean"] = dg.n.mean()
    F["daily_n_max"] = dg.n.max()
    F["daily_n_std"] = dg.n.std()
    F["daily_e_max"] = dg.e.max()
    F["daily_e_std"] = dg.e.std()

    # monthly trend of counts and amounts
    t["mon"] = np.clip((t.dd // 30).astype(int), 0, 5)
    m = t[t.dd > 0].pivot_table(index="signal_id", columns="mon", values="amt", aggfunc=["size", "mean"])
    x = np.arange(6)[::-1]
    cnt = m["size"].reindex(columns=range(6)).fillna(0).values
    mam = m["mean"].reindex(columns=range(6)).values
    xc = x - x.mean()
    F["cnt_slope"] = (cnt - cnt.mean(1, keepdims=True)) @ xc / (xc @ xc)
    F["amt_slope"] = np.nan_to_num(mam - np.nanmean(mam, 1, keepdims=True)) @ xc / (xc @ xc)

    # amount repeat / round patterns
    F["amt_nuniq_frac"] = g.amt.nunique() / F.n_all
    F["sig_month"] = sig.set_index("signal_id").signal_sanasi.dt.month
    F["sig_dow"] = sig.set_index("signal_id").signal_sanasi.dt.dayofweek
    return F


def extra(tx, sig):
    t=tx.merge(sig[['signal_id','signal_sanasi']],on='signal_id')
    t['dd']=(t.signal_sanasi-t.tranzaksiya_vaqti).dt.total_seconds()/86400
    t['td']=t.tranzaksiya_turi+'_'+t.kirim_chiqim
    out=[]
    for key in ['td','tranzaksiya_turi','kirim_chiqim']:
        q=t.groupby(['signal_id',key]).miqdor_indeksi.quantile([.05,.25,.5,.75,.95]).unstack([1,2])
        q.columns=[f'q_{a}_{b}' for a,b in q.columns]; out.append(q)
        sd=t.groupby(['signal_id',key]).miqdor_indeksi.agg(['std','skew']).unstack(); sd.columns=[f'{a}_{b}' for a,b in sd.columns]; out.append(sd)
    # recent 30d vs older per type mean
    r=t[t.dd<=30].groupby(['signal_id','tranzaksiya_turi']).miqdor_indeksi.mean().unstack()
    o=t[t.dd>30].groupby(['signal_id','tranzaksiya_turi']).miqdor_indeksi.mean().unstack()
    out.append((r-o).add_prefix('recent_minus_old_'))
    rn=t[t.dd<=30].groupby(['signal_id','tranzaksiya_turi']).size().unstack()
    on=t[t.dd>30].groupby(['signal_id','tranzaksiya_turi']).size().unstack()
    out.append((rn/30/(on/150)).add_prefix('rate_ratio_type_'))
    return pd.concat(out,axis=1).reindex(sig.signal_id)

