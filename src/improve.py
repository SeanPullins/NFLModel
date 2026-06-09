import pandas as pd, numpy as np, lightgbm as lgb, joblib
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
exec(open("pipeline.py").read().split("# ---- time-based evaluation")[0])

FEATS = FEATS_A
def make_baseline(train):
    isos={}
    glob = IsotonicRegression(out_of_bounds="clip").fit(-train.Pick, train.y)
    for p,g in train.groupby("pos_g",observed=True):
        if len(g)>=300: isos[p]=IsotonicRegression(out_of_bounds="clip").fit(-g.Pick,g.y)
    def base(part):
        out = glob.predict(-part.Pick)
        for p,iso in isos.items():
            m = part.pos_g==p
            if m.any(): out[m.values] = .5*out[m.values] + .5*iso.predict(-part.Pick[m])
        return out
    return base, glob, isos

def make_resid(train, base, seeds=(1,2,3,4,5)):
    r = train.y - base(train)
    ms=[]
    for s in seeds:
        m = lgb.LGBMRegressor(objective="regression", learning_rate=.02, num_leaves=15,
            min_data_in_leaf=80, feature_fraction=.7, bagging_fraction=.8, bagging_freq=1,
            lambda_l2=5.0, n_estimators=600, verbose=-1, random_state=s)
        m.fit(train[FEATS+CATS], r, categorical_feature=CATS); ms.append(m)
    return lambda part: np.mean([m.predict(part[FEATS+CATS]) for m in ms],0), ms

tr = df[df.Year<=2009].copy(); va = df[(df.Year>=2010)&(df.Year<=2011)].copy()
te = df[(df.Year>=2012)&(df.Year<=2014)].copy()
for p_ in (tr,va,te): p_["col_enc"]=college_enc(tr,p_)
base,_ ,_ = make_baseline(tr); resid,_ = make_resid(tr, base)

# per-position shrinkage tuned on validation (drafted)
vd = va[va.Pick<260].copy(); vd_b=base(vd); vd_r=resid(vd)
shr={}
for p,g in vd.groupby("pos_g",observed=True):
    i=vd.pos_g==p; cand=np.arange(0,1.05,.1)
    shr[p]=float(max(cand,key=lambda s:spearmanr(vd_b[i.values]+s*vd_r[i.values],g.y).statistic)) if len(g)>=40 else .4
print("per-pos shrink:",shr)

# final eval: train 2000-2011
tr2=df[df.Year<=2011].copy(); tr2["col_enc"]=college_enc(tr2,tr2); te["col_enc"]=college_enc(tr2,te)
base,_,_=make_baseline(tr2); resid,_=make_resid(tr2,base)
td=te[te.Pick<260].copy(); b=base(td); r=resid(td)
s_vec=td.pos_g.map(shr).fillna(.4).values
apex=b+s_vec*r
print("TEST 2012-14 | pick-only:",round(spearmanr(IsotonicRegression(out_of_bounds='clip').fit(-tr2.Pick,tr2.y).predict(-td.Pick),td.y).statistic,4),
      "| pos-iso base:",round(spearmanr(b,td.y).statistic,4),
      "| APEX v1.1:",round(spearmanr(apex,td.y).statistic,4),
      "| hitAUC:",round(roc_auc_score(td.hit,apex),4))
for p,g in td.groupby("pos_g",observed=True):
    if len(g)>=60:
        i=(td.pos_g==p).values
        print(f"{p:5s} APEX={spearmanr(apex[i],g.y).statistic:.3f}  base={spearmanr(b[i],g.y).statistic:.3f}  shr={shr.get(p,.4)}")

# production refit on all data
df["col_enc"]=college_enc(df,df)
baseF,glob,isos=make_baseline(df); residF,ms=make_resid(df,baseF)
bF=baseF(df); rF=residF(df)
df["apex"]=bF+df.pos_g.map(shr).fillna(.4).values*rF
df["exp_at_pick"]=bF; df["talent_resid"]=rF; df["surplus"]=df.apex-df.exp_at_pick
df[["Year","Player","Pos","pos_g","College" if "College" in df else "college","Pick","Rnd","CarAV","y","apex","exp_at_pick","talent_resid","surplus"]]\
  .rename(columns={"college":"College"}).round(4).to_csv("/home/claude/apex_board.csv",index=False)
for i,m in enumerate(ms): m.booster_.save_model(f"/home/claude/apex_resid_{i}.txt")
joblib.dump({"glob":glob,"isos":isos,"shrink":shr},"/home/claude/apex_baseline.pkl")
