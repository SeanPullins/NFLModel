"""APEX v2 — extended 2000-2024 outcomes + 2025/26 prospect classes."""
import pandas as pd, numpy as np, lightgbm as lgb, re, joblib
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

POSMAP={"HB":"RB","FB":"RB","S":"DB","FS":"DB","SS":"DB","CB":"DB","DB":"DB","OLB":"LB","ILB":"LB",
"MLB":"LB","LB":"LB","DE":"EDGE","EDGE":"EDGE","EDG":"EDGE","NT":"DT","DT":"DT","DL":"DT","C":"OL","G":"OL","OG":"OL",
"OT":"OL","T":"OL","OL":"OL","QB":"QB","WR":"WR","TE":"TE","RB":"RB","K":"ST","P":"ST","LS":"ST"}
def norm(s): return re.sub(r"[^a-z]","",str(s).lower())

p = pd.read_csv("ph/nfl-draft-dataset-main/nfl_data.csv", sep=";")
p = p[(p.year>=2000)&(p.year<=2024)].rename(columns={
 "name":"Player","40_yard":"dash","bench_press":"bench","vert_leap":"vert_leap",
 "broad_jump":"broad","3_cone":"cone","shuttle":"shuttle","draft_pick":"Pick",
 "career_av":"CarAV","pos":"Pos","year":"Year","college":"College","g":"G"})
p["pos_g"]=p.simple_pos.map(POSMAP).fillna(p.Pos.map(POSMAP)).fillna("OTH")
p["Pick"]=p.Pick.fillna(263); p["CarAV"]=p.CarAV.fillna(0)
p["Rnd"]=p.draft_round.fillna(8)

a = pd.read_csv("ac/nfl-draft-data-master/data/combine_pro_day.csv")
a = a[a.Year>=2025].rename(columns={"player":"Player","College":"College","POS_GP":"pos_g0",
 "Height (in)":"height","Weight (lbs)":"weight","40 Yard":"dash","Vert Leap (in)":"vert_leap",
 "Broad Jump (in)":"broad","3Cone":"cone","Shuttle":"shuttle","Bench Press":"bench"})
a["pos_g"]=a.pos_g0.map(POSMAP).fillna(a.POS.map(POSMAP)).fillna("OTH")
a["Pos"]=a.POS; a["Pick"]=np.nan; a["Rnd"]=np.nan; a["CarAV"]=np.nan; a["age"]=np.nan
cols=["Year","Player","Pos","pos_g","College","Pick","Rnd","CarAV","height","weight","dash","vert_leap","bench","broad","cone","shuttle"]
p["age"]=p.age
df=pd.concat([p[cols+["age"]], a[cols+["age"]]], ignore_index=True)
df=df[df.pos_g!="ST"].copy()
df["key"]=df.Player.map(norm)+"_"+df.Year.astype(str); df=df.drop_duplicates("key")

hist=df.Year<=2024
df.loc[hist,"y"]=df[hist].groupby("Year").CarAV.rank(pct=True)
df.loc[hist,"hit"]=(df[hist].groupby("Year").CarAV.rank(pct=True,ascending=False)<=.10).astype(float)

df["bmi"]=df.weight/(df.height**2)*703
df["speed_score"]=(df.weight*200)/(df.dash**4)
df["explosion"]=df.vert_leap+df.broad/12
df["agility"]=df.cone+df.shuttle
for c in ["dash","speed_score","explosion","agility","weight","height","bmi","bench","vert_leap","broad"]:
    mu=df.groupby("pos_g")[c].transform("mean"); sd=df.groupby("pos_g")[c].transform("std")
    df[c+"_z"]=(df[c]-mu)/sd
df["pos_g"]=df.pos_g.astype("category")
FEATS=[c+"_z" for c in ["dash","speed_score","explosion","agility","weight","height","bmi","bench","vert_leap","broad"]]+["age","col_enc"]
CATS=["pos_g"]
def college_enc(train, part, k=12):
    g=train.groupby("College").y.agg(["mean","count"]); pr=train.y.mean()
    enc=(g["mean"]*g["count"]+pr*k)/(g["count"]+k)
    return part.College.map(enc).fillna(pr)

def make_baseline(train):
    glob=IsotonicRegression(out_of_bounds="clip").fit(-train.Pick,train.y); isos={}
    for pg,g in train.groupby("pos_g",observed=True):
        if len(g)>=300: isos[pg]=IsotonicRegression(out_of_bounds="clip").fit(-g.Pick,g.y)
    def base(part):
        out=glob.predict(-part.Pick.fillna(263))
        for pg,iso in isos.items():
            m=(part.pos_g==pg).values
            if m.any(): out[m]=.5*out[m]+.5*iso.predict(-part.Pick.fillna(263)[m])
        return out
    return base,glob,isos
def make_resid(train,base):
    r=train.y-base(train); ms=[]
    for s in range(5):
        m=lgb.LGBMRegressor(objective="regression",learning_rate=.02,num_leaves=15,min_data_in_leaf=80,
          feature_fraction=.7,bagging_fraction=.8,bagging_freq=1,lambda_l2=5.0,n_estimators=600,verbose=-1,random_state=s)
        m.fit(train[FEATS+CATS],r,categorical_feature=CATS); ms.append(m)
    return (lambda part: np.mean([m.predict(part[FEATS+CATS]) for m in ms],0)), ms

# eval: train<=2013, tune 2014-15, test 2016-2019 (all 5+ seasons of outcomes)
tr=df[df.Year<=2013].copy(); va=df[df.Year.isin([2014,2015])].copy(); te=df[(df.Year>=2016)&(df.Year<=2019)].copy()
for x in (tr,va,te): x["col_enc"]=college_enc(tr,x)
base,_,_=make_baseline(tr); resid,_=make_resid(tr,base)
vd=va[va.Pick<263]; vb=base(vd); vr=resid(vd)
shr={}
for pg,g in vd.groupby("pos_g",observed=True):
    i=(vd.pos_g==pg).values
    shr[pg]=float(max(np.arange(0,1.05,.1),key=lambda s:spearmanr(vb[i]+s*vr[i],g.y).statistic)) if len(g)>=50 else .4
tr2=df[df.Year<=2015].copy(); tr2["col_enc"]=college_enc(tr2,tr2); te["col_enc"]=college_enc(tr2,te)
base,_,_=make_baseline(tr2); resid,_=make_resid(tr2,base)
td=te[te.Pick<263].copy(); b=base(td); r=resid(td)
apex=b+td.pos_g.map(shr).fillna(.4).values*r
pickonly=IsotonicRegression(out_of_bounds="clip").fit(-tr2.Pick,tr2.y).predict(-td.Pick)
print("TEST 2016-19 drafted | pick-only:",round(spearmanr(pickonly,td.y).statistic,4),
 "| APEX v2:",round(spearmanr(apex,td.y).statistic,4),
 "| hitAUC:",round(roc_auc_score(td.hit,apex),4),"vs",round(roc_auc_score(td.hit,pickonly),4))
print("shrink:",{k:round(v,1) for k,v in shr.items()})

# production: fit on all labeled data
lab=df[hist].copy(); lab["col_enc"]=college_enc(lab,lab)
baseF,glob,isos=make_baseline(lab); residF,ms=make_resid(lab,baseF)
df["col_enc"]=college_enc(lab,df)
rAll=residF(df); sv=df.pos_g.map(shr).fillna(.4).values
bAll=np.where(df.Pick.notna()&(df.Pick<263), baseF(df), np.nan)
df["exp_at_pick"]=bAll
df["talent_resid"]=rAll
df["apex"]=np.where(np.isnan(bAll), lab.y.mean()+rAll, bAll+sv*rAll)  # prospects: athletic grade
df["surplus"]=np.where(np.isnan(bAll), np.nan, df.apex-df.exp_at_pick)
out=df[["Year","Player","Pos","pos_g","College","Pick","CarAV","y","apex","exp_at_pick","talent_resid","surplus"]].round(4)
out.to_csv("/home/claude/apex_board.csv",index=False)
for i,m in enumerate(ms): m.booster_.save_model(f"/home/claude/apex_resid_{i}.txt")
joblib.dump({"glob":glob,"isos":isos,"shrink":shr},"/home/claude/apex_baseline.pkl")
print("board rows:",len(out),"| 2025:",(out.Year==2025).sum(),"| 2026:",(out.Year==2026).sum())
