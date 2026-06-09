"""APEX Draft Model — combine + draft-capital ensemble, nflverse/PFR public data."""
import pandas as pd, numpy as np, lightgbm as lgb, re, json
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

D = "/home/claude/kv/NFL-Player-Valuation-master/data/"
POSMAP = {"HB":"RB","FB":"RB","TB":"RB","S":"DB","FS":"DB","SS":"DB","CB":"DB","DB":"DB",
 "OLB":"LB","ILB":"LB","MLB":"LB","LB":"LB","DE":"EDGE","NT":"DT","DT":"DT","C":"OL","G":"OL",
 "OG":"OL","OT":"OL","T":"OL","OL":"OL","QB":"QB","WR":"WR","TE":"TE","K":"ST","P":"ST","LS":"ST","RB":"RB"}

def norm(s): return re.sub(r"[^a-z]","",str(s).lower().replace("jr","").replace("iii","").replace("ii",""))

# ---- load
dr = pd.read_csv(D+"draft_data.csv")
dr = dr[(dr.Year>=2000)&(dr.Year<=2016)].copy()
dr["key"] = dr.Player.map(norm)+"_"+dr.Year.astype(str)
dr["pos_g"] = dr.Pos.map(POSMAP).fillna("OTH")
dr["CarAV"] = dr.CarAV.fillna(0)

cb = pd.read_csv(D+"combine_data_pfr_with_stats.csv")
cb = cb[cb.year<=2016].copy()
for c in ["height","weight","dash","vert_leap","bench","broad","cone","shuttle"]:
    cb[c] = cb[c].replace(0, np.nan)
cb["key"] = cb.player.map(norm)+"_"+cb.year.astype(str)
cb = cb.drop_duplicates("key")

df = dr.merge(cb[["key","height","weight","dash","vert_leap","bench","broad","cone","shuttle","college"]],
              on="key", how="left")
# undrafted combine attendees -> outcome 0 rows (teach model the full talent pool)
ud = cb[~cb.key.isin(dr.key)].copy()
ud = ud.rename(columns={"year":"Year","player":"Player","position":"Pos"})
ud["pos_g"]=ud.Pos.map(POSMAP).fillna("OTH"); ud["CarAV"]=0.0; ud["Pick"]=260; ud["Rnd"]=8
df = pd.concat([df, ud[["Year","Player","Pos","pos_g","Pick","Rnd","CarAV","key","height","weight",
        "dash","vert_leap","bench","broad","cone","shuttle","college"]]], ignore_index=True)
df = df[df.pos_g!="ST"]

# ---- target: within-class AV percentile (censoring-proof)
df["y"] = df.groupby("Year")["CarAV"].rank(pct=True)
df["hit"] = (df.groupby("Year")["CarAV"].rank(pct=True, ascending=False) <= .10).astype(int)  # top-10% of class

# ---- features
df["bmi"] = df.weight/(df.height**2)*703
df["speed_score"] = (df.weight*200)/(df.dash**4)
df["explosion"] = df.vert_leap.fillna(df.vert_leap.median()) + df.broad.fillna(df.broad.median())/12
df["agility"] = df.cone + df.shuttle
for c in ["dash","speed_score","explosion","agility","weight","height","bmi","bench","vert_leap","broad"]:
    mu = df.groupby("pos_g")[c].transform("mean"); sd = df.groupby("pos_g")[c].transform("std")
    df[c+"_z"] = (df[c]-mu)/sd
df["age"] = df.Age if "Age" in df else np.nan
df["logpick"] = np.log(df.Pick.clip(1,262))

# college strength: shrunken historical mean y (train-era only to avoid leakage; fit inside CV below)
def college_enc(train, full, k=12):
    g = train.groupby(full.college.reindex(train.index))["y"].agg(["mean","count"])
    prior = train.y.mean()
    enc = (g["mean"]*g["count"] + prior*k)/(g["count"]+k)
    return full.college.map(enc).fillna(prior)

FEATS_A = ["dash_z","speed_score_z","explosion_z","agility_z","weight_z","height_z","bmi_z",
           "bench_z","vert_leap_z","broad_z","age","col_enc"]
FEATS_C = FEATS_A + ["logpick"]
CATS = ["pos_g"]
df["pos_g"] = df.pos_g.astype("category")

def fit(train, feats, mono=None):
    p = dict(objective="regression", metric="rmse", learning_rate=.03, num_leaves=31,
             min_data_in_leaf=40, feature_fraction=.8, bagging_fraction=.8, bagging_freq=1,
             lambda_l2=2.0, verbose=-1, n_estimators=900)
    m = lgb.LGBMRegressor(**p)
    m.fit(train[feats+CATS], train.y, categorical_feature=CATS)
    return m

# ---- time-based evaluation: train 2000-2011, test 2012-2014 (mature outcomes)
tr = df[df.Year<=2011].copy(); te = df[(df.Year>=2012)&(df.Year<=2014)].copy()
tr["col_enc"] = college_enc(tr, tr); te["col_enc"] = college_enc(tr, te)

mA = fit(tr, FEATS_A)                      # pre-draft athletic model
mC = fit(tr, FEATS_C)                      # full ensemble w/ draft capital
te["pA"] = mA.predict(te[FEATS_A+CATS]); te["pC"] = mC.predict(te[FEATS_C+CATS])

# baseline: pick-only expectation (isotonic on -pick)
iso = IsotonicRegression(out_of_bounds="clip").fit(-tr.Pick, tr.y)
te["pB"] = iso.predict(-te.Pick)

res = {}
drafted = te[te.Pick<260]
for name,col in [("A_athletic_only","pA"),("B_pick_only","pB"),("C_ensemble","pC")]:
    res[name] = {
      "spearman_all": round(spearmanr(te[col], te.y).statistic,4),
      "spearman_drafted": round(spearmanr(drafted[col], drafted.y).statistic,4)}
print(json.dumps(res, indent=1))

# per-position ensemble lift vs pick-only
print("\nPer-position Spearman (drafted, 2012-14): ensemble vs pick-only")
for p,g in drafted.groupby("pos_g", observed=True):
    if len(g)>=60:
        print(f"{p:5s} n={len(g):4d}  C={spearmanr(g.pC,g.y).statistic:.3f}  B={spearmanr(g.pB,g.y).statistic:.3f}")

# ---- refit on all data, save artifacts + surplus board
df["col_enc"] = college_enc(df, df)
mA_f = fit(df, FEATS_A); mC_f = fit(df, FEATS_C)
iso_f = IsotonicRegression(out_of_bounds="clip").fit(-df.Pick, df.y)
df["pred_talent"] = mA_f.predict(df[FEATS_A+CATS])
df["exp_at_pick"] = iso_f.predict(-df.Pick)
df["surplus"] = df.pred_talent - df.exp_at_pick
mA_f.booster_.save_model("/home/claude/apex_athletic.txt")
mC_f.booster_.save_model("/home/claude/apex_ensemble.txt")
cols=["Year","Player","Pos","pos_g","Pick","CarAV","y","pred_talent","exp_at_pick","surplus"]
df[cols].round(4).to_csv("/home/claude/apex_board.csv", index=False)

fi = pd.Series(mC_f.feature_importances_, index=FEATS_C+CATS).sort_values(ascending=False)
print("\nTop features (ensemble):"); print(fi.head(8).to_string())
print("\nBiggest steals (model loved, draft didn't):")
print(df[(df.Pick>=80)&(df.Pick<260)].nlargest(8,"surplus")[["Year","Player","Pos","Pick","CarAV","surplus"]].to_string(index=False))
