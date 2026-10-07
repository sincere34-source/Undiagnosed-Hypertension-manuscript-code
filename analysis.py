# ============================================================================
# ANALYTIC PYTHON CODE
# Comparative Performance of Classical Statistical and Machine Learning Models
# in Predicting Undiagnosed Hypertension among Rwandans aged 18-69
# (2022 Rwanda NCD STEPS Survey)
# Running this code with the STEPS dataset reproduces every table and figure
# reported in Chapter 4. Fixed seed (RANDOM_STATE = 42) keeps results stable.
# ============================================================================

# ============ PART 0  Setup, imports, config ============
# !pip install "scikit-learn==1.4.2" "imbalanced-learn==0.12.3" "pandas==2.2.2" \
#              "numpy==1.26.4" "scipy==1.13.1" "matplotlib==3.8.4" \
#              "seaborn==0.13.2" "pyreadstat==1.2.7"
# %matplotlib inline   # (Jupyter magic; omit in a .py script)
import os, warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from IPython.display import display
sns.set_theme(style="whitegrid"); warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 40)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, cross_val_predict)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, roc_curve, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_curve, average_precision_score,
    classification_report)
from sklearn.pipeline import Pipeline as SkPipeline
import statsmodels.api as sm
import statsmodels.formula.api as smf

RANDOM_STATE = 42; np.random.seed(RANDOM_STATE)
DATA_FILE = os.environ.get("HTN_DATA", "final dataset_STEP_2023.dta")
SENTINELS = [77,88,99,777,888,999,7777,8888,9999]

SAVE = False                       # <-- set True to ALSO write PNG/CSV to results/
FIGDIR, TABDIR = "results/figures", "results/tables"
def savefig(fig, name):
    if SAVE:
        os.makedirs(FIGDIR, exist_ok=True); fig.savefig(f"{FIGDIR}/{name}", dpi=140, bbox_inches="tight")
def savecsv(df, name):
    if SAVE:
        os.makedirs(TABDIR, exist_ok=True); df.to_csv(f"{TABDIR}/{name}", index=False)
print("Setup ready. SAVE =", SAVE)

df = pd.read_stata(DATA_FILE, convert_categoricals=False)
print("Raw shape:", df.shape[0], "respondents x", df.shape[1], "variables")
display(df.head())

sbp = pd.to_numeric(df["average_sbp"], errors="coerce")
dbp = pd.to_numeric(df["average_dbp"], errors="coerce")
measured_high = (sbp >= 140) | (dbp >= 90)
prev_diag = df["h2a"] == 1
bp_measured = sbp.notna() & dbp.notna()
df["undiagnosed_htn"] = np.where(bp_measured & measured_high & (~prev_diag), 1,
                                 np.where(bp_measured, 0, np.nan))
# Breakdown computed on the 18-69 analytic sample (consistent with the reported figures)
_age = pd.to_numeric(df["age"], errors="coerce")
elig = bp_measured & _age.between(18, 69)
breakdown = pd.DataFrame({
    "Group": ["Measured hypertension"," previously diagnosed"," undiagnosed (outcome=1)","Eligible sample (18-69, valid BP)"],
    "n": [int(measured_high[elig].sum()),
          int((measured_high & prev_diag)[elig].sum()),
          int(np.nansum(df["undiagnosed_htn"][elig])),
          int(elig.sum())]})
display(breakdown)
print(f"Unweighted prevalence of undiagnosed HTN (18-69): {100*np.nanmean(df['undiagnosed_htn'][elig]):.1f}%")

def clean_sentinels(s): return s.replace(SENTINELS, np.nan)

def compute_gpaq_met(df):
    def minutes(hh, mm):
        h = clean_sentinels(pd.to_numeric(df[hh], errors="coerce")).fillna(0)
        m = clean_sentinels(pd.to_numeric(df[mm], errors="coerce")).fillna(0)
        return h*60 + m
    def days(dd): return clean_sentinels(pd.to_numeric(df[dd], errors="coerce")).fillna(0)
    total = (8*days("p2")*minutes("p3a","p3b") + 4*days("p5")*minutes("p6a","p6b")
             + 4*days("p8")*minutes("p9a","p9b") + 8*days("p11")*minutes("p12a","p12b")
             + 4*days("p14")*minutes("p15a","p15b"))
    return total, pd.cut(total, [-1,600,3000,np.inf], labels=["Low","Moderate","High"]).astype("object")

X = pd.DataFrame(index=df.index)
X["age"] = clean_sentinels(pd.to_numeric(df["age"], errors="coerce"))
X["sex"] = df["c1"].map({1:"Male",2:"Female"})
X["residence"] = df["ur"].map({1:"Urban",2:"Rural"})
X["marital_status"] = df["c7"].map({1:"Never married",2:"Married/Cohabiting",6:"Married/Cohabiting",
    3:"Separated/Divorced/Widowed",4:"Separated/Divorced/Widowed",5:"Separated/Divorced/Widowed",88:np.nan})
X["education"] = df["c5"].map({1:"None",2:"Primary",3:"Primary",4:"Secondary",5:"Secondary",6:"Tertiary",88:np.nan})
X["occupation"] = df["c8"].map({1:"Employed",3:"Self-employed",2:"Unemployed",8:"Unemployed",9:"Unemployed",
    4:"Other/Inactive",5:"Other/Inactive",6:"Other/Inactive",7:"Other/Inactive",88:np.nan})
X["ubudehe"] = df["x1"].map({1:"Cat_1",2:"Cat_2",3:"Cat_3",4:"Cat_4",5:"Cat_5",77:np.nan,88:np.nan})
X["tobacco_use"] = df["t1"].map({1:"Yes",0:"No",2:"No"})
X["alcohol_use"] = np.where(df["a2"]==1,"Yes","No")
d1=clean_sentinels(pd.to_numeric(df["d1"],errors="coerce")); d2=clean_sentinels(pd.to_numeric(df["d2"],errors="coerce"))
d3=clean_sentinels(pd.to_numeric(df["d3"],errors="coerce")); d4=clean_sentinels(pd.to_numeric(df["d4"],errors="coerce"))
X["fruit_servings_day"]=(d1.fillna(0)*d2.fillna(0))/7.0; X["veg_servings_day"]=(d3.fillna(0)*d4.fillna(0))/7.0
met, met_cat = compute_gpaq_met(df); X["phys_activity_met"]=met; X["phys_activity_level"]=met_cat
X["salt_intake"] = df["d5"].map({1:"High",2:"High",3:"Moderate",4:"Low",5:"Low",77:np.nan})
bmi=clean_sentinels(pd.to_numeric(df["bmi"],errors="coerce")); X["bmi"]=bmi.where((bmi>=10)&(bmi<=70))
waist=clean_sentinels(pd.to_numeric(df["m14"],errors="coerce")); X["waist_cm"]=waist.where((waist>=40)&(waist<=200))
gluc=clean_sentinels(pd.to_numeric(df["b5"],errors="coerce")); X["fasting_glucose_mmol"]=(gluc/18.0).where(gluc.between(20,500))
chol=clean_sentinels(pd.to_numeric(df["b8"],errors="coerce")); X["total_chol_mmol"]=(chol/38.67).where(chol.between(50,500))
X["survey_weight"]=pd.to_numeric(df["indwt"],errors="coerce")
X["stratum"]=df["stratum"].astype(str); X["psu"]=df["psu"].astype(str)
X["undiagnosed_htn"]=df["undiagnosed_htn"]

NUMERIC_FEATURES=["age","fruit_servings_day","veg_servings_day","phys_activity_met",
                  "bmi","waist_cm","fasting_glucose_mmol","total_chol_mmol"]
CATEGORICAL_FEATURES=["sex","residence","marital_status","education","occupation",
                      "ubudehe","tobacco_use","alcohol_use","phys_activity_level","salt_intake"]
data = X[X["undiagnosed_htn"].notna()].copy(); data["undiagnosed_htn"]=data["undiagnosed_htn"].astype(int)
print("Analytic sample (valid BP):", len(data))
display(data.head())

idx = data.index
height = clean_sentinels(pd.to_numeric(df["m11"], errors="coerce")); height=height.where((height>=120)&(height<=210)).reindex(idx)
hr  = clean_sentinels(pd.to_numeric(df["average_heartrate"], errors="coerce")); hr=hr.where((hr>=30)&(hr<=200)).reindex(idx)
hip = clean_sentinels(pd.to_numeric(df["m15"], errors="coerce")); hip=hip.where((hip>=60)&(hip<=200)).reindex(idx)
data["heart_rate"]=hr; data["waist_to_height"]=data["waist_cm"]/height
data["waist_to_hip"]=data["waist_cm"]/hip; data["age_sq"]=data["age"]**2; data["age_x_bmi"]=data["age"]*data["bmi"]
data["diabetes"]=df["Diebet"].map({1:"Yes",0:"No"}).reindex(idx)
NUM = NUMERIC_FEATURES + ["heart_rate","waist_to_height","waist_to_hip","age_sq","age_x_bmi"]
CAT = CATEGORICAL_FEATURES + ["diabetes"]
data = data[(data.age>=18)&(data.age<=69)].copy()
print("Modelling sample (18-69):", len(data), "| positives:", int(data.undiagnosed_htn.sum()),
      f"({100*data.undiagnosed_htn.mean():.1f}%)")
display(data[NUM].describe().T.round(2))

y = data["undiagnosed_htn"]; N=len(data)
# Table 4.1 - Profile of respondents (full demographic/behavioural/clinical profile)
labels={"sex":"Sex","residence":"Residence","education":"Education level","marital_status":"Marital status",
        "occupation":"Occupation","ubudehe":"Ubudehe category","tobacco_use":"Current tobacco use",
        "alcohol_use":"Alcohol use","phys_activity_level":"Physical activity","diabetes":"Diabetes"}
prof=[]
for col,lab in labels.items():
    for k,v in data[col].value_counts(dropna=False).items():
        prof.append({"Characteristic":lab,"Category":("Missing" if pd.isna(k) else str(k)),
                     "n":int(v),"Percent (%)":round(100*v/N,1)})
t41=pd.DataFrame(prof); print("Table 4.1  Profile of respondents"); display(t41); savecsv(t41,"table_4_1_profile.csv")
# Table 4.2 - Numeric predictors by outcome group
rows=[{"Variable":c,"Overall mean":round(data[c].mean(),2),
       "No-undx mean":round(data.loc[y==0,c].mean(),2),
       "Undx mean":round(data.loc[y==1,c].mean(),2)} for c in NUM]
t42=pd.DataFrame(rows); print("Table 4.2  Numeric predictors by outcome"); display(t42); savecsv(t42,"table_4_2_numeric.csv")
# Table 4.3 - Missing values
t43=(data[NUM+CAT].isna().mean()*100).round(2).sort_values(ascending=False).rename("percent_missing").reset_index().rename(columns={"index":"variable"})
print("Table 4.3  Missing values (%)"); display(t43); savecsv(t43,"table_4_3_missing.csv")
# Table 4.4 - Summary statistics
t44=data[NUM].describe().T.round(2); print("Table 4.4  Summary statistics"); display(t44); savecsv(t44.reset_index(),"table_4_4_summary.csv")
# Table 4.5 - Outliers (1.5*IQR rule)
orows=[]
for c in NUM:
    s=data[c].dropna(); q1,q3=s.quantile([.25,.75]); iqr=q3-q1
    n=int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum()); orows.append({"variable":c,"n_outliers":n,"%":round(100*n/len(data),1)})
t45=pd.DataFrame(orows); print("Table 4.5  Outliers per variable"); display(t45); savecsv(t45,"table_4_5_outliers.csv")

# Figure 4.1 - Prevalence
fig,ax=plt.subplots(figsize=(6,5)); c=y.value_counts().sort_index()
ax.bar(["No","Yes"],c.values,color=["#4C72B0","#C44E52"])
for i,v in enumerate(c.values): ax.text(i,v,f"{v}\n({100*v/len(y):.1f}%)",ha="center",va="bottom")
ax.set_title(f"Undiagnosed hypertension (prevalence {100*y.mean():.1f}%)"); ax.set_ylabel("Respondents"); ax.set_ylim(0,c.max()*1.22)
savefig(fig,"01_prevalence.png"); plt.show()
# Figure 4.2 - Missing values
miss=(data[NUM].isna().mean()*100).sort_values()
fig,ax=plt.subplots(figsize=(8,5)); miss.plot.barh(ax=ax,color="#4C72B0"); ax.set_title("Missing values by variable"); ax.set_xlabel("% missing")
savefig(fig,"02_missing.png"); plt.show()
# Figure 4.3 - Boxplots (outliers)
fig,axes=plt.subplots(2,4,figsize=(15,7))
for ax,c in zip(axes.ravel(),NUM): ax.boxplot(data[c].dropna(),vert=True); ax.set_title(c,fontsize=9)
fig.suptitle("Boxplots of numeric predictors (outlier check)"); fig.tight_layout(); savefig(fig,"03_boxplots.png"); plt.show()
# Figure 4.4 - Histograms
fig,axes=plt.subplots(2,4,figsize=(15,7))
for ax,c in zip(axes.ravel(),NUM): ax.hist(data[c].dropna(),bins=30,color="#4C72B0"); ax.set_title(c,fontsize=9)
fig.suptitle("Distributions of numeric predictors"); fig.tight_layout(); savefig(fig,"04_histograms.png"); plt.show()
# Figure 4.5 - Correlation matrix
fig,ax=plt.subplots(figsize=(10,8))
sns.heatmap(data[NUM+["undiagnosed_htn"]].corr(),annot=True,fmt=".2f",cmap="coolwarm",center=0,ax=ax,annot_kws={"size":7})
ax.set_title("Correlation matrix"); savefig(fig,"05_correlation.png"); plt.show()

w=data["survey_weight"]; valid=w.notna(); w2,y2,d=w[valid],y[valid],data[valid]
wprev=lambda m=None: 100*np.average(y2 if m is None else y2[m], weights=w2 if m is None else w2[m])
rows=[{"group":"Overall","category":"All (18-69)","weighted_%":round(wprev(),1),"unweighted_%":round(100*y2.mean(),1),"n":int(valid.sum())}]
ageg=pd.cut(d["age"],[17,29,39,49,59,69],labels=["18-29","30-39","40-49","50-59","60-69"])
for gname,series in [("Age",ageg),("Sex",d["sex"]),("Residence",d["residence"]),("Education",d["education"])]:
    for cat in series.dropna().unique():
        m=(series==cat).values
        rows.append({"group":gname,"category":str(cat),"weighted_%":round(wprev(m),1),"unweighted_%":round(100*y2[m].mean(),1),"n":int(m.sum())})
t46=pd.DataFrame(rows); print("Table 4.6  Survey-weighted prevalence"); display(t46); savecsv(t46,"table_4_6_weighted.csv")
# Design-based 95% CI for overall weighted prevalence (stratified ultimate-cluster, Taylor linearization)
_ph=(w2*y2).sum()/w2.sum(); _e=(w2*(y2-_ph)).values
_cl=pd.DataFrame({"h":d["stratum"].values,"psu":d["psu"].values,"e":_e}); _v=0.0
for _h,_g in _cl.groupby("h"):
    _pc=_g.groupby("psu")["e"].sum(); _nh=len(_pc)
    if _nh>1: _v+=(_nh/(_nh-1))*((_pc-_pc.mean())**2).sum()
_se=(_v**0.5)/w2.sum()
print(f"Overall undiagnosed prevalence (design-based): {100*_ph:.1f}% (95% CI {100*(_ph-1.96*_se):.1f}-{100*(_ph+1.96*_se):.1f})")
# Figure 4.6 - Weighted prevalence of undiagnosed hypertension by age group
ar=t46[t46.group=="Age"].sort_values("category")
fig,ax=plt.subplots(figsize=(8,5)); ax.bar(ar["category"],ar["weighted_%"],color="#C44E52")
for i,v in enumerate(ar["weighted_%"]): ax.text(i,v,f"{v:.1f}%",ha="center",va="bottom")
ax.set_title("Weighted prevalence by age group"); ax.set_ylabel("Weighted %"); savefig(fig,"06_weighted_age.png"); plt.show()

rows=[]
for c in NUM:
    g0=data.loc[y==0,c].dropna(); g1=data.loc[y==1,c].dropna(); t,p=stats.ttest_ind(g0,g1,equal_var=False)
    rows.append({"variable":c,"test":"Welch t","statistic":round(t,3),"p_value":p})
for c in CAT:
    ct=pd.crosstab(data[c],y); chi2,p,_,_=stats.chi2_contingency(ct)
    rows.append({"variable":c,"test":"chi-square","statistic":round(chi2,3),"p_value":p})
t47=pd.DataFrame(rows).sort_values("p_value"); t47["significant"]=np.where(t47.p_value<0.05,"Yes","No")
print("Table 4.7  Bivariate associations"); display(t47); savecsv(t47,"table_4_7_bivariate.csv")

numeric=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())])
categorical=Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore",drop="first"))])
prep=ColumnTransformer([("num",numeric,NUM),("cat",categorical,CAT)])
def pipe(clf,smote=False):
    # Class imbalance is handled by class weighting (and threshold tuning), NOT oversampling.
    # 'smote' retained only for signature compatibility; SMOTE is never applied.
    return SkPipeline([("prep",prep),("clf",clf)])
def best_threshold(yv,prob):
    p,r,t=precision_recall_curve(yv,prob); f1=2*p*r/(p+r+1e-12); return float(t[np.argmax(f1[:-1])]) if len(t) else 0.5
models={
 "Logistic Regression":pipe(LogisticRegression(C=0.01,penalty="l2",solver="liblinear",class_weight="balanced",max_iter=4000,random_state=RANDOM_STATE)),
 "Decision Tree":pipe(DecisionTreeClassifier(max_depth=3,min_samples_leaf=20,criterion="gini",class_weight="balanced",random_state=RANDOM_STATE)),
 "Random Forest":pipe(RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=4,min_samples_split=10,max_features="log2",class_weight="balanced_subsample",n_jobs=-1,random_state=RANDOM_STATE)),
 "Support Vector Machine":pipe(SVC(kernel="linear",C=1.0,probability=True,class_weight="balanced",random_state=RANDOM_STATE)),
 "Gradient Boosting":pipe(GradientBoostingClassifier(n_estimators=150,learning_rate=0.03,max_depth=3,subsample=0.8,min_samples_leaf=20,random_state=RANDOM_STATE),smote=False),
 "Artificial Neural Network":pipe(MLPClassifier(hidden_layer_sizes=(64,),activation="tanh",alpha=1e-4,learning_rate_init=1e-3,max_iter=800,early_stopping=True,random_state=RANDOM_STATE))}
t48=pd.DataFrame([
 {"Model":"Logistic Regression","Key tuned hyperparameters":"C=0.01, L2 penalty, liblinear, class_weight=balanced"},
 {"Model":"Decision Tree","Key tuned hyperparameters":"max_depth=3, min_samples_leaf=20, gini, class_weight=balanced"},
 {"Model":"Random Forest","Key tuned hyperparameters":"n_estimators=300, max_depth=8, min_samples_leaf=4, max_features=log2"},
 {"Model":"Support Vector Machine","Key tuned hyperparameters":"linear kernel, C=1.0, class_weight=balanced"},
 {"Model":"Gradient Boosting","Key tuned hyperparameters":"n_estimators=150, learning_rate=0.03, max_depth=3, subsample=0.8, min_samples_leaf=20"},
 {"Model":"Artificial Neural Network","Key tuned hyperparameters":"hidden=(64,), tanh, alpha=1e-4, early_stopping"}])
print("Table 4.8  Tuned hyperparameters"); display(t48); savecsv(t48,"table_4_8_hyperparameters.csv")

Xall=data[NUM+CAT]; yall=data["undiagnosed_htn"]
Xtr,Xte,ytr,yte=train_test_split(Xall,yall,test_size=0.20,stratify=yall,random_state=RANDOM_STATE)
rows,probs,preds,thr_,roc_data=[],{},{},{},{}
for name,pl in models.items():
    pl.fit(Xtr,ytr); prob=pl.predict_proba(Xte)[:,1]; probs[name]=prob
    oof=cross_val_predict(pl,Xtr,ytr,cv=3,method="predict_proba",n_jobs=-1)[:,1]
    thr=best_threshold(ytr,oof); thr_[name]=thr; pred=(prob>=thr).astype(int); preds[name]=pred
    _tn,_fp,_fn,_tp=confusion_matrix(yte,pred).ravel()
    _spec=_tn/(_tn+_fp) if (_tn+_fp) else 0.0
    _ppv=_tp/(_tp+_fp) if (_tp+_fp) else 0.0
    _npv=_tn/(_tn+_fn) if (_tn+_fn) else 0.0
    rows.append({"Model":name,"AUC_ROC":round(roc_auc_score(yte,prob),3),"Accuracy":round(accuracy_score(yte,pred),3),
        "Recall":round(recall_score(yte,pred),3),"Specificity":round(_spec,3),"PPV":round(_ppv,3),"NPV":round(_npv,3),
        "Balanced_Acc":round(balanced_accuracy_score(yte,pred),3),
        "F1":round(f1_score(yte,pred),3),"Threshold":round(thr,3)})
    fpr,tpr,_=roc_curve(yte,prob); roc_data[name]=(fpr,tpr,roc_auc_score(yte,prob))
t49=pd.DataFrame(rows).sort_values("AUC_ROC",ascending=False).reset_index(drop=True)
print("Table 4.9  Comparative performance (test set)"); display(t49); savecsv(t49,"table_4_9_performance.csv")

# Figure 4.7 - Metrics bar
melt=t49.melt(id_vars="Model",value_vars=["AUC_ROC","Accuracy","Recall","Balanced_Acc","F1"],var_name="Metric",value_name="Score")
fig,ax=plt.subplots(figsize=(12,6)); sns.barplot(data=melt,x="Model",y="Score",hue="Metric",ax=ax)
ax.set_title("Performance metrics by model"); ax.set_xticklabels(ax.get_xticklabels(),rotation=25,ha="right"); ax.legend(bbox_to_anchor=(1.01,1),loc="upper left")
savefig(fig,"07_metrics_bar.png"); plt.show()
# Figure 4.8 - ROC
fig,ax=plt.subplots(figsize=(8,7))
for name,(fpr,tpr,a) in sorted(roc_data.items(),key=lambda kv:-kv[1][2]): ax.plot(fpr,tpr,lw=2,label=f"{name} (AUC={a:.3f})")
ax.plot([0,1],[0,1],"k--",lw=1); ax.legend(loc="lower right",fontsize=9); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC comparison")
savefig(fig,"08_roc.png"); plt.show()
# Figure 4.9 - Precision-Recall
fig,ax=plt.subplots(figsize=(8,7))
for name in sorted(probs,key=lambda n:-average_precision_score(yte,probs[n])):
    pr,rc,_=precision_recall_curve(yte,probs[name]); ap=average_precision_score(yte,probs[name]); ax.plot(rc,pr,lw=2,label=f"{name} (AP={ap:.3f})")
ax.axhline(yte.mean(),ls="--",c="grey",label=f"baseline={yte.mean():.2f}"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall curves"); ax.legend(fontsize=8)
savefig(fig,"09_pr.png"); plt.show()
# Figure 4.10 - all confusion matrices
fig,axes=plt.subplots(2,3,figsize=(14,8))
for ax,name in zip(axes.ravel(),models):
    ConfusionMatrixDisplay(confusion_matrix(yte,preds[name]),display_labels=["No","Yes"]).plot(ax=ax,cmap="Blues",colorbar=False); ax.set_title(f"{name}\n(thr={thr_[name]:.2f})",fontsize=9)
fig.tight_layout(); savefig(fig,"10_confusion_all.png"); plt.show()
# Figure 4.11 - primary model confusion matrix
P="Gradient Boosting"; tn,fp,fn,tp=confusion_matrix(yte,preds[P]).ravel()
fig,ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay(confusion_matrix(yte,preds[P]),display_labels=["No","Yes"]).plot(ax=ax,cmap="Blues",colorbar=False)
ax.set_title(f"Gradient Boosting (primary, threshold = {thr_[P]:.3f})"); fig.tight_layout(); savefig(fig,"11_primary_cm.png"); plt.show()
print(f"Primary GB: detects {tp}/{int(yte.sum())}, misses {fn}, false alarms {fp}")

cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=RANDOM_STATE)
rows=[]
for name,pl in models.items():
    s=cross_val_score(pl,Xtr,ytr,cv=cv,scoring="roc_auc",n_jobs=-1)
    rows.append({"Model":name,"CV_AUC_mean":round(s.mean(),3),"CV_AUC_sd":round(s.std(),3)})
t410=pd.DataFrame(rows).sort_values("CV_AUC_mean",ascending=False)
print("Table 4.11  Cross-validated AUC"); display(t410); savecsv(t410,"table_4_11_cv.csv")

gb=models["Gradient Boosting"]      # already fitted in Part 10
perm=permutation_importance(gb,Xte,yte,n_repeats=5,random_state=RANDOM_STATE,scoring="roc_auc")
imp=pd.DataFrame({"feature":NUM+CAT,"importance":perm.importances_mean}).sort_values("importance",ascending=False)
print("Table 4.16  Top predictors (permutation importance)"); display(imp.head(10)); savecsv(imp,"table_4_16_importance.csv")
# Figure 4.12 - Permutation feature importance (Gradient Boosting)
fig,ax=plt.subplots(figsize=(8,7)); sns.barplot(data=imp.head(15),x="importance",y="feature",color="#4C72B0",ax=ax)
ax.set_title("Permutation feature importance (Gradient Boosting)"); ax.set_xlabel("Mean AUC decrease"); fig.tight_layout(); savefig(fig,"12_feature_importance.png"); plt.show()

baseline={
 "Logistic Regression":pipe(LogisticRegression(max_iter=2000,random_state=RANDOM_STATE)),
 "Decision Tree":pipe(DecisionTreeClassifier(random_state=RANDOM_STATE)),
 "Random Forest":pipe(RandomForestClassifier(n_jobs=-1,random_state=RANDOM_STATE)),
 "Support Vector Machine":pipe(SVC(probability=True,random_state=RANDOM_STATE)),
 "Gradient Boosting":pipe(GradientBoostingClassifier(random_state=RANDOM_STATE),smote=False),
 "Artificial Neural Network":pipe(MLPClassifier(max_iter=500,random_state=RANDOM_STATE))}
rows=[]
tuned_auc={r["Model"]:r["AUC_ROC"] for _,r in t49.iterrows()}
for name,pl in baseline.items():
    pl.fit(Xtr,ytr); a0=round(roc_auc_score(yte,pl.predict_proba(Xte)[:,1]),3); a1=tuned_auc[name]
    rows.append({"Model":name,"Baseline AUC-ROC":a0,"Tuned AUC-ROC":a1,"Improvement":round(a1-a0,3)})
t413=pd.DataFrame(rows).sort_values("Tuned AUC-ROC",ascending=False)
print("Table 4.12  AUC-ROC before and after tuning"); display(t413); savecsv(t413,"table_4_12_before_after.csv")

yte_a=np.asarray(yte); actual10=[int(v) for v in yte_a[:10]]
abbr={"Logistic Regression":"LR","Decision Tree":"DT","Random Forest":"RF","Support Vector Machine":"SVM","Gradient Boosting":"GB","Artificial Neural Network":"ANN"}
prob_tab={"Record":list(range(1,11)),"Actual":actual10}; pred_tab={"Record":list(range(1,11)),"Actual":actual10}; rep_rows=[]
for name in models:
    prob=probs[name]; pred=preds[name]
    prob_tab[abbr[name]]=[round(float(prob[i]),3) for i in range(10)]
    pred_tab[abbr[name]]=[int(pred[i]) for i in range(10)]
    rep=classification_report(yte_a,pred,output_dict=True,zero_division=0)
    for cls,lab in [("0","0 (No)"),("1","1 (Yes)")]:
        rep_rows.append({"Model":name if cls=="0" else "","Class":lab,"Precision":round(rep[cls]["precision"],3),
                         "Recall":round(rep[cls]["recall"],3),"F1-score":round(rep[cls]["f1-score"],3),"Support":int(rep[cls]["support"])})
t414=pd.DataFrame(prob_tab); t415=pd.DataFrame(pred_tab); t416=pd.DataFrame(rep_rows)
print("Table 4.13  Predicted probabilities (first 10)"); display(t414); savecsv(t414,"table_4_13_prob.csv")
print("Table 4.14  Predicted classes at operating threshold (first 10)"); display(t415); savecsv(t415,"table_4_14_pred.csv")
print("Table 4.15  Test set classification report"); display(t416); savecsv(t416,"table_4_15_report.csv")

def mcnemar(a,b,yv):
    a,b,yv=np.asarray(a),np.asarray(b),np.asarray(yv); ca,cb=a==yv,b==yv
    n01=int(np.sum(ca&~cb)); n10=int(np.sum(~ca&cb))
    if n01+n10==0: return 0.0,1.0
    chi2=(abs(n01-n10)-1.0)**2/(n01+n10); return float(chi2),float(stats.chi2.sf(chi2,1))
base=preds["Logistic Regression"]; rows=[]
for name,pred in preds.items():
    if name=="Logistic Regression": continue
    chi2,p=mcnemar(base,pred,yte_a); rows.append({"comparison":f"LR vs {name}","McNemar_chi2":round(chi2,3),"p_value":round(p,4),"significant":"Yes" if p<0.05 else "No"})
t412=pd.DataFrame(rows); print("Table 4.17  McNemar (vs Logistic Regression)"); display(t412); savecsv(t412,"table_4_17_mcnemar.csv")

def delong_test(yv,p1,p2):
    yv=np.asarray(yv); P=np.vstack([np.asarray(p1),np.asarray(p2)]); pos=P[:,yv==1]; neg=P[:,yv==0]; m,n,k=pos.shape[1],neg.shape[1],2
    def midrank(x):
        J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
        while i<N:
            j=i
            while j<N and Z[j]==Z[i]: j+=1
            T[i:j]=0.5*(i+j-1)+1; i=j
        T2=np.empty(N); T2[J]=T; return T2
    tx=np.array([midrank(pos[r]) for r in range(k)]); ty=np.array([midrank(neg[r]) for r in range(k)]); tz=np.array([midrank(np.r_[pos[r],neg[r]]) for r in range(k)])
    aucs=(tz[:,:m].sum(axis=1)/m-(m+1)/2.0)/n; v01=(tz[:,:m]-tx)/n; v10=1.0-(tz[:,m:]-ty)/m
    var=np.cov(v01)/m+np.cov(v10)/n; z=(aucs[0]-aucs[1])/np.sqrt(var[0,0]+var[1,1]-2*var[0,1]+1e-12)
    return float(aucs[0]),float(aucs[1]),float(z),float(2*stats.norm.sf(abs(z)))
dgb=probs["Gradient Boosting"]; drows=[]
for name in models:
    if name=="Gradient Boosting": continue
    a1,a2,z,p=delong_test(yte_a,dgb,probs[name])
    drows.append({"comparison":f"GB vs {name}","delta_AUC":round(a1-a2,3),"p_value":round(p,3),"significant":"Yes" if p<0.05 else "No"})
t417=pd.DataFrame(drows).sort_values("delta_AUC").reset_index(drop=True)
print("Table 4.18  DeLong (Gradient Boosting vs each model)"); display(t417); savecsv(t417,"table_4_18_delong.csv")

def boot_ci(yv,prob,n_boot=1000,seed=RANDOM_STATE):
    yv=np.asarray(yv); prob=np.asarray(prob); rng=np.random.default_rng(seed); a=[]
    for _ in range(n_boot):
        i=rng.integers(0,len(yv),len(yv))
        if len(np.unique(yv[i]))<2: continue
        a.append(roc_auc_score(yv[i],prob[i]))
    return round(np.percentile(a,2.5),3),round(np.percentile(a,97.5),3)
ci=pd.DataFrame([{"Model":n,"AUC":round(roc_auc_score(yte_a,probs[n]),3),"CI95_low":boot_ci(yte_a,probs[n])[0],"CI95_high":boot_ci(yte_a,probs[n])[1]} for n in models])
print("Table 4.10  Bootstrap 95% CIs for test set AUC-ROC"); display(ci); savecsv(ci,"table_4_10_bootci.csv")

# ------------------------------------------------------------------
# Complete-case sensitivity analysis (reported in text, Section 4.2)
# Gradient Boosting refitted on respondents with no missing model predictors
cc = data.dropna(subset=NUM+CAT)
Xcc, ycc = cc[NUM+CAT], cc["undiagnosed_htn"].astype(int)
Xtr_c,Xte_c,ytr_c,yte_c = train_test_split(Xcc,ycc,test_size=0.20,stratify=ycc,random_state=RANDOM_STATE)
gb_cc = pipe(GradientBoostingClassifier(n_estimators=150,learning_rate=0.03,max_depth=3,
             subsample=0.8,min_samples_leaf=20,random_state=RANDOM_STATE),smote=False)
gb_cc.fit(Xtr_c,ytr_c)
auc_cc = roc_auc_score(yte_c, gb_cc.predict_proba(Xte_c)[:,1])
print(f"Complete-case sensitivity: n={len(cc)}, Gradient Boosting test AUC-ROC={auc_cc:.3f}")

# ------------------------------------------------------------------

# ============================================================================
# ADDITIONAL MANUSCRIPT ANALYSES
# (reduced logistic models + calibration; survey-weighted adjusted odds ratios;
#  survey-weighted Table 1 by outcome). Added for the journal revision.
# ============================================================================

# --- Reduced logistic models (age-only, age+BMI, full) on the same 80/20 split ---
def _lr_auc(cols):
    num=[c for c in cols if c in NUM]; cat=[c for c in cols if c in CAT]
    tr=[("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),num)]
    if cat: tr.append(("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),
                                       ("e",OneHotEncoder(handle_unknown="ignore",drop="first"))]),cat))
    pl=SkPipeline([("p",ColumnTransformer(tr)),
                   ("c",LogisticRegression(C=0.01,penalty="l2",solver="liblinear",
                                           class_weight="balanced",max_iter=4000,random_state=RANDOM_STATE))])
    pl.fit(Xtr[cols],ytr); return roc_auc_score(yte,pl.predict_proba(Xte[cols])[:,1])

t_red=pd.DataFrame([
    {"Model":"Age only","AUC_ROC":round(_lr_auc(["age"]),3)},
    {"Model":"Age + BMI","AUC_ROC":round(_lr_auc(["age","bmi"]),3)},
    {"Model":"Full logistic model","AUC_ROC":round(_lr_auc(NUM+CAT),3)}])
print("Reduced logistic models (AUC-ROC)"); display(t_red); savecsv(t_red,"table_reduced_models.csv")

# Calibration of the full (class-weighted) logistic model: intercept and slope
_prep_full=ColumnTransformer([
    ("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),
    ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),
                     ("e",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])
_lrf=SkPipeline([("p",_prep_full),("c",LogisticRegression(C=0.01,penalty="l2",solver="liblinear",
                 class_weight="balanced",max_iter=4000,random_state=RANDOM_STATE))]).fit(Xtr,ytr)
_ph=np.clip(_lrf.predict_proba(Xte)[:,1],1e-6,1-1e-6); _logit=np.log(_ph/(1-_ph))
_cal=sm.Logit(np.asarray(yte),sm.add_constant(_logit)).fit(disp=0)
print(f"Calibration (full logistic model): intercept={_cal.params[0]:.3f}, slope={_cal.params[1]:.3f}")

# --- Survey-weighted multivariable logistic regression: adjusted odds ratios ---
_cont=[c for c in ["age","bmi","waist_cm","heart_rate","fasting_glucose_mmol","total_chol_mmol"] if c in data.columns]
_catv=[c for c in ["sex","residence","alcohol_use","tobacco_use","phys_activity_level","diabetes"] if c in data.columns]
_dw=data[data["survey_weight"].notna()].dropna(subset=_cont+_catv+["undiagnosed_htn"]).copy()
_rhs=" + ".join(_cont+[f"C({c})" for c in _catv])
_glm=smf.glm("undiagnosed_htn ~ "+_rhs, data=_dw, family=sm.families.Binomial(),
             var_weights=_dw["survey_weight"]).fit(cov_type="cluster", cov_kwds={"groups":_dw["psu"]})
_or=np.exp(_glm.params); _ci=np.exp(_glm.conf_int())
aor_rows=[]
for k in _glm.params.index:
    if k=="Intercept": continue
    aor_rows.append({"Predictor":k,"aOR":round(_or[k],2),
                     "CI_low":round(_ci.loc[k,0],2),"CI_high":round(_ci.loc[k,1],2),
                     "p_value":round(_glm.pvalues[k],3)})
t_aor=pd.DataFrame(aor_rows)
print("Survey-weighted adjusted odds ratios (cluster-robust by PSU)"); display(t_aor); savecsv(t_aor,"table_adjusted_or.csv")

# --- Survey-weighted Table 1 by undiagnosed-hypertension status ---
_w=data["survey_weight"]
def _wmsd(col,mask):
    x=pd.to_numeric(data.loc[mask,col],errors="coerce"); ww=_w[mask]; v=x.notna(); x,ww=x[v],ww[v]
    mu=np.average(x,weights=ww); return mu,np.sqrt(np.average((x-mu)**2,weights=ww))
def _wpct(col,val,mask):
    x=data.loc[mask,col].astype(str); ww=_w[mask]; v=x.notna()
    return 100*ww[v][x[v]==str(val)].sum()/ww[v].sum()
_all=data["undiagnosed_htn"].notna(); _yes=data["undiagnosed_htn"]==1; _no=data["undiagnosed_htn"]==0
t1_rows=[]
for c in _cont:
    t1_rows.append({"Variable":c,
        "Overall":"%.1f (%.1f)"%_wmsd(c,_all),"Undiagnosed":"%.1f (%.1f)"%_wmsd(c,_yes),"Not undiagnosed":"%.1f (%.1f)"%_wmsd(c,_no)})
t1w=pd.DataFrame(t1_rows)
print("Survey-weighted Table 1 (continuous, mean (SD))"); display(t1w); savecsv(t1w,"table_1_weighted.csv")
print(f"Weighted undiagnosed prevalence: {100*_w[_yes].sum()/_w[_all].sum():.1f}% | n overall/yes/no = {int(_all.sum())}/{int(_yes.sum())}/{int(_no.sum())}")
