#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os

DATA_PATH = r"C:\Users\admin\Desktop\WESAD DATASET\WESAD"

for dirname, _, filenames in os.walk(DATA_PATH):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Libraries

# In[4]:


import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ( accuracy_score,classification_report,confusion_matrix,roc_auc_score,f1_score)
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import warnings



# In[6]:


DATA_PATH = r"C:\Users\admin\Desktop\WESAD DATASET\WESAD"
print(os.listdir(DATA_PATH))
subjects = [s for s in os.listdir(DATA_PATH) if s.startswith("S")]
print(subjects)
all_data = []


# # Reading & Downsampling Data 

# In[7]:


for s in subjects:
    folder = os.path.join(DATA_PATH, s, s + "_E4_Data")

    # قراءة الملفات
    hr   = pd.read_csv(os.path.join(folder, "HR.csv"),  header=None, skiprows=2)
    ibi  = pd.read_csv(os.path.join(folder, "IBI.csv"), header=None, skiprows=1)
    eda  = pd.read_csv(os.path.join(folder, "EDA.csv"),  header=None, skiprows=2)
    temp = pd.read_csv(os.path.join(folder, "TEMP.csv"), header=None, skiprows=2)
    acc  = pd.read_csv(os.path.join(folder, "ACC.csv"),  header=None, skiprows=2)

    # ========================================================================================
    # Downsample to 4Hz
    # HR أصلاً 1Hz محتاج يترفع لـ 4Hz عن طريق repeat
    hr  = hr.loc[hr.index.repeat(4)].reset_index(drop=True)  # 1Hz -> 4Hz
    acc = acc.groupby(acc.index // 8).mean().reset_index(drop=True)  # 32Hz -> 4Hz
    eda = eda.groupby(eda.index // 1).mean().reset_index(drop=True)  # 4Hz as is

    # HRV من IBI
    ibi_values = ibi[1].values
    hrv_series = pd.Series(np.full(len(hr), np.std(ibi_values)))

    # ========================================================================================
    # Load labels
    pkl_path = os.path.join(DATA_PATH, s, f"{s}.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    raw_labels = data["label"]

    step = 175
    labels_ds = np.array([
        stats.mode(raw_labels[i:i+step], keepdims=False).mode.item()
        for i in range(0, len(raw_labels), step)
    ])

    # ========================================================================================
    # توحيد الطول
    final_len = min(len(hr), len(eda), len(temp), len(acc), len(labels_ds))

    hr        = hr.iloc[:final_len]
    eda       = eda.iloc[:final_len]
    temp      = temp.iloc[:final_len]
    acc       = acc.iloc[:final_len]
    hrv_series = hrv_series.iloc[:final_len]
    labels_ds = labels_ds[:final_len]

    # ========================================================================================
    # Build DataFrame
    df = pd.DataFrame({
        "HR":    hr[0].values,
        "HRV":   hrv_series.values,
        "EDA":   eda[0].values,
        "TEMP":  temp[0].values,
        "ACC_X": acc[0].values,
        "ACC_Y": acc[1].values,
        "ACC_Z": acc[2].values,
        "label": labels_ds,
        "subject": s
    })
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)


# In[8]:


df_all.head()


# In[9]:


df_all.describe()


# # Data Cleaning 

# In[10]:


df_all = df_all[(df_all["EDA"] >= 0) & (df_all["EDA"] <= 20)] 


# In[11]:


df_all = df_all[(df_all["TEMP"] >= 25) & (df_all["TEMP"] <= 40)]


# In[12]:


df_all = df_all[(df_all["HR"] >= 40) & (df_all["HR"] <= 200)]
df_all = df_all[(df_all["HRV"] >= 0) & (df_all["HRV"] <= 0.3)]


# In[13]:


for col in ["ACC_X", "ACC_Y", "ACC_Z"]:
    df_all = df_all[(df_all[col] >= -128) & (df_all[col] <= 127)]


# In[14]:


df_all = df_all[df_all["label"].isin([1, 2])].reset_index(drop=True)
print(df_all["label"].value_counts())     


# In[15]:


print("Final dataset shape:", df_all.shape)


# In[16]:





# > __________________________________________________________________________________________________________________________________________

# # Window Sliding 

# > ========================================================================================================================

# In[16]:


WINDOW_SIZE = 5 * 4  # 20 sample  [1,2,3,..100]  [50,.100]  10 windows 
STEP_SIZE   = 2 * 4  #1 ->8  

feature_cols = ["HR", "HRV", "EDA", "TEMP", "ACC_X", "ACC_Y", "ACC_Z"]


# In[17]:


all_windows = []
for subject in df_all["subject"].unique():
    df_sub     = df_all[df_all["subject"] == subject].reset_index(drop=True)
    labels_arr = df_sub["label"].values

    #============================================================================================    

    df_sub[feature_cols] = (df_sub[feature_cols] - df_sub[feature_cols].mean()) / df_sub[feature_cols].std()

    for start in range(0, len(df_sub) - WINDOW_SIZE, STEP_SIZE):
        end           = start + WINDOW_SIZE
        window        = df_sub.iloc[start:end]
        window_labels = labels_arr[start:end]                           #  [baisline , stress , stress ]  
        win_label = stats.mode(window_labels, keepdims=False).mode.item() #{stress, baislaine , stress}
        if win_label not in [1, 2]:
            continue

        dominant_ratio = np.sum(window_labels == win_label) / len(window_labels) 
        if dominant_ratio < 0.8:
            continue
        feats = {}
        for col in feature_cols:
            sig = window[col].values
            feats[f"{col}_mean"]   = np.mean(sig)
            feats[f"{col}_std"]    = np.std(sig)
            feats[f"{col}_min"]    = np.min(sig)
            feats[f"{col}_max"]    = np.max(sig)
            feats[f"{col}_range"]  = np.max(sig) - np.min(sig)
            feats[f"{col}_median"] = np.median(sig)
        feats["label"]   = win_label
        feats["subject"] = subject
        all_windows.append(feats)


# In[18]:


df_windows = pd.DataFrame(all_windows)


#   # Mapping
# 

# In[19]:


df_windows["risk"] = df_windows["label"].map({
    1: "Low Risk",
    2: "High Risk"
})


# In[20]:


print(f"Total windows: {len(df_windows)}")
print(df_windows["risk"].value_counts())


# In[21]:


feature_col = [c for c in df_windows.columns if c not in ["label", "subject", "risk"]]

X      = df_windows[feature_col].values
y      = df_windows["label"].map({1: 0, 2: 1}).values
groups = df_windows["subject"].values

n_neg = np.sum(y == 0)
n_pos = np.sum(y == 1)
scale_pos_weight = n_neg / n_pos
print(f"scale_pos_weight = {scale_pos_weight:.2f}")


# > _____________________________________________________________________________________________________________________________

# # Data Visalization 
# 

# > _____________________________________________________________________________________________________________________________

# # Signal Distribution  

# In[23]:


signals = ["BVP", "EDA", "TEMP"]

for sig in signals:
    fig = px.histogram(
        df_all,
        x=sig,
        nbins=50,
        color="label",     
        marginal="box",     
        title=f"Distribution of {sig} by Label"
    )
    fig.show()


# In[ ]:


fig = px.scatter(
    df_windows.sample(1000),
    x="BVP_mean",
    y="EDA_mean",
    color="risk",
    title="BVP vs EDA with Trendline"
)
fig.show()


# In[ ]:


df_windows["time_idx"] = range(len(df_windows))

fig = px.line(
    df_windows,
    x="time_idx",
    y="EDA_mean",
    color="risk",
    hover_data=["subject", "label"],
    title="EDA_mean over time by Risk"
)
fig.show()


# In[ ]:


cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[ ]:


fig = px.imshow(
    cm,
    text_auto=True,
    color_continuous_scale="Blues",
    labels=dict(x="Predicted", y="Actual"),
    x=["Low Risk", "High Risk"],
    y=["Low Risk", "High Risk"],
    title="Confusion Matrix"
)

fig.show()


# > _____________________________________________________________________________________________________________________________

# # XGBOOST Prepration 

# In[24]:


no_s17      = groups != "S17"
X_xgb       = X[no_s17]
y_xgb       = y[no_s17]
groups_xgb  = groups[no_s17]


# In[25]:


pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ))
])


# In[26]:


unique_subjects = np.unique(groups_xgb)
results = []

for test_subject in unique_subjects: #s2 
    train_mask = groups_xgb != test_subject # كل الداتا ما عدا s2 
    test_mask  = groups_xgb == test_subject

    X_train, y_train = X_xgb[train_mask], y_xgb[train_mask]
    X_test,  y_test  = X_xgb[test_mask],  y_xgb[test_mask]

    if len(np.unique(y_test)) < 2:
        print(f"{test_subject} → skipped")
        continue

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] 

    results.append({
        "subject"     : test_subject,
        "accuracy"    : (y_pred == y_test).mean(),
        "f1_macro"    : f1_score(y_test, y_pred, average="macro"),
        "f1_high_risk": f1_score(y_test, y_pred, pos_label=1),
        "roc_auc"     : roc_auc_score(y_test, y_prob)
    })

    print(f"{test_subject:4s} | "
          f"acc={results[-1]['accuracy']*100:5.1f}% | "
          f"f1={results[-1]['f1_macro']:.3f} | "
          f"AUC={results[-1]['roc_auc']:.3f}")


# In[27]:


df_r = pd.DataFrame(results)
print("="*50)
print(f"Accuracy  : {df_r['accuracy'].mean()*100:.1f}% ± {df_r['accuracy'].std()*100:.1f}%")
print(f"F1 macro  : {df_r['f1_macro'].mean():.3f} ± {df_r['f1_macro'].std():.3f}")
print(f"F1 high   : {df_r['f1_high_risk'].mean():.3f} ± {df_r['f1_high_risk'].std():.3f}")
print(f"ROC-AUC   : {df_r['roc_auc'].mean():.3f} ± {df_r['roc_auc'].std():.3f}")
print("="*50)


# In[28]:


print(df_r.sort_values("accuracy").head(14))


# In[30]:


print(classification_report(y_test, y_pred,
      target_names=["Low Risk", "High Risk"]))


# # Save The Model 

# In[31]:


import joblib

joblib.dump(pipeline, "panic_model.pkl")
print("Model saved!")


# In[32]:


import json

with open("feature_names.json", "w") as f:
    json.dump(feature_col, f)

print("Features saved!")


# In[36]:


joblib.dump(pipeline, "panic_model.pkl")

with open("feature_names.json", "w") as f:
    json.dump(feature_col, f)


# In[37]:


pipeline = joblib.load("panic_model.pkl")


# > =============================================================================================================================

# # Random Forest 

# In[38]:


mask_no_s17 = groups != "S17"

X_rf      = X[mask_no_s17]
y_rf      = y[mask_no_s17]
groups_rf = groups[mask_no_s17]


# In[39]:


pipeline_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])


# In[40]:


unique_subjects_rf = np.unique(groups_rf)
results_rf = []

for test_subject in unique_subjects_rf:
    train_mask = groups_rf != test_subject
    test_mask  = groups_rf == test_subject

    X_train, y_train = X_rf[train_mask], y_rf[train_mask]
    X_test,  y_test  = X_rf[test_mask],  y_rf[test_mask]

    if len(np.unique(y_test)) < 2:
        print(f"{test_subject} → skipped")
        continue

    pipeline_rf.fit(X_train, y_train)
    y_pred = pipeline_rf.predict(X_test)
    y_prob = pipeline_rf.predict_proba(X_test)[:, 1]

    results_rf.append({
        "subject"     : test_subject,
        "accuracy"    : (y_pred == y_test).mean(),
        "f1_macro"    : f1_score(y_test, y_pred, average="macro"),
        "f1_high_risk": f1_score(y_test, y_pred, pos_label=1),
        "roc_auc"     : roc_auc_score(y_test, y_prob)
    })

    print(f"{test_subject:4s} | "
          f"acc={results_rf[-1]['accuracy']*100:5.1f}% | "
          f"f1={results_rf[-1]['f1_macro']:.3f} | "
          f"AUC={results_rf[-1]['roc_auc']:.3f}")


# In[41]:


df_rf = pd.DataFrame(results_rf)

print("\n" + "="*50)
print(f"{'':15} {'RF':>10} {'XGBoost':>10}")
print("="*50)
print(f"{'Accuracy':15} {df_rf['accuracy'].mean()*100:>9.1f}%"
      f" {df_r['accuracy'].mean()*100:>9.1f}%")
print(f"{'F1 macro':15} {df_rf['f1_macro'].mean():>10.3f}"
      f" {df_r['f1_macro'].mean():>10.3f}")
print(f"{'F1 high':15} {df_rf['f1_high_risk'].mean():>10.3f}"
      f" {df_r['f1_high_risk'].mean():>10.3f}")
print(f"{'ROC-AUC':15} {df_rf['roc_auc'].mean():>10.3f}"
      f" {df_r['roc_auc'].mean():>10.3f}")
print("="*50)


# In[1]:


from fastapi import FastAPI
import joblib, json
import numpy as np
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

pipeline = joblib.load(r"C:\Users\admin\Downloads\graduation project\panic_model.pkl")

feature_names = json.load(open(r"C:\Users\admin\Downloads\graduation project\feature_names.json"))
WINDOW_SIZE = 20
THRESHOLD = 0.7

buffer = []
counter = 0
alarm_on = False


class SensorData(BaseModel):
    heart_rate: float
    gsr_value: float
    temperature: float
    ax: float
    ay: float
    az: float


# Feature Extraction
def extract_features(window):
    feats = {}

    cols = ["BVP", "EDA", "TEMP", "ACC_X", "ACC_Y", "ACC_Z"]
    df = pd.DataFrame(window, columns=cols)

    for col in cols:
        sig = df[col].values
        feats[f"{col}_mean"]   = np.mean(sig)
        feats[f"{col}_std"]    = np.std(sig)
        feats[f"{col}_min"]    = np.min(sig)
        feats[f"{col}_max"]    = np.max(sig)
        feats[f"{col}_range"]  = np.max(sig) - np.min(sig)
        feats[f"{col}_median"] = np.median(sig)

    return feats


@app.post("/predict")
def predict(data: SensorData):
    global buffer, counter, alarm_on

    bvp  = 60000 / data.heart_rate
    eda  = data.gsr_value
    temp = data.temperature

    row = [bvp, eda, temp, data.ax, data.ay, data.az]
    buffer.append(row)

    if len(buffer) < WINDOW_SIZE:
        return {
            "status": "collecting data",
            "buffer_size": len(buffer)
        }

    window = buffer[-WINDOW_SIZE:]

    feats = extract_features(window)

    df_feats = pd.DataFrame([feats])[feature_names]

    # prediction
    prob = pipeline.predict_proba(df_feats)[0][1]
    pred = int(prob > THRESHOLD)

    # alarm logic
    if prob > THRESHOLD:
        counter += 1
    else:
        counter = 0
        alarm_on = False

    alarm = False
    if counter >= 3 and not alarm_on:
        alarm = True
        alarm_on = True

    return {
        "risk": "High Risk" if pred == 1 else "Low Risk",
        "confidence": round(prob * 100, 2),
        "alarm": alarm,
        "buffer_size": len(buffer)
    }


# In[ ]:




