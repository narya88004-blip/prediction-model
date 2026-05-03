"""
FirstPass v2 — XGBoost RFT Prediction Engine
=============================================
Predicts Right-First-Time (Approved/Rejected) for apparel samples.

FEATURE LOGIC
─────────────
STYLE DIFFICULTY   — Garment, Fit type, Fabric Type, Stretch, Ease, Block Pattern, Sealer
CONSTRUCTION       — Pattern pieces, seam length, complexity score, vertices/piece
PROCESS SIGNALS    — TAT Days (strongest, AUC 0.61), Sample Round, RFT% Rolling
INTERACTION FEATS  — garment×fabric, garment×fit, fabric difficulty, block RFT, ease RFT

EXCLUDED (vendor scorecards constant within vendor, AUC ≈ 0.50–0.53, variance ratio ≈ 0.001):
  AQL, DHU%, Cut to Ship, Fabric Utilization, Lead Time, MOQ, On-time, Replenishment
  → These are vendor profile numbers copied to every row. Zero per-sample signal.
  → They are still shown in UI as vendor context but NOT fed to the model.
  Vendors (direct name) excluded — rolling RFT% captures history without identity lock.
  Season excluded — rolling RFT already encodes seasonal capacity effects.
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, accuracy_score, roc_curve, confusion_matrix)
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

DATA_PATH  = os.path.join(os.path.dirname(__file__), 'data.xlsx')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
BASELINE   = 72.0

# ── Feature lists ─────────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = ['Garment','Fit type','Fabric Type','Fabric Stretch Category',
                        'Ease Allowance Category','Block Pattern Name','Sealer']
NUMERIC_FEATURES     = ['Sample Round','TAT Days','RFT% (Computed Rolling)',
                        'No. of Pattern Pieces (DXF)','Total Seam Length (mm)',
                        'Pattern Complexity Score','Avg Vertices/Piece']
INTERACTION_FEATURES = ['garment_fabric_rft','garment_fit_rft','fabric_difficulty',
                        'block_pattern_rft','ease_rft','complexity_tier','tat_risk_tier']
TARGET = 'Status'

# ── Style difficulty tables (computed from full dataset) ──────────────────────
FABRIC_DIFFICULTY = {'Stretch Denim':62.7,'Viscose Blend':66.1,'Stretch Twill':66.2,
    'Rigid Denim':69.4,'Cotton Jersey':70.6,'Cotton Twill':71.8,'Rib Knit':72.3,
    'Viscose Woven':72.3,'Cotton Woven':73.5,'Single Jersey':75.2,'French Terry':75.6,
    'Woven Fabric':79.3,'Stretch Woven':83.3}
GARMENT_FABRIC_RFT = {'Bottoms|Cotton Twill':71.8,'Bottoms|Stretch Twill':66.2,
    'Denim|Rigid Denim':69.4,'Denim|Stretch Denim':62.7,'Dress|Viscose Blend':66.1,
    'Dress|Woven Fabric':79.3,'Shirt|Cotton Woven':73.5,'Shirt|Stretch Woven':83.3,
    'Shirt|Viscose Woven':76.6,'T shirt|Cotton Jersey':70.7,'T shirt|Rib Knit':71.7,
    'T shirt|Single Jersey':76.8,'Top|Cotton Jersey':70.6,'Top|French Terry':75.6,
    'Top|Rib Knit':74.3,'Top|Single Jersey':74.8,'Top|Viscose Woven':70.8}
GARMENT_FIT_RFT = {'Bottoms|Regular':76.6,'Bottoms|Relaxed':76.0,'Bottoms|Slim Fit':66.7,
    'Bottoms|Straight Fit':69.9,'Bottoms|Wide Leg':65.3,'Denim|Regular':70.1,
    'Denim|Slim Fit':64.3,'Denim|Straight Fit':70.4,'Denim|Wide Leg':66.7,
    'Dress|Boxy':76.3,'Dress|Regular':81.2,'Dress|Relaxed':80.4,'Dress|Slim Fit':66.1,
    'Shirt|Boxy':71.9,'Shirt|Oversized':84.8,'Shirt|Regular':72.9,'Shirt|Relaxed':78.0,
    'T shirt|Boxy':68.5,'T shirt|Oversized':75.4,'T shirt|Regular':72.3,
    'Top|Boxy':70.2,'Top|Oversized':72.0,'Top|Regular':70.8,'Top|Relaxed':73.1,'Top|Slim Fit':66.7}
BLOCK_PATTERN_RFT = {'Mini Flare':72.2,'Mom Fit':75.0,'Skinny Fit':65.7,'Straight Fit':72.4,
    'T-Shirt Fitted Rib':76.7,'T-Shirt Relaxed DS':73.9,'T-Shirt Relaxed FS':72.2,'Wide Leg':66.1}
EASE_RFT = {'Oversized (>20%)':74.3,'Relaxed (10\u201320%)':73.0,'Relaxed (10-20%)':73.0,
    'Standard (5\u201310%)':71.4,'Standard (5-10%)':71.4,'Tight (0\u20135%)':65.8,'Tight (0-5%)':65.8}
SEALER_RFT = {'Blue Seal':69.8,'Development Sample':73.8,'Green Seal':72.0,'Silver Seal':77.4}

def _cx_tier(s):
    try: s=float(s)
    except: return 2
    if np.isnan(s): return 2
    if s<1.5: return 0
    if s<2.5: return 1
    if s<3.5: return 2
    if s<4.5: return 3
    return 4

def _tat_tier(t):
    try: t=float(t)
    except: return 2
    if np.isnan(t): return 2
    if t<=1: return 4
    if t<=2: return 3
    if t<=3: return 2
    if t<=5: return 1
    return 0

def engineer(df):
    df = df.copy()
    G  = df.get('Garment', pd.Series(['']*len(df))).fillna('').astype(str)
    FT = df.get('Fit type', pd.Series(['']*len(df))).fillna('').astype(str)
    FA = df.get('Fabric Type', pd.Series(['']*len(df))).fillna('').astype(str)
    BL = df.get('Block Pattern Name', pd.Series(['']*len(df))).fillna('').astype(str)
    EA = df.get('Ease Allowance Category', pd.Series(['']*len(df))).fillna('').astype(str)
    CX = pd.to_numeric(df.get('Pattern Complexity Score', pd.Series([np.nan]*len(df))), errors='coerce')
    TA = pd.to_numeric(df.get('TAT Days', pd.Series([np.nan]*len(df))), errors='coerce')
    df['garment_fabric_rft'] = [GARMENT_FABRIC_RFT.get(f'{g}|{f}',BASELINE) for g,f in zip(G,FA)]
    df['garment_fit_rft']    = [GARMENT_FIT_RFT.get(f'{g}|{f}',BASELINE) for g,f in zip(G,FT)]
    df['fabric_difficulty']  = [FABRIC_DIFFICULTY.get(f,BASELINE) for f in FA]
    df['block_pattern_rft']  = [BLOCK_PATTERN_RFT.get(b,BASELINE) for b in BL]
    df['ease_rft']           = [EASE_RFT.get(e,BASELINE) for e in EA]
    df['complexity_tier']    = [_cx_tier(s) for s in CX]
    df['tat_risk_tier']      = [_tat_tier(t) for t in TA]
    return df

def load_and_prepare(path=DATA_PATH):
    df = pd.read_excel(path, sheet_name='ML_Training_Data', header=1)
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].str.strip().map({'Approved':1,'Rejected':0})
    df = df.dropna(subset=[TARGET])
    df = engineer(df)
    all_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES + INTERACTION_FEATURES
    feature_cols = [c for c in all_cols if c in df.columns]
    X = df[feature_cols].copy(); y = df[TARGET].astype(int)
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col not in X.columns: continue
        le = LabelEncoder()
        X[col] = X[col].fillna('Unknown').astype(str)
        X[col] = le.fit_transform(X[col]); encoders[col] = le
    for col in NUMERIC_FEATURES + INTERACTION_FEATURES:
        if col not in X.columns: continue
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].median())
    return X, y, encoders, df, feature_cols

def _make_xgb(y_train):
    spw = float((y_train==1).sum()) / max(float((y_train==0).sum()),1)
    return XGBClassifier(n_estimators=700,max_depth=5,learning_rate=0.03,
        subsample=0.8,colsample_bytree=0.75,min_child_weight=10,
        gamma=0.15,reg_alpha=0.2,reg_lambda=1.5,scale_pos_weight=spw,
        eval_metric='logloss',early_stopping_rounds=50,random_state=42,n_jobs=-1)

def train_model(X, y):
    Xtr,Xv,ytr,yv = train_test_split(X,y,test_size=0.15,stratify=y,random_state=42)
    m = _make_xgb(ytr); m.fit(Xtr,ytr,eval_set=[(Xv,yv)],verbose=False); return m

def evaluate_oof(X, y):
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    oof = np.zeros(len(y))
    for tr,val in cv.split(X,y):
        m = _make_xgb(y.iloc[tr]); m.set_params(early_stopping_rounds=None,n_estimators=500)
        m.fit(X.iloc[tr],y.iloc[tr],verbose=False); oof[val] = m.predict_proba(X.iloc[val])[:,1]
    fpr,tpr,thrs = roc_curve(y,oof)
    thresh = float(thrs[np.argmax(tpr-fpr)])
    pred = (oof>=thresh).astype(int)
    return {'auc_roc':round(float(roc_auc_score(y,oof)),4),
            'accuracy':round(float(accuracy_score(y,pred)),4),
            'f1':round(float(f1_score(y,pred,zero_division=0)),4),
            'precision':round(float(precision_score(y,pred,zero_division=0)),4),
            'recall':round(float(recall_score(y,pred,zero_division=0)),4),
            'f1_rejected':round(float(f1_score(y,pred,pos_label=0,zero_division=0)),4),
            'recall_rejected':round(float(recall_score(y,pred,pos_label=0,zero_division=0)),4),
            'best_threshold':round(thresh,4),
            'cm':confusion_matrix(y,pred).tolist(),
            'roc_fpr':fpr.tolist()[::5],'roc_tpr':tpr.tolist()[::5],
            'oof_proba':oof.tolist(),'oof_labels':y.tolist()}

def train_and_save():
    print('Loading data...')
    X,y,encoders,df_eng,feature_cols = load_and_prepare()
    print(f'Dataset: {len(X)} rows | {len(feature_cols)} features')
    print(f'Balance: Approved={int(y.sum())} Rejected={int((y==0).sum())}')
    print('Training XGBoost...')
    model = train_model(X,y)
    print('5-fold OOF evaluation...')
    metrics = evaluate_oof(X,y)
    fi = pd.DataFrame({'feature':feature_cols,'importance':model.feature_importances_})
    fi = fi.sort_values('importance',ascending=False).to_dict('records')
    print(f'AUC={metrics["auc_roc"]}  F1={metrics["f1"]}  Recall(Rej)={metrics["recall_rejected"]}')
    print('Top features:')
    for r in fi[:10]: print(f'  {r["feature"]:40s} {r["importance"]:.4f}')

    # ── Build all dashboard lookup tables ─────────────────────────────────────
    df_raw = pd.read_excel(DATA_PATH, sheet_name='ML_Training_Data', header=1)
    df_raw['_approved'] = (df_raw['Status']=='Approved').astype(int)

    def rft(s): return round(float((s=='Approved').mean()*100),1) if len(s)>0 else BASELINE
    def num(col,s): return round(float(pd.to_numeric(s[col],errors='coerce').mean()),2) if len(s)>0 else 0.0

    # VS — per-vendor stats
    VS = {}
    for v in df_raw['Vendors'].dropna().unique():
        sub = df_raw[df_raw['Vendors']==v]
        VS[str(v)] = {
            'rft': rft(sub['Status']), 'rolling': round(float(pd.to_numeric(sub['RFT% (Computed Rolling)'],errors='coerce').mean()),1),
            'dhu': num('DHU%',sub), 'aql': num('AQL',sub), 'otd': num('On-time delivery',sub),
            'cut_ship': num('Cut to Ship ratio',sub), 'lead_time': num('Lead Time',sub),
            'moq': num('MOQ Flexibility',sub), 'replen': num('Replenishment Lead time',sub),
            'fabric_util': num('Fabric Utilization',sub), 'samples': int(len(sub))
        }

    # VG — vendor×garment RFT (≥3 samples)
    VG = {}
    for v in df_raw['Vendors'].dropna().unique():
        vsub = df_raw[df_raw['Vendors']==v]; VG[str(v)] = {}
        for g in vsub['Garment'].dropna().unique():
            gsub = vsub[vsub['Garment']==g]
            if len(gsub)>=3: VG[str(v)][str(g)] = rft(gsub['Status'])

    # VF — vendor×fabric RFT (≥3 samples)
    VF = {}
    for v in df_raw['Vendors'].dropna().unique():
        vsub = df_raw[df_raw['Vendors']==v]; VF[str(v)] = {}
        for f in vsub['Fabric Type'].dropna().unique():
            fsub = vsub[vsub['Fabric Type']==f]
            if len(fsub)>=3: VF[str(v)][str(f)] = rft(fsub['Status'])

    VG_CAP = {v: list(VG[v].keys()) for v in VG}
    VF_CAP = {v: list(VF[v].keys()) for v in VF}

    # GF — garment+fit RFT
    GF = {f'{g}|{f}': rft(sub['Status'])
          for (g,f),sub in df_raw.groupby(['Garment','Fit type'])
          if len(sub)>=3}

    # FAB — fabric overall RFT
    FAB = {f: rft(sub['Status']) for f,sub in df_raw.groupby('Fabric Type') if len(sub)>=3}

    # BP — block pattern stats
    BP = {}
    for b,sub in df_raw.groupby('Block Pattern Name'):
        if len(sub)>=3:
            BP[str(b)] = {
                'rft': rft(sub['Status']),
                'complexity': round(float(pd.to_numeric(sub['Pattern Complexity Score'],errors='coerce').mean()),2),
                'pieces': round(float(pd.to_numeric(sub['No. of Pattern Pieces (DXF)'],errors='coerce').mean()),0),
                'seam_length': round(float(pd.to_numeric(sub['Total Seam Length (mm)'],errors='coerce').mean()),0)
            }

    SEAL = {s: rft(sub['Status']) for s,sub in df_raw.groupby('Sealer') if len(sub)>=3}
    GARMENT_RFT = {g: rft(sub['Status']) for g,sub in df_raw.groupby('Garment') if len(sub)>=3}

    # SEASON_DATA
    SEASON_DATA = {}
    for s,sub in df_raw.groupby('Season'):
        SEASON_DATA[str(s)] = {'rft':rft(sub['Status']),'total':int(len(sub)),
            'approved':int((sub['Status']=='Approved').sum()),'rejected':int((sub['Status']=='Rejected').sum())}

    # Cascade data — all ≥3 samples
    FITS    = {g: sorted([f for f,s in sub.groupby('Fit type') if len(s)>=3])
               for g,sub in df_raw.groupby('Garment')}
    STRETS  = {g: sorted([s for s,sub2 in sub.groupby('Fabric Stretch Category') if len(sub2)>=3])
               for g,sub in df_raw.groupby('Garment')}
    EASES   = {g: sorted([e for e,sub2 in sub.groupby('Ease Allowance Category') if len(sub2)>=3])
               for g,sub in df_raw.groupby('Garment')}
    BLOCKS  = {g: sorted([b for b,sub2 in sub.groupby('Block Pattern Name') if len(sub2)>=3])
               for g,sub in df_raw.groupby('Garment')}

    # GSF — garment+stretch → valid fabrics (≥3 samples) — KEY cascade
    GSF = {}
    for (g,s),sub in df_raw.groupby(['Garment','Fabric Stretch Category']):
        fabs = [f for f,fsub in sub.groupby('Fabric Type') if len(fsub)>=3]
        if fabs: GSF[f'{g}|{s}'] = fabs

    # Vendor analytics
    rej_by_vendor = (df_raw.groupby('Vendors')['Status']
        .apply(lambda x: round((x=='Rejected').mean()*100,1)).reset_index()
        .rename(columns={'Status':'rej_rate','Vendors':'vendor'})
        .sort_values('rej_rate',ascending=False).to_dict('records'))
    rej_by_garment = (df_raw.groupby('Garment')['Status']
        .apply(lambda x: round((x=='Rejected').mean()*100,1)).reset_index()
        .rename(columns={'Status':'rej_rate','Garment':'garment'}).to_dict('records'))
    rej_by_round = (df_raw.groupby('Sample Round')['Status']
        .apply(lambda x: round((x=='Rejected').mean()*100,1)).reset_index()
        .rename(columns={'Status':'rej_rate','Sample Round':'round'}).to_dict('records'))
    rej_by_fabric = (df_raw.groupby('Fabric Type')['Status']
        .apply(lambda x: round((x=='Rejected').mean()*100,1)).reset_index()
        .rename(columns={'Status':'rej_rate','Fabric Type':'fabric'})
        .sort_values('rej_rate',ascending=False).to_dict('records'))

    # REJ — rejection reference keyed by fabric|garment
    df_rej = pd.read_excel(DATA_PATH, sheet_name='Rejection_Reference', header=1)
    df_rej_only = df_rej[df_rej['Status']=='Rejected'].copy()
    df_merged = pd.merge(df_rej_only,
        df_raw[['Season','Vendors','Sample Round','Garment','Fabric Type','Status']],
        on=['Season','Vendors','Sample Round','Status'], how='left')
    REJ = {}
    for _,row in df_merged.dropna(subset=['Garment','Fabric Type']).iterrows():
        key = f"{row['Fabric Type']}|{row['Garment']}"
        if key not in REJ: REJ[key] = {'reasons':[],'pom':[],'cnt':0}
        REJ[key]['cnt'] += 1
        r = str(row.get('Rejection Reason',''))
        p = str(row.get('Deviated Measurement Point',''))
        if r and r!='nan' and r not in REJ[key]['reasons']: REJ[key]['reasons'].append(r)
        if p and p!='nan' and p not in REJ[key]['pom']:     REJ[key]['pom'].append(p)

    artifacts = {
        # ML core
        'model': model, 'encoders': encoders, 'feature_cols': feature_cols,
        'METRICS': metrics, 'FEATURE_IMPORTANCE': fi,
        # Lookup tables
        'VS': VS, 'VG': VG, 'VF': VF, 'VG_CAP': VG_CAP, 'VF_CAP': VF_CAP,
        'GF': GF, 'FAB': FAB, 'BP': BP, 'SEAL': SEAL,
        'GARMENT_RFT': GARMENT_RFT, 'SEASON_DATA': SEASON_DATA,
        # Cascade
        'FITS': FITS, 'STRETS': STRETS, 'EASES': EASES, 'BLOCKS': BLOCKS, 'GSF': GSF,
        # Analytics
        'rej_by_vendor': rej_by_vendor, 'rej_by_garment': rej_by_garment,
        'rej_by_round': rej_by_round, 'rej_by_fabric': rej_by_fabric,
        'REJ': REJ,
        # Summary
        'total_records': int(len(df_raw)),
        'total_approved': int((df_raw['Status']=='Approved').sum()),
        'total_rejected': int((df_raw['Status']=='Rejected').sum()),
        'num_vendors': int(df_raw['Vendors'].nunique()),
        'num_garments': int(df_raw['Garment'].nunique()),
        'vendor_list': sorted(df_raw['Vendors'].dropna().unique().tolist()),
        'tech_list': sorted(df_raw['Technologist'].dropna().unique().tolist()),
    }
    with open(MODEL_PATH,'wb') as f: pickle.dump(artifacts,f)
    print(f'\n✓ Saved → {MODEL_PATH}')
    return artifacts

def load_artifacts():
    with open(MODEL_PATH,'rb') as f: return pickle.load(f)

def predict_single(data, arts):
    model=arts['model']; encoders=arts['encoders']; feature_cols=arts['feature_cols']
    row_df = engineer(pd.DataFrame([data]))
    row = {}
    for col in feature_cols:
        if col in CATEGORICAL_FEATURES and col in encoders:
            le=encoders[col]; val=str(data.get(col,'Unknown'))
            row[col] = int(le.transform([val])[0]) if val in le.classes_ else 0
        elif col in INTERACTION_FEATURES and col in row_df.columns:
            row[col] = float(row_df[col].iloc[0])
        else:
            raw=data.get(col,0); row[col]=float(raw) if raw not in ('',None) else 0.0
    proba = model.predict_proba(pd.DataFrame([row]))[0]
    thresh = arts['METRICS'].get('best_threshold',0.5)
    pred = int(proba[1]>=thresh)
    return {'prediction':'Approved' if pred else 'Rejected',
            'probability_approved':round(float(proba[1]),4),
            'probability_rejected':round(float(proba[0]),4),
            'confidence':round(float(max(proba)),4)}

def retrain_with_updated_data(): return train_and_save()

if __name__=='__main__': train_and_save()
