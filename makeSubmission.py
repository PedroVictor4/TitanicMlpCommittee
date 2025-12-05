import argparse, os, sys, pickle
import numpy as np
import pandas as pd


def safe_label_encode(series, encoder):
    mapping = {c: i for i, c in enumerate(encoder.classes_)}
    def map_val(v):
        if pd.isna(v): return 0
        vs = str(v)
        return mapping.get(vs, 0)
    return series.apply(map_val).astype(int).values

def ensemble_predict(models, X, method='soft'):
    """Lógica de predição Scikit-Learn"""
    all_preds = []
    for model in models:
        # USO DO PREDICT_PROBA EXIGIDO
        # Retorna array [n_samples, 2], pegamos coluna 1 (Survival)
        p = model.predict_proba(X)[:, 1]
        all_preds.append(p)
    
    all_preds = np.array(all_preds) 
    
    if method == 'soft':
        avg_proba = np.mean(all_preds, axis=0)
        final_pred = (avg_proba > 0.5).astype(int)
    else: 
        votes = (all_preds > 0.5).astype(int)
        sum_votes = np.sum(votes, axis=0)
        threshold = len(models) / 2
        final_pred = (sum_votes > threshold).astype(int)
        
    return final_pred

def main(args):
    # Checagens
    for p in (args.preproc, args.test):
        if not os.path.isfile(p):
            print(f"[ERRO] arquivo não encontrado: {p}")
            sys.exit(1)

    with open(args.preproc, "rb") as f:
        prep = pickle.load(f)

    scaler = prep.get('scaler')
    encoder = prep.get('encoder')
    age_median = prep.get('age_median')
    best_method = prep.get('best_method', 'soft')
    n_members = prep.get('n_members', 5)

    print(f"Carregando Comitê de {n_members} modelos (Scikit-Learn)...")
    models = []
    for i in range(n_members):
        # extensão .pkl
        fname = f"model_{i}.pkl" 
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                models.append(pickle.load(f))
        else:
            print(f"[AVISO] {fname} não encontrado.")
    
    if not models:
        print("[ERRO] Nenhum modelo carregado.")
        sys.exit(1)

    # Lógica de Features
    if age_median is None:
        if os.path.isfile('dados_originais/train.csv'):
            age_median = pd.read_csv('dados_originais/train.csv')['Age'].median()
    
    df = pd.read_csv(args.test)
    passenger_ids = df['PassengerId'].values

    if hasattr(scaler, 'feature_names_in_'):
        desired_cols = list(scaler.feature_names_in_)
    else:
        desired_cols = ['Pclass', 'Sex', 'Age', 'family_size']

    if 'family_size' in desired_cols and 'family_size' not in df.columns:
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            df['family_size'] = df['SibSp'] + df['Parch'] + 1
        else:
            df['family_size'] = 1

    if 'Age' in df.columns:
        if age_median is None: age_median = df['Age'].median()
        df['Age'] = df['Age'].fillna(age_median)
    else:
        df['Age'] = age_median if age_median else 0.0

    X_df = pd.DataFrame()
    for c in desired_cols:
        X_df[c] = df[c] if c in df.columns else 0

    if 'Sex' in X_df.columns:
        X_df['Sex'] = safe_label_encode(X_df['Sex'], encoder)

    X_df = X_df.astype(float)
    X_scaled = scaler.transform(X_df)

    # Predição
    print("Gerando submissão...")
    preds = ensemble_predict(models, X_scaled, method=best_method)

    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": preds
    })
    submission.to_csv(args.output, index=False)
    print(f"Salvo em {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preproc", default="preprocessing.pkl")
    p.add_argument("--test", default="dados_originais/test.csv")
    p.add_argument("--output", default="submission.csv")
    args = p.parse_args()
    main(args)