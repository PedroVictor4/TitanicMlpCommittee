import argparse, os, sys, pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

def safe_label_encode(series, encoder):
    mapping = {c: i for i, c in enumerate(encoder.classes_)}
    def map_val(v):
        if pd.isna(v): return 0
        vs = str(v)
        return mapping.get(vs, 0)
    return series.apply(map_val).astype(int).values

def ensemble_predict(models, X, method='soft'):
    # Lógica de predição do comitê
    all_preds = []
    for model in models:
        # Previsão individual [n_samples, 1]
        p = model.predict(X, verbose=0)
        all_preds.append(p)
    
    all_preds = np.array(all_preds) # [n_models, n_samples, 1]
    
    if method == 'soft':
        # Média das probabilidades
        avg_proba = np.mean(all_preds, axis=0)
        final_pred = (avg_proba > 0.5).astype(int).reshape(-1)
    else: 
        # Votação majoritária
        votes = (all_preds > 0.5).astype(int)
        sum_votes = np.sum(votes, axis=0)
        threshold = len(models) / 2
        final_pred = (sum_votes > threshold).astype(int).reshape(-1)
        
    return final_pred

def main(args):
    # Verifica arquivos básicos
    for p in (args.preproc, args.test):
        if not os.path.isfile(p):
            print(f"[ERRO] arquivo não encontrado: {p}")
            sys.exit(1)

    print("Carregando preprocessing pickle...")
    with open(args.preproc, "rb") as f:
        prep = pickle.load(f)

    scaler = prep.get('scaler')
    encoder = prep.get('encoder')
    age_median = prep.get('age_median', None)
    
    # Recupera configurações do comitê salvas no treino
    best_method = prep.get('best_method', 'soft')
    n_members = prep.get('n_members', 5)

    if scaler is None or encoder is None:
        print("[ERRO] preprocessing.pkl corrompido.")
        sys.exit(1)

    # Carregamento do Comitê
    print(f"Carregando Comitê de {n_members} modelos (Método: {best_method})...")
    models = []
    for i in range(n_members):
        model_name = f"model_{i}.h5"
        if os.path.isfile(model_name):
            models.append(load_model(model_name))
        else:
            print(f"[AVISO] {model_name} não encontrado. O comitê ficará incompleto.")
    
    if not models:
        print("[ERRO] Nenhum modelo carregado.")
        sys.exit(1)

    # Tratamento da mediana (Fallback)
    if age_median is None:
        if os.path.isfile('dados_originais/train.csv'):
            age_median = pd.read_csv('dados_originais/train.csv')['Age'].median()
            print(f"[AVISO] Usando mediana de train.csv = {age_median:.4f}")
        else:
            print("[AVISO] Usando mediana do test.csv (risco de data leakage).")

    print("Lendo test.csv...")
    df = pd.read_csv(args.test)
    if 'PassengerId' not in df.columns:
        print("[ERRO] test.csv precisa conter PassengerId.")
        sys.exit(1)
    passenger_ids = df['PassengerId'].values

    # Reconstrução de features
    if hasattr(scaler, 'feature_names_in_'):
        desired_cols = list(scaler.feature_names_in_)
    else:
        desired_cols = ['Pclass', 'Sex', 'Age', 'family_size']

    # Engenharia de features
    if 'family_size' in desired_cols and 'family_size' not in df.columns:
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            df['family_size'] = df['SibSp'] + df['Parch'] + 1
        else:
            df['family_size'] = 1

    # Imputação de Age
    if 'Age' in df.columns:
        if age_median is None: age_median = df['Age'].median()
        df['Age'] = df['Age'].fillna(age_median)
    else:
        df['Age'] = age_median if age_median else 0.0

    # Montar DataFrame final na ordem correta
    X_df = pd.DataFrame()
    for c in desired_cols:
        X_df[c] = df[c] if c in df.columns else 0

    if 'Sex' in X_df.columns:
        X_df['Sex'] = safe_label_encode(X_df['Sex'], encoder)

    X_df = X_df.astype(float)
    X_scaled = scaler.transform(X_df)

    # Predição com Comitê
    print(f"Realizando predição com {len(models)} modelos via {best_method.upper()} voting...")
    preds = ensemble_predict(models, X_scaled, method=best_method)

    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": preds
    })
    submission.to_csv(args.output, index=False)
    print(f"Submissão salva em {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preproc", default="preprocessing.pkl")
    p.add_argument("--test", default="../dados_originais/test.csv")
    p.add_argument("--output", default="submission.csv")
    args = p.parse_args()
    main(args)