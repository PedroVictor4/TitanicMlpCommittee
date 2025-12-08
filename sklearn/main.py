import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Importações dos módulos
from prepross import preprocess_data
from generateModel import build_model
from calculateBestHiperparam import genetic_algorithm

def ensemble_predict(models, X, method='aggregation'):
    """
    Faz a predição usando o comitê de modelos Scikit-Learn.
    Usa EXPLICITAMENTE o predict_proba.
    """
    all_preds = []
    
    for model in models:
        # Pega a probabilidade da classe 1 (sobreviveu)
        # Retorno do predict_proba é [[prob_0, prob_1], ...]
        p = model.predict_proba(X)[:, 1]
        all_preds.append(p)
    
    # [n_models, n_samples]
    all_preds = np.array(all_preds)
    
    if method == 'aggregation':
        # Média das probabilidades
        avg_proba = np.mean(all_preds, axis=0)
        final_pred = (avg_proba > 0.5).astype(int)
        
    else: 
        # Votação baseada na probabilidade > 0.5 (majority Voting)
        votes = (all_preds > 0.5).astype(int)
        sum_votes = np.sum(votes, axis=0)
        threshold = len(models) / 2
        final_pred = (sum_votes > threshold).astype(int)
        
    return final_pred

def train(generations, pop_size):
    # Preprocessamento
    X, y, scaler, encoder = preprocess_data(
        'dados_originais/train.csv',
        fit_scaler=True,
        fit_encoder=True
    )
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"--- Iniciando Algoritmo Genético ({generations} gens, {pop_size} pop) ---")
    
    # Agora retorna uma LISTA com os 5 melhores conjuntos de parâmetros ÚNICOS
    best_params_list = genetic_algorithm(
        X_train, y_train,
        generations=generations,
        pop_size=pop_size,
        n_best=5
    )
    print("\nMelhores conjuntos de hiperparâmetros encontrados:")
    for p in best_params_list:
        print(p)

    # Treina o Comitê
    models = []
    print(f"\nTreinando Comitê de {len(best_params_list)} MLPs (Scikit-Learn)")
    
    for i, params in enumerate(best_params_list):
        print(f"Treinando modelo {i+1} com params: {params}...")
        current_seed = 42 + i
        model = build_model(params, input_dim=X_train.shape[1], seed = current_seed)
        model.fit(X_train, y_train) # fit simples no sklearn
        models.append(model)
        
        # Salva com pickle (sklearn não usa .h5)
        with open(f'model_{i}.pkl', 'wb') as f:
            pickle.dump(model, f)

    # Avalia as estratégias
    print("\nAvaliação do Comitê (Validação)")
    
    # votação por maioria
    preds_majority = ensemble_predict(models, X_val, method='majority')
    acc_majority = accuracy_score(y_val, preds_majority)
    print(f"Acurácia Votação Majoritária (Majority): {acc_majority:.5f}")
    
    # agregação (soma) das probabilidades
    preds_aggregation = ensemble_predict(models, X_val, method='aggregation')
    acc_aggregation = accuracy_score(y_val, preds_aggregation)
    print(f"Acurácia Soma Probabilidades (Aggregation): {acc_aggregation:.5f}")
    
    # Escolhe a melhor estratégia para salvar no pkl
    best_method = 'aggregation' if acc_aggregation >= acc_majority else 'majority'
    print(f"\n>> Melhor estratégia escolhida: {best_method.upper()}")
    
    df_raw_train = pd.read_csv('dados_originais/train.csv')
    age_median = df_raw_train['Age'].median()

    with open('preprocessing.pkl', 'wb') as f:
        pickle.dump({
            'scaler': scaler, 
            'encoder': encoder,
            'age_median': age_median,
            'best_method': best_method,
            'n_members': len(models)
        }, f)
        
    print("Treino concluído. Modelos (.pkl) e preprocessing.pkl salvos.")

def test():
    if not os.path.exists('preprocessing.pkl'):
        print("Erro: preprocessing.pkl não encontrado.")
        return

    with open('preprocessing.pkl', 'rb') as f:
        prep = pickle.load(f)
    scaler = prep['scaler']
    encoder = prep['encoder']
    best_method = prep.get('best_method', 'aggregation')
    n_members = prep.get('n_members', 5)
    
    print(f"Carregando comitê de {n_members} modelos...")
    models = []
    for i in range(n_members):
        fname = f'model_{i}.pkl'
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                models.append(pickle.load(f))
        else:
            print(f"Aviso: {fname} não encontrado.")

    if not models:
        return

    df_raw = pd.read_csv('dados_originais/test.csv')
    passenger_ids = df_raw['PassengerId'].values if 'PassengerId' in df_raw.columns else None
    has_labels = 'Survived' in df_raw.columns
    y_test = df_raw['Survived'].values if has_labels else None

    if has_labels:
        df_features = df_raw.drop(columns=['Survived'])
    else:
        df_features = df_raw

    df_features.to_csv('_tmp_test.csv', index=False)

    X_test, _, _, _ = preprocess_data(
        '_tmp_test.csv',
        fit_scaler=False, scaler=scaler,
        fit_encoder=False, encoder=encoder
    )

    print(f"Realizando predição usando método: {best_method.upper()}...")
    final_preds = ensemble_predict(models, X_test, method=best_method)

    if passenger_ids is not None:
        submission = pd.DataFrame({
            'PassengerId': passenger_ids,
            'Survived': final_preds
        })
        submission.to_csv('submission.csv', index=False)
        print("Arquivo 'submission.csv' gerado!")

    if has_labels:
        acc = accuracy_score(y_test, final_preds)
        print(f"\nAccuracy no teste: {acc:.4f}")

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    mode = input("Digite 'train' ou 'test': ").strip().lower()
    if mode == 'train':
        gens = int(input("Gerações [15]: ") or 15)
        pop = int(input("População [10]: ") or 10)
        train(gens, pop)
    elif mode == 'test':
        test()