import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# Importações dos módulos
from prepross import preprocess_data
from generateModel import build_model
from calculateBestHiperparam import genetic_algorithm

def ensemble_predict(models, X, method='aggregation'):

    # Lista para armazenar probabilidades de cada modelo: [n_samples, 1]
    all_preds = []
    
    for model in models:
        # Probabilidade da classe 1
        p = model.predict(X, verbose=0)
        all_preds.append(p)
    
    # Converte para array numpy: [n_models, n_samples, 1]
    all_preds = np.array(all_preds)
    
    if method == 'aggregation':
        # Soma as probabilidades (ou média)
        avg_proba = np.mean(all_preds, axis=0) # [n_samples, 1]
        final_pred = (avg_proba > 0.5).astype(int).reshape(-1)
        
    else: # 'majority' -  Votação majoritária
        # Converte probas em votos (0 ou 1)
        votes = (all_preds > 0.5).astype(int)
        # Soma os votos
        sum_votes = np.sum(votes, axis=0) # [n_samples, 1]
        # Se mais da metade dos modelos votou 1
        threshold = len(models) / 2
        final_pred = (sum_votes > threshold).astype(int).reshape(-1)
        
    return final_pred

def train(generations, pop_size):
    # Carrega dados completos
    X, y, scaler, encoders = preprocess_data(
        '../dados_originais/trainCorrigido.csv',
        fit_scaler=True,
        fit_encoder=True
    )
    
    # Separa validação para decidir qual método de comitê é melhor (Majority vs Aggregation)
    # IMPORTANTE: random_state=42 para garantir split IDÊNTICO ao do Scikit-Learn
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Iniciando Algoritmo Genético ({generations} gens, {pop_size} pop)")
    # Busca hiperparâmetros usando apenas X_train para evitar data leakage
    # Retorna lista dos 5 melhores para o comitê
    best_params_list = genetic_algorithm(
        X_train, y_train,
        generations=generations,
        pop_size=pop_size,
        n_best=5
    )
    
    # Treina o Comitê (5 MLPs Heterogêneas)
    models = []
    print(f"\nTreinando Comitê de {len(best_params_list)} MLPs")
    
    for i, params in enumerate(best_params_list):
        # Seed dinâmica (42, 43...) para diversidade de inicialização gerando modelos diferentes para o comitê
        current_seed = 42 + i
        print(f"Treinando modelo {i+1} (Seed {current_seed})...")
        
        # Cria nova instância do modelo
        model = build_model(params, input_dim=X_train.shape[1], seed=current_seed)
        
        # Treina no conjunto de treino
        model.fit(
            X_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            verbose=0 # Silencioso para não poluir o terminal
        )
        models.append(model)
        # Salva cada modelo individualmente
        model.save(f'model_{i}.h5')

    # Avalia as estratégias no conjunto de Validação
    print("\nAvaliação do Comitê (Validação)")
    
    # Votação por maioria
    preds_majority = ensemble_predict(models, X_val, method='majority')
    acc_majority = accuracy_score(y_val, preds_majority)
    print(f"Acurácia Votação Majoritária (Majority): {acc_majority:.5f}")
    
    # Soma das probabilidades
    preds_aggregation = ensemble_predict(models, X_val, method='aggregation')
    acc_aggregation = accuracy_score(y_val, preds_aggregation)
    print(f"Acurácia Soma Probabilidades (Aggregation): {acc_aggregation:.5f}")
    
    # Decide o melhor método
    best_method = 'aggregation' if acc_aggregation >= acc_majority else 'majority'
    print(f"\n>> Melhor estratégia escolhida: {best_method.upper()}")
    
    df_raw_train = pd.read_csv('../dados_originais/trainCorrigido.csv')
    age_median = df_raw_train['Age'].median()

    # Salva os objetos de pré-processamento e a config do ensemble
    # Para uso posterior no teste/submissão
    with open('preprocessing.pkl', 'wb') as f:
        pickle.dump({
            'scaler': scaler, 
            'encoders': encoders,
            'age_median': age_median,
            'best_method': best_method,
            'n_members': len(models)
        }, f)
        
    print("Treino concluído. Modelos e preprocessing.pkl salvos.")

def test():
    # Carrega configurações
    if not os.path.exists('preprocessing.pkl'):
        print("Erro: preprocessing.pkl não encontrado. Execute 'train' primeiro.")
        return

    with open('preprocessing.pkl', 'rb') as f:
        prep = pickle.load(f)
    scaler = prep['scaler']
    encoders = prep['encoders']
    best_method = prep.get('best_method', 'aggregation')
    n_members = prep.get('n_members', 5)
    
    print(f"Carregando comitê de {n_members} modelos...")
    models = []
    for i in range(n_members):
        if os.path.exists(f'model_{i}.h5'):
            models.append(load_model(f'model_{i}.h5'))
        else:
            print(f"Aviso: model_{i}.h5 não encontrado.")

    if not models:
        print("Nenhum modelo carregado.")
        return

    # Carrega Dataset de Teste
    # Pegar o PassengerId para submissão
    df_raw = pd.read_csv('../dados_originais/test.csv')
    passenger_ids = df_raw['PassengerId'].values if 'PassengerId' in df_raw.columns else None

    # Verifica se tem label para calcular métricas (teste local) ou se é submissão cega
    has_labels = 'Survived' in df_raw.columns
    y_test = df_raw['Survived'].values if has_labels else None

    if has_labels:
        # Remove target para preprocessar igual ao treino
        df_features = df_raw.drop(columns=['Survived'])
    else:
        df_features = df_raw

    # Salva temp para usar o prepross.py existente
    df_features.to_csv('_tmp_test.csv', index=False)

    # Pré-processamento
    X_test, _, _, _ = preprocess_data(
        '_tmp_test.csv',
        fit_scaler=False, scaler=scaler,
        fit_encoder=False, encoders=encoders
    )

    # Predição com Comitê
    print(f"Realizando predição usando método: {best_method.upper()}...")
    final_preds = ensemble_predict(models, X_test, method=best_method)

    # Gera arquivo de submissão (Kaggle)
    if passenger_ids is not None:
        submission = pd.DataFrame({
            'PassengerId': passenger_ids,
            'Survived': final_preds
        })
        submission.to_csv('submission.csv', index=False)
        print("Arquivo 'submission.csv' gerado com sucesso!")

    # Métricas
    if has_labels:
        acc = accuracy_score(y_test, final_preds)
        report = classification_report(y_test, final_preds, digits=4)
        cm = confusion_matrix(y_test, final_preds)
        
        cm_df = pd.DataFrame(
            cm,
            index=['True: 0', 'True: 1'],
            columns=['Pred: 0', 'Pred: 1']
        )

        print(f"\nAccuracy no conjunto de teste: {acc:.4f}")
        print("\nClassification Report:")
        print(report)
        print("Matriz de Confusão:")
        print(cm_df)

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    
    mode = input("Digite 'train' para treinar ou 'test' para testar: ").strip().lower()

    if mode == 'train':
        gens = input("Número de gerações [default 15]: ").strip()
        pop = input("Tamanho da população [default 10]: ").strip()
        generations = int(gens) if gens.isdigit() else 15
        pop_size    = int(pop) if pop.isdigit() else 10
        train(generations, pop_size)

    elif mode == 'test':
        test()

    else:
        print("Opção inválida. Use 'train' ou 'test'.")