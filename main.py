# main.py

import pandas as pd
from prepross import preprocess_data
from gera_model import build_model
from calcula_best_hiperparam import genetic_algorithm
from tensorflow.keras.models import load_model
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
def train(generations, pop_size):
    X_train, y_train, scaler, encoder = preprocess_data(
        'train.csv',
        fit_scaler=True,
        fit_encoder=True
    )
    best_params = genetic_algorithm(
        X_train, y_train,
        generations=generations,
        pop_size=pop_size
    )
    print("Melhor conjunto de hiperparâmetros:", best_params)

    model = build_model(best_params, input_dim=X_train.shape[1])
    model.fit(
        X_train, y_train,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        verbose=1
    )

    model.save('best_model.h5')
    with open('preprocessing.pkl', 'wb') as f:
        pickle.dump({'scaler': scaler, 'encoder': encoder}, f)
    print("Treino concluído. Arquivos: best_model.h5, preprocessing.pkl")

def test():
    # 1) carrega modelo e pré-processadores
    model = load_model('best_model.h5')
    with open('preprocessing.pkl', 'rb') as f:
        prep = pickle.load(f)
    scaler, encoder = prep['scaler'], prep['encoder']

    # 2) carrega DataFrame de teste para extrair y_test
    df_test = pd.read_csv('test.csv')
    if 'Survived' not in df_test.columns:
        raise ValueError("O arquivo test.csv precisa conter a coluna 'Survived' para calcular métricas.")
    y_test = df_test['Survived'].values

    # remove a coluna alvo para pré-processar apenas as features
    df_features = df_test.drop(columns=['Survived'])
    df_features.to_csv('_tmp_test.csv', index=False)

    # 3) pré-processa X_test
    X_test, _, _, _ = preprocess_data(
        '_tmp_test.csv',
        fit_scaler=False,
        scaler=scaler,
        fit_encoder=False,
        encoder=encoder
    )

    # 4) faz previsões
    preds_proba = model.predict(X_test)
    preds = (preds_proba > 0.5).astype(int).reshape(-1)

    # 5) calcula métricas
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, digits=4)
    cm = confusion_matrix(y_test, preds)

    # 6) monta DataFrame com labels para matriz de confusão
    cm_df = pd.DataFrame(
        cm,
        index=['True: Não Sobreviveu (0)', 'True: Sobreviveu (1)'],
        columns=['Pred: Não Sobreviveu (0)', 'Pred: Sobreviveu (1)']
    )

    print(f"Accuracy no conjunto de teste: {acc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Matriz de Confusão:")
    print(cm_df)

if __name__ == '__main__':
    os.system('cls')
    mode = input("Digite 'train' para treinar ou 'test' para testar: ").strip().lower()

    if mode == 'train':
        gens = input("Número de gerações [default 10]: ").strip()
        pop = input("Tamanho da população [default 10]: ").strip()
        generations = int(gens) if gens.isdigit() else 10
        pop_size    = int(pop) if pop.isdigit() else 10
        train(generations, pop_size)

    elif mode == 'test':
        test()

    else:
        print("Opção inválida. Use 'train' ou 'test'.")
