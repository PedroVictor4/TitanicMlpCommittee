import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from random import choice, random as rand_float, randint

# Bibliotecas Sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score,  confusion_matrix, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.neural_network import MLPClassifier

# Bibliotecas Keras/Tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2

# =============================================================================
# CONFIGURAÇÕES GLOBAIS E PARAMETRIZAÇÃO
# =============================================================================

# Garante que os caminhos são relativos ao local deste script .py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dados_originais')

# Caminhos absolutos
TRAIN_PATH = os.path.join(DATA_DIR, 'trainCorrigido.csv')
TEST_PATH  = os.path.join(DATA_DIR, 'test.csv')

# Parâmetros do Algoritmo Genético
GENERATIONS = 15      # Número de gerações
POP_SIZE = 10         # Tamanho da população
N_BEST = 5            # Tamanho do comitê (Top N indivíduos)

# Seed Global para reprodutibilidade
GLOBAL_SEED = 42

# Diretório para salvar imagens das matrizes de confusão
IMG_DIR = 'matrizes_confusao'
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

# Espaço de busca de hiperparâmetros
# Unificado para garantir justiça entre Keras e Sklearn
PARAM_SPACE = {
    'layers':     [1, 2],              # número de camadas escondidas
    'neurons1':   [16, 32, 64, 128, 256],  # neurônios na 1ª camada 
    'neurons2':   [16, 32, 64, 128, 256],   # neurônios na 2ª camada (se houver)
    # Batch size e Epochs definem o regime de treinamento
    'batch_size': [16, 32, 64],            
    'epochs':     [50, 100, 200],
    # Regularização L2 ("Equivalente" ao dropout do Keras ou alpha do Sklearn)
    'alpha':      [0.0001, 0.001, 0.01, 0.05, 0.1], 
    'activation': ['relu', 'tanh'],
    # Dropout específico para Keras (Sklearn MLP não tem dropout padrão, mas mantivemos a chave para o Keras usar)
    'dropout1':   [0.0, 0.2, 0.4, 0.5], 
    'dropout2':   [0.0, 0.2, 0.4, 0.5]
}
PARAM_KEYS = list(PARAM_SPACE.keys())

# =============================================================================
# PRÉ-PROCESSAMENTO
# =============================================================================

def preprocessData(path, fit_scaler=True, scaler=None, fit_encoder=True, encoders=None):
    """
    Preprocessa os dados (train ou test).
    Retorna: X (np.array), y (ou None), scaler, encoders (dicionário)
    """
    # Acrescentei a feature family_size e a feature Deck (de Cabin)
    # A feature Deck veio pelos slides do Saraiva junto com várias outras métricas
    # Que antes desconsiderávamos devido a forte correlação com outras features
    df = pd.read_csv(path)
    
    # cria feature family_size
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    
    # Transforma Cabin em Deck (letra inicial) e trata nulos
    if 'Cabin' in df.columns:
        df['Deck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'U')
    else:
        # Caso dataset de teste não tenha (raro, mas preventivo)
        df['Deck'] = 'U'
        
    # Trata Embarked
    df['Embarked'] = df['Embarked'].apply(lambda x: str(x)[0] if pd.notna(x) else 'U')

    # DEFINIÇÃO DE X:
    # Removi 'Name' e 'Ticket' pois o StandardScaler não funciona com strings arbitrárias.
    # Se precisar dessas colunas, elas devem ser tratadas (ex: TF-IDF) antes de entrar no X numérico.
    # Note que usamos 'Deck' aqui, não 'Cabin'.
    # Garantir que todas colunas existam (para o caso do teste)
    cols_needed = ['Pclass', 'Sex', 'Age', 'family_size', 'Fare', 'Deck', 'Embarked']
    for c in cols_needed:
        if c not in df.columns:
            df[c] = 0 # Valor default seguro
            
    X = df[cols_needed].copy()
    
    # Inputação básica de Age (Mediana)
    # Idealmente salvaria a mediana do treino, mas para simplificar conforme dataAnalysis.py original:
    X['Age'] = X['Age'].fillna(X['Age'].median())
    X['Fare'] = X['Fare'].fillna(X['Fare'].median()) # Test set as vezes tem Fare nulo

    # Inicializa dicionário de encoders se não existir
    if encoders is None:
        encoders = {}

    # --- Label Encoding SEX ---
    if fit_encoder:
        le_sex = LabelEncoder()
        X['Sex'] = le_sex.fit_transform(X['Sex'])
        encoders['Sex'] = le_sex
    else:
        # Usa o encoder salvo (safe transform logic)
        X['Sex'] = X['Sex'].apply(lambda s: safeLabelEncodeSingle(s, encoders['Sex']))

    # --- Label Encoding DECK ---
    if fit_encoder:
        le_deck = LabelEncoder()
        X['Deck'] = le_deck.fit_transform(X['Deck'])
        encoders['Deck'] = le_deck
    else:
        X['Deck'] = X['Deck'].apply(lambda s: safeLabelEncodeSingle(s, encoders['Deck']))

    # --- Label Encoding EMBARKED ---
    if fit_encoder:
        le_emb = LabelEncoder()
        X['Embarked'] = le_emb.fit_transform(X['Embarked'])
        encoders['Embarked'] = le_emb
    else:
        X['Embarked'] = X['Embarked'].apply(lambda s: safeLabelEncodeSingle(s, encoders['Embarked']))

    # Standard Scaling
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    # Extrai y se existir
    y = df['Survived'].values if 'Survived' in df.columns else None
    
    return X_scaled, y, scaler, encoders

def safeLabelEncodeSingle(value, encoder):
    """Auxiliar para evitar erro em label novo no teste"""
    try:
        return encoder.transform([value])[0]
    except ValueError:
        # Se valor não visto no treino, retorna o primeiro classe (ou trata como 0)
        return 0

# =============================================================================
# CONSTRUÇÃO DE MODELOS (BUILDERS)
# =============================================================================

def buildModelKeras(params, input_dim, seed=GLOBAL_SEED):
    """
    Constrói modelo Keras baseado nos hiperparâmetros.
    """
    # Garante reprodutibilidade (padronização com Scikit-Learn)
    tf.random.set_seed(seed)
    # Também configura semente do numpy/python locais se necessário, mas o global já cuida
    
    # Modelo sequencial para classificação binária
    model = Sequential()
    
    # Primeira camada densa com ReLU (ou a definida no params)
    model.add(Input(shape=(input_dim,)))  # Camada explícita de input
    model.add(Dense(
        params['neurons1'], 
        activation=params['activation'],
        kernel_regularizer=l2(params['alpha']) # Adicionado para equivalência ao alpha do sklearn
    ))
    # Dropout na primeira camada
    model.add(Dropout(params['dropout1']))

    if params['layers'] == 2:
        # Segunda camada densa (se houver)
        model.add(Dense(
            params['neurons2'], 
            activation=params['activation'],
            kernel_regularizer=l2(params['alpha'])
        ))
        # Dropout na segunda camada
        model.add(Dropout(params['dropout2']))

    # Camada de saída com sigmoide
    model.add(Dense(1, activation='sigmoid'))

    # Compila com Adam e binary_crossentropy
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def buildModelSklearn(params, seed=GLOBAL_SEED):
    """
    Constrói um modelo MLPClassifier do Scikit-Learn com parâmetros dinâmicos.
    """
    # Define a estrutura das camadas ocultas
    if params['layers'] == 2:
        hidden_layers = (params['neurons1'], params['neurons2'])
    else:
        hidden_layers = (params['neurons1'],)

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        
        # Parâmetros vindos do Genético
        activation=params['activation'], # Pelo o que eu li, precisa ser a mesma função de ativação para as duas camadas
        alpha=params['alpha'],           # Penalidade L2 ("Equivalente" ao dropout do Keras)
        
        solver='adam',
        batch_size=params['batch_size'],
        max_iter=params['epochs'],
        random_state=seed,
        early_stopping=True, 
        validation_fraction=0.1
    )
    
    return model

# =============================================================================
# ALGORITMO GENÉTICO
# =============================================================================

def generateIndividual():
    return [choice(PARAM_SPACE[k]) for k in PARAM_KEYS]

def mutate(ind, rate=0.1):
    return [
        choice(PARAM_SPACE[k]) if rand_float() < rate else v
        for v, k in zip(ind, PARAM_KEYS)
    ]

def crossover(p1, p2):
    pt = randint(1, len(p1) - 1)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

def toDict(ind):
    return {k: v for k, v in zip(PARAM_KEYS, ind)}

def evaluateIndividual(ind, X, y, backend='keras'):
    """
    Avalia um indivíduo usando validação cruzada (KFold).
    backend: 'keras' ou 'sklearn'
    """
    params = toDict(ind)
    kf = KFold(n_splits=3, shuffle=True, random_state=GLOBAL_SEED)
    scores = []
    
    # Para economizar tempo no GA, usamos menos epochs ou simplificações se necessário, 
    # mas o requisito é "tempo de processamento... igual". Manteremos o params['epochs'].
    
    for train_index, val_index in kf.split(X):
        X_t, X_v = X[train_index], X[val_index]
        y_t, y_v = y[train_index], y[val_index]
        
        if backend == 'keras':
            # Seed fixa para avaliação ser determinística dado o parametro
            model = buildModelKeras(params, input_dim=X.shape[1], seed=GLOBAL_SEED)
            # Verbose 0 para não poluir
            model.fit(X_t, y_t, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
            # Evaluate retorna [loss, acc]
            loss, acc = model.evaluate(X_v, y_v, verbose=0)
            scores.append(acc)
            
        elif backend == 'sklearn':
            model = buildModelSklearn(params, seed=GLOBAL_SEED)
            model.fit(X_t, y_t)
            acc = model.score(X_v, y_v)
            scores.append(acc)
            
    return np.mean(scores)

def geneticAlgorithm(X, y, backend='keras'):
    """
    Executa o GA para encontrar melhores hiperparâmetros.
    """
    print(f"\n[{backend.upper()}] Iniciando Algoritmo Genético ({GENERATIONS} gens, {POP_SIZE} pop)")
    
    pop = [generateIndividual() for _ in range(POP_SIZE)]
    hall_of_fame = [] # Guarda histórico de todos
    
    for g in range(GENERATIONS):
        # Avalia população
        scored = []
        for ind in pop:
            acc = evaluateIndividual(ind, X, y, backend=backend)
            scored.append((ind, acc))
            
        scored.sort(key=lambda x: x[1], reverse=True)
        hall_of_fame.extend(scored)
        
        # Elitismo: mantém os 2 melhores
        next_pop = [scored[0][0], scored[1][0]]
        
        # Gera filhos
        while len(next_pop) < POP_SIZE:
            parent1 = choice(scored)[0]
            parent2 = choice(scored)[0]
            child1, child2 = crossover(parent1, parent2)
            next_pop.extend([mutate(child1), mutate(child2)])
            
        pop = next_pop[:POP_SIZE]
        print(f"[{backend.upper()}] Geração {g+1}: Melhor Acc = {scored[0][1]:.4f}")
        
    # Seleciona e retorna os N melhores indivíduos ÚNICOS final
    hall_of_fame.sort(key=lambda x: x[1], reverse=True)
    
    unique_params = []
    seen_configs = set()
    
    for ind, score in hall_of_fame:
        # Cria tupla hashable dos valores para checar duplicidade
        cfg_tuple = tuple(ind)
        if cfg_tuple not in seen_configs:
            seen_configs.add(cfg_tuple)
            unique_params.append(toDict(ind))
            
        if len(unique_params) >= N_BEST:
            break
            
    return unique_params

# =============================================================================
# COMITÊ E PREDIÇÕES
# =============================================================================

def ensemblePredict(models, X, method='aggregation', backend='keras'):
    """
    Realiza predição do comitê.
    method: 'aggregation' (soma prob) ou 'majority' (votação)
    """
    all_preds = []
    
    for model in models:
        if backend == 'keras':
            # Keras retorna prob [n_samples, 1]
            p = model.predict(X, verbose=0)
        else:
            # Sklearn predict_proba retorna [n_samples, 2], pegamos a classe 1
            # Nesse caso, para o majority, eu poderia usar o predict.
            # Mas para facilitar eu uso a probabilidade e converto depois.
            # Lembrando que nesse caso o predict usa threshold 0.5 internamente.
            p = model.predict_proba(X)[:, 1].reshape(-1, 1)
        all_preds.append(p)
    
    # [n_models, n_samples, 1]
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

# =============================================================================
# AUXILIARES DE RELATÓRIO E PLOTAGEM
# =============================================================================

def saveMetricsAndPlot(y_true, y_pred, title, file_obj):
    """
    Calcula métricas, salva no arquivo aberto e gera plot da matriz.
    """
    # Calculando as métricas solicitadas
    # São calculadas pelo sklearn 
    # Mas isso não é um problema, pois ele só recebe y_true e y_pred
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Escreve no txt
    file_obj.write(f"\n--- {title} ---\n")
    file_obj.write(f"Acurácia:  {acc:.5f}\n")
    file_obj.write(f"F1-Score:  {f1:.5f}\n")
    file_obj.write(f"Precisão:  {prec:.5f}\n")
    file_obj.write(f"Recall:    {rec:.5f}\n")
    file_obj.write(f"MCC:       {mcc:.5f}\n")
    file_obj.write("Matriz de Confusão (Texto):\n")
    file_obj.write(str(cm) + "\n")
    
    # Gera imagem
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matriz de Confusão: {title}")
    plt.ylabel('Real')
    plt.xlabel('Predito')
    
    # Nome do arquivo conforme solicitado: Ex: decisorio_Keras_validacao.png
    safe_title = title.replace(" ", "_")
    path_img = os.path.join(IMG_DIR, f"{safe_title}.png")
    plt.savefig(path_img)
    plt.close()
    
    return acc
def saveHyperparams(filename, params_list, backend):
    """Salva os hiperparâmetros ordenados em arquivo"""
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(filename, mode) as f:
        f.write(f"\n=== MELHORES HIPERPARÂMETROS: {backend.upper()} ===\n")
        for i, p in enumerate(params_list):
            f.write(f"Indivíduo {i+1}: {p}\n")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # 1. Carrega e Preprocessa Treino
    print("Carregando dados...")
    # fit_scaler=True, fit_encoder=True pois é o treino inicial
    # Debug: Mostra onde o script acha que está e onde está procurando os arquivos
    print(f"Diretório Base (Script): {BASE_DIR}")
    print(f"Procurando Treino em:    {TRAIN_PATH}")
    print(f"Procurando Teste em:     {TEST_PATH}")
    X, y, scaler, encoders = preprocessData(TRAIN_PATH, fit_scaler=True, fit_encoder=True)
    
    # Separa validação para decidir qual método de comitê é melhor (Majority vs Aggregation)
    # IMPORTANTE: random_state=42 para garantir split IDÊNTICO ao do Scikit-Learn
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=GLOBAL_SEED)
    
    # Prepara dados de submissão (Carrega o test.csv sem labels)
    if os.path.exists(TEST_PATH):
        df_test_raw = pd.read_csv(TEST_PATH)
        passenger_ids = df_test_raw['PassengerId'].values
        # Preprocessa Teste (USANDO scaler e encoders DO TREINO)
        X_test_sub, _, _, _ = preprocessData(TEST_PATH, fit_scaler=False, scaler=scaler, fit_encoder=False, encoders=encoders)
    else:
        print(f"AVISO: {TEST_PATH} não encontrado. Submissão não será gerada.")
        passenger_ids = None
        X_test_sub = None

    # Arquivos de saída
    FILE_PARAMS = "melhores_hiperparametros.txt"
    # Limpa arquivo de params se existir para não acumular rodadas anteriores
    if os.path.exists(FILE_PARAMS): os.remove(FILE_PARAMS)
    
    # ---------------------------------------------------------
    # LOOP PARA AS DUAS BIBLIOTECAS (KERAS E SKLEARN)
    # ---------------------------------------------------------
    backends = ['keras', 'sklearn']
    
    for backend in backends:
        print(f"\n{'='*40}")
        print(f"INICIANDO PIPELINE: {backend.upper()}")
        print(f"{'='*40}")
        
        # A. Algoritmo Genético
        best_params_list = geneticAlgorithm(X_train, y_train, backend=backend)
        
        # B. Salvar Hiperparâmetros
        saveHyperparams(FILE_PARAMS, best_params_list, backend)
        
        # C. Treinar Comitê (Top N)
        models = []
        print(f"\nTreinando Comitê de {N_BEST} modelos ({backend})...")
        
        # Cria diretório para salvar modelos se não existir
        MODELS_DIR = "modelos_salvos"
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)

        for i, params in enumerate(best_params_list):
            # Seed dinâmica (42, 43...) para diversidade de inicialização
            current_seed = GLOBAL_SEED + i
            
            # Nome do arquivo para salvar
            # Ex: modelos_salvos/modelo_keras_0.h5
            model_filename = os.path.join(MODELS_DIR, f"modelo_{backend}_{i}") 
            
            if backend == 'keras':
                model = buildModelKeras(params, input_dim=X_train.shape[1], seed=current_seed)
                model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                
                # Salva modelo Keras (.h5)
                model.save(f"{model_filename}.h5")
                
            else: # sklearn
                model = buildModelSklearn(params, seed=current_seed)
                model.fit(X_train, y_train)
                
                # Salva modelo Sklearn (.pkl)
                with open(f"{model_filename}.pkl", 'wb') as f:
                    pickle.dump(model, f)
                
            models.append(model)
            print(f"Modelo {i+1} salvo em: {model_filename}")
            
        # D. Avaliação na Validação (Decidir Melhor Comitê)
        preds_majority = ensemblePredict(models, X_val, method='majority', backend=backend)
        preds_aggregation = ensemblePredict(models, X_val, method='aggregation', backend=backend)
        
        # Arquivo de métricas específico da lib
        FILE_METRICS = f"metricas_validacao_{backend}.txt"
        with open(FILE_METRICS, 'w') as f_met:
            f_met.write(f"Relatório de Validação - {backend.upper()}\n")
            
            # Avalia Majority
            title_maj = f"decisorio_{backend}_validacao" # Nome para arquivo imagem
            acc_maj = saveMetricsAndPlot(y_val, preds_majority, title_maj, f_met)
            
            # Avalia Aggregation
            title_agg = f"soma_de_probabilidades_{backend}_validacao"
            acc_agg = saveMetricsAndPlot(y_val, preds_aggregation, title_agg, f_met)
            
            # Decide vencedor
            if acc_agg >= acc_maj:
                winner_method = 'aggregation'
                f_met.write("\n>> VENCEDOR: Soma de Probabilidades (Aggregation)\n")
            else:
                winner_method = 'majority'
                f_met.write("\n>> VENCEDOR: Decisório (Majority)\n")

        print(f"Métricas salvas em {FILE_METRICS}. Gráficos em {IMG_DIR}/")
        
        # E. Gerar Submissão (Make Submission) para TODOS os cenários
        if passenger_ids is not None:
            methods_to_submit = ['majority', 'aggregation']
            for method in methods_to_submit:
                # Predição no Teste
                final_preds = ensemblePredict(models, X_test_sub, method=method, backend=backend)
                
                # Nomenclatura descritiva
                # Ex: submission_keras_majority.csv
                sub_filename = f"submission_{backend}_{method}.csv"
                
                submission = pd.DataFrame({
                    'PassengerId': passenger_ids,
                    'Survived': final_preds
                })
                submission.to_csv(sub_filename, index=False)
                print(f"Arquivo gerado: {sub_filename}")

    print("\n\nProcesso finalizado com sucesso!")
    print(f"Hiperparâmetros salvos em: {FILE_PARAMS}")