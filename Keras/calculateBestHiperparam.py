import numpy as np
from random import choice, random, randint
from sklearn.model_selection import KFold
import generateModel

# Espaço de busca de hiperparâmetros
# Descrever a relação dos parâmetros Regularização L2 e Dropout
# Como ambos visam evitar que um neurônio se torne muito dominante
# Descrever o motivo de ter deixado o Droupout e a Regularização L2 juntos no espaço de busca 
# para o Keras mesmo que o skcit-learn não tenha Droupout
# Resumindo: Já que o Keras permite o uso de Dropout, vantagem para o Keras
param_space = {
    'layers':     [1, 2],              # número de camadas escondidas
    'neurons1':   [32, 64, 128, 256],  # neurônios na 1ª camada 
    'neurons2':   [16, 32, 64, 128],   # neurônios na 2ª camada (se houver)
    'dropout1':   [0.0, 0.2, 0.4, 0.5], # taxa de dropout na 1ª camada
    'dropout2':   [0.0, 0.2, 0.4, 0.5], # taxa de dropout na 2ª camada
    'batch_size': [16, 32],            # tamanho do lote
    'epochs':     [50, 100, 150],      # número de épocas
    'alpha':      [0.0, 0.001, 0.01, 0.05], # Regularização L2 
    'activation': ['relu', 'tanh']     # Funções de ativação 
}
param_keys = list(param_space.keys())

def generate_individual():
    # Cria um indivíduo (lista de valores) escolhendo aleatoriamente cada parâmetro
    return [choice(param_space[k]) for k in param_keys]

def mutate(ind, rate=0.1):
    # Para cada gene (valor) em ind, com probabilidade rate escolhe outro aleatório
    return [
        choice(param_space[k]) if random() < rate else v
        for v, k in zip(ind, param_keys)
    ]

def crossover(p1, p2):
    # Escolhe ponto de corte e cruza dois pais
    pt = randint(1, len(p1) - 1)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

def evaluate_individual(ind, X, y):
    # Converte lista em dicionário de parâmetros
    params = {k: ind[i] for i, k in enumerate(param_keys)}
    
    # Random_state=42 para ser IGUAL ao Scikit-Learn
    # IMPORTANTE para comparação justa de hiperparâmetros
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for tr_idx, va_idx in kf.split(X):
        # Seed=42 FIXA na busca para comparação justa de hiperparâmetros
        # Só depois no treinamento (main.py) que a seed precisa ser "aleatória" para diversidade do comitê
        model = generateModel.build_model(params, input_dim=X.shape[1], seed=42)
        
        model.fit(X[tr_idx], y[tr_idx],
                  epochs=params['epochs'],
                  batch_size=params['batch_size'],
                  verbose=0)
        _, acc = model.evaluate(X[va_idx], y[va_idx], verbose=0)
        scores.append(acc)
    return np.mean(scores)

def genetic_algorithm(X, y, generations=10, pop_size=10, n_best=5):
    # Inicializa população com indivíduos aleatórios
    pop = [generate_individual() for _ in range(pop_size)]
    
    # Histórico para diversidade
    hall_of_fame = []

    for g in range(generations):
        # Avalia todos os indivíduos
        scored = [(ind, evaluate_individual(ind, X, y)) for ind in pop]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        hall_of_fame.extend(scored)
        
        # Elitismo: mantém os dois melhores
        next_pop = [scored[0][0], scored[1][0]]
        # Gera filhos até completar a população
        while len(next_pop) < pop_size:
            parent1 = choice(scored)[0]
            parent2 = choice(scored)[0]
            child1, child2 = crossover(parent1, parent2)
            next_pop.extend([mutate(child1), mutate(child2)])
        pop = next_pop[:pop_size]
        print(f"Geração {g+1}: melhor acc = {scored[0][1]:.4f}")
    
    # Seleciona e retorna os N melhores indivíduos ÚNICOS final
    # N é o tamanho do comitê
    hall_of_fame.sort(key=lambda x: x[1], reverse=True)
    unique_params = []
    seen_configs = set()
    
    # Verificar se precisa desse score
    # Acho que foi pq usei esse algoritmo genético para um trabalho de regressão
    for ind, score in hall_of_fame:
        cfg_tuple = tuple(ind)
        if cfg_tuple not in seen_configs:
            seen_configs.add(cfg_tuple)
            unique_params.append({k: ind[i] for i, k in enumerate(param_keys)})
        if len(unique_params) >= n_best:
            break
            
    print(f"Comitê selecionado com {len(unique_params)} configurações distintas.")
    return unique_params

