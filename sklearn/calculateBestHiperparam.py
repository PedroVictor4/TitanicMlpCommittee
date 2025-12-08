import numpy as np
from random import choice, random, randint
from sklearn.model_selection import KFold
import generateModel

# Espaço de busca de hiperparâmetros 
param_space = {
    'layers':     [1, 2],
    'neurons1':   [16, 32, 64, 128, 256],
    'neurons2':   [16, 32, 64, 128, 256],
    'batch_size': [16, 32, 64],
    'epochs':     [50, 100, 200],
    'alpha':      [0.0001, 0.001, 0.01, 0.05, 0.1], # Substitui o Dropout, penalizando neurônios com valores muito altos.
    'activation': ['relu', 'tanh']      # Testar ativações diferente (adicionei tanh para diversidade)
}
param_keys = list(param_space.keys())

def generate_individual():
    return [choice(param_space[k]) for k in param_keys]

def mutate(ind, rate=0.1):
    return [
        choice(param_space[k]) if random() < rate else v
        for v, k in zip(ind, param_keys)
    ]

def crossover(p1, p2):
    pt = randint(1, len(p1) - 1)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

def evaluate_individual(ind, X, y):
    params = {k: ind[i] for i, k in enumerate(param_keys)}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for tr_idx, va_idx in kf.split(X):
        # Constrói o modelo (já com batch_size e epochs configurados no init)
        model = generateModel.build_model(params, input_dim=X.shape[1])
        
        # Treina (Scikit-learn não usa verbose ou epochs no fit)
        model.fit(X[tr_idx], y[tr_idx])
        
        # Avalia (retorna acurácia)
        acc = model.score(X[va_idx], y[va_idx])
        scores.append(acc)
        
    return np.mean(scores)

def genetic_algorithm(X, y, generations=10, pop_size=10, n_best=5):
    pop = [generate_individual() for _ in range(pop_size)]
    
    # Lista para armazenar histórico de todos os individuos avaliados (para diversidade no final)
    hall_of_fame = []

    for g in range(generations):
        scored = [(ind, evaluate_individual(ind, X, y)) for ind in pop]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Salva histórico
        hall_of_fame.extend(scored)
        
        next_pop = [scored[0][0], scored[1][0]]
        while len(next_pop) < pop_size:
            parent1 = choice(scored)[0]
            parent2 = choice(scored)[0]
            child1, child2 = crossover(parent1, parent2)
            next_pop.extend([mutate(child1), mutate(child2)])
        pop = next_pop[:pop_size]
        print(f"Geração {g+1}: melhor acc = {scored[0][1]:.4f}")
        
    # Seleciona e retorna os N melhores indivíduos ÚNICOS final
    # N é o tamanho do comitê
    # Pega indivíduos com configurações de hiperparâmetros distintas
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