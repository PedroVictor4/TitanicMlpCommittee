import numpy as np
from random import choice, random, randint
from sklearn.model_selection import KFold
import gera_model
# Espaço de busca de hiperparâmetros
param_space = {
    'layers':     [1, 2],              # número de camadas escondidas
    'neurons1':   [16, 32, 64, 128, 256],   # neurônios na 1ª camada
    'neurons2':   [16, 32, 64, 128],   # neurônios na 2ª camada (se houver)
    'dropout1':   [0.0, 0.2, 0.4, 0.6], # taxa de dropout na 1ª camada
    'dropout2':   [0.0, 0.2, 0.4, 0.6], # taxa de dropout na 2ª camada
    'batch_size': [16, 32, 64],        # tamanho do lote
    'epochs':     [10, 20, 30, 40]         # número de épocas
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
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, va_idx in kf.split(X):
        model = gera_model.build_model(params, input_dim=X.shape[1])
        model.fit(X[tr_idx], y[tr_idx],
                  epochs=params['epochs'],
                  batch_size=params['batch_size'],
                  verbose=0)
        _, acc = model.evaluate(X[va_idx], y[va_idx], verbose=0)
        scores.append(acc)
    return np.mean(scores)

def genetic_algorithm(X, y, generations=10, pop_size=10):
    # Inicializa população com indivíduos aleatórios
    pop = [generate_individual() for _ in range(pop_size)]
    for g in range(generations):
        # Avalia todos os indivíduos
        scored = [(ind, evaluate_individual(ind, X, y)) for ind in pop]
        scored.sort(key=lambda x: x[1], reverse=True)
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
    # Seleciona e retorna o melhor indivíduo final
    best = max(pop, key=lambda ind: evaluate_individual(ind, X, y))
    return {k: best[i] for i, k in enumerate(param_keys)}
