import numpy as np
from random import choice, random, randint
from sklearn.model_selection import KFold
import generateModel

# Espaço de busca de hiperparâmetros
# Nota: Dropout existe aqui para não quebrar a geração de indivíduos, 
# mas não terá efeito no MLPClassifier padrão.
param_space = {
    'layers':     [1, 2],
    'neurons1':   [16, 32, 64, 128, 256],
    'neurons2':   [16, 32, 64, 128, 256],
    'dropout1':   [0.0, 0.2, 0.4, 0.6], 
    'dropout2':   [0.0, 0.2, 0.4, 0.6], 
    'batch_size': [16, 32, 64],
    'epochs':     [50, 100, 200] # Aumentei um pouco pois sklearn precisa de mais iterações
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

def genetic_algorithm(X, y, generations=10, pop_size=10):
    pop = [generate_individual() for _ in range(pop_size)]
    for g in range(generations):
        scored = [(ind, evaluate_individual(ind, X, y)) for ind in pop]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        next_pop = [scored[0][0], scored[1][0]]
        while len(next_pop) < pop_size:
            parent1 = choice(scored)[0]
            parent2 = choice(scored)[0]
            child1, child2 = crossover(parent1, parent2)
            next_pop.extend([mutate(child1), mutate(child2)])
        pop = next_pop[:pop_size]
        print(f"Geração {g+1}: melhor acc = {scored[0][1]:.4f}")
        
    best = max(pop, key=lambda ind: evaluate_individual(ind, X, y))
    return {k: best[i] for i, k in enumerate(param_keys)}