from sklearn.neural_network import MLPClassifier


# ToDo: Identificar o usdo o input_dim;
# Colcoquei o default como None para garatir que vai ser aleatorio
# Pelo o que vi, o sklearn ja randomiza internamente
def build_model(params,seed=None):
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