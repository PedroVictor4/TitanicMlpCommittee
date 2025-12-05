from sklearn.neural_network import MLPClassifier

def build_model(params, input_dim=4):
    """
    Constrói um modelo MLPClassifier do Scikit-Learn.
    Nota: input_dim é inferido automaticamente pelo sklearn no fit, 
    """
    
    # Define a estrutura das camadas ocultas
    if params['layers'] == 2:
        hidden_layers = (params['neurons1'], params['neurons2'])
    else:
        hidden_layers = (params['neurons1'],)

    # Scikit-learn recebe batch_size e epochs (max_iter) na construção
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        batch_size=params['batch_size'],
        max_iter=params['epochs'],
        random_state=42,
        # O sklearn não tem Dropout explícito, usa-se alpha (L2). 
        # Para manter compatibilidade com seu GA, só ignorei os params['dropout'].
        # To do modificar isso
        early_stopping=True, 
        validation_fraction=0.1
    )
    
    return model