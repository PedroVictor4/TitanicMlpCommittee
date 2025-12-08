import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2

def build_model(params, input_dim=4, seed=42):
    # Garante reprodutibilidade (padronização com Scikit-Learn)
    tf.random.set_seed(seed)
    
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