from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

def build_model(params, input_dim=4):
    # Modelo sequencial para classificação binária
    model = Sequential()
    # Primeira camada densa com ReLU
    model.add(Input(shape=(input_dim,)))  # Camada explícita de input
    model.add(Dense(params['neurons1'], activation='relu'))
    # Dropout na primeira camada
    model.add(Dropout(params['dropout1']))

    if params['layers'] == 2:
        # Segunda camada densa (se houver)
        model.add(Dense(params['neurons2'], activation='relu'))
        # Dropout na segunda camada
        model.add(Dropout(params['dropout2']))

    # Camada de saída com sigmoide
    model.add(Dense(1, activation='sigmoid'))

    # Compila com Adam e binary_crossentropy
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model