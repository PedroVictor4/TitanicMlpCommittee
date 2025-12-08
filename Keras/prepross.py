import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(path, fit_scaler=True, scaler=None, fit_encoder=True, encoder=None):
    """
    Preprocessa train.csv
    - Se fit_scaler/fit_encoder == True, ajusta e retorna scaler/encoder.
    - Caso contrário, reaplica os objetos fornecidos.
    Retorna: X (np.array), y (ou None), scaler, encoder
    """
    df = pd.read_csv(path)
    # cria feature family_size
    df['family_size'] = df['SibSp'] + df['Parch'] + 1

    X = df[['Pclass', 'Sex', 'Age', 'family_size']].copy()
    X['Age'] = X['Age'].fillna(X['Age'].median())

    # label encoding de Sex
    if fit_encoder:
        encoder = LabelEncoder()
        X['Sex'] = encoder.fit_transform(X['Sex'])
    else:
        X['Sex'] = encoder.transform(X['Sex'])

    # standard scaling
    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    # extrai y se existir
    y = df['Survived'].values if 'Survived' in df.columns else None
    return X, y, scaler, encoder