import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(path, fit_scaler=True, scaler=None, fit_encoder=True, encoders=None):
    """
    Preprocessa trainCorrigidio.csv
    Retorna: X (np.array), y (ou None), scaler, encoders (dicionário)
    """
    df = pd.read_csv(path)
    
    # Cria feature family_size
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    
    # Transforma Cabin em Deck (letra inicial) e trata nulos
    df['Deck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'U')
    
    # Trata Embarked
    df['Embarked'] = df['Embarked'].apply(lambda x: str(x)[0] if pd.notna(x) else 'U')

    # DEFINIÇÃO DE X:
    # Removi 'Name' e 'Ticket' pois o StandardScaler não funciona com strings arbitrárias e pelo o que eu vi são apenas identificadores.
    # Se precisar dessas colunas, elas devem ser tratadas (ex: TF-IDF) antes de entrar no X numérico.
    X = df[['Pclass', 'Sex', 'Age', 'family_size', 'Fare', 'Deck', 'Embarked']].copy()
    # Fiz isso no dataAnalysis.py
    #X['Age'] = X['Age'].fillna(X['Age'].median())

    # Inicializa dicionário de encoders se não existir
    if encoders is None:
        encoders = {}

    # Label Encoding SEX
    if fit_encoder:
        le_sex = LabelEncoder()
        X['Sex'] = le_sex.fit_transform(X['Sex'])
        encoders['Sex'] = le_sex
    else:
        # Usa o encoder salvo
        X['Sex'] = encoders['Sex'].transform(X['Sex'])

    # Label Encoding DECK
    if fit_encoder:
        le_deck = LabelEncoder()
        X['Deck'] = le_deck.fit_transform(X['Deck'])
        encoders['Deck'] = le_deck
    else:
        X['Deck'] = encoders['Deck'].transform(X['Deck'])

    # Label Encoding EMBARKED
    if fit_encoder:
        le_emb = LabelEncoder()
        X['Embarked'] = le_emb.fit_transform(X['Embarked'])
        encoders['Embarked'] = le_emb
    else:
        X['Embarked'] = encoders['Embarked'].transform(X['Embarked'])

    # Standard Scaling
    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    # Extrai y se existir
    y = df['Survived'].values if 'Survived' in df.columns else None
    
    # Retorna o dicionário de encoders em vez de um único encoder sobrescrito
    return X, y, scaler, encoders