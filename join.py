import pandas as pd

# Carregar os datasets
database = pd.read_csv('gender_submission.csv')  # Contém a coluna 'Survived'
test = pd.read_csv('test.csv')          # Não contém 'Survived'

# Verificar colunas disponíveis
print("Colunas do database:", database.columns.tolist())
print("Colunas do test:", test.columns.tolist())

# Realizar LEFT JOIN usando PassengerId como chave
# Traz apenas a coluna 'Survived' do database para o test
merged = pd.merge(
    left=test,
    right=database[['PassengerId', 'Survived']],  # Apenas colunas relevantes
    how='left',
    on='PassengerId'
)
# Reordenar as colunas: PassengerId, Survived, demais colunas
cols = merged.columns.tolist()  # Lista atual de colunas

# Remover 'Survived' da posição atual e inserir como segunda coluna
cols.remove('Survived')  # Remove da posição atual
cols.insert(1, 'Survived')  # Insere como segunda coluna (índice 1)

# Reorganizar o DataFrame com a nova ordem de colunas
merged = merged[cols]

# Verificar resultado
print("\nColunas após o join:", merged.columns.tolist())
print("\nExemplo de linhas:")
print(merged.head(3))

# Salvar resultado em novo arquivo CSV
merged.to_csv('test_com_survived.csv', index=False)
print("\nArquivo 'test_com_survived.csv' salvo com sucesso!")