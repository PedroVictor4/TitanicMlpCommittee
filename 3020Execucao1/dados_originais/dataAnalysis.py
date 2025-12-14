import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analisar_dados():
    arquivo = 'train.csv'
    try:
        df = pd.read_csv(arquivo)
    except FileNotFoundError:
        print(f"Arquivo {arquivo} não encontrado.")
        return

    print("Análise Inicial de Dados")
    print(f"Total de linhas: {df.shape[0]}")
    print(f"Total de colunas: {df.shape[1]}")

    # Verificar duplicatas
    duplicadas = df.duplicated().sum()
    print(f"\nLinhas duplicadas encontradas: {duplicadas}")
    if duplicadas > 0:
        resp = input("Deseja remover linhas duplicadas? (s/n): ").strip().lower()
        if resp == 's':
            df.drop_duplicates(inplace=True)
            print("Duplicatas removidas.")

    # Verificar valores nulos e interagir
    colunas_com_nulos = df.columns[df.isnull().any()].tolist()
    
    if colunas_com_nulos:
        print("\nColunas com valores faltantes encontradas.")
        for col in colunas_com_nulos:
            qtd_nulos = df[col].isnull().sum()
            print(f"\nColuna: {col} | Quantidade de nulos: {qtd_nulos}")
            
            print("Escolha uma ação:")
            print("1 - Excluir as linhas com valores nulos nesta coluna")
            print("2 - Substituir valores faltantes (Média para numéricos, Moda para categóricos)")
            print("3 - Ignorar")
            
            opcao = input("Opção: ").strip()
            
            if opcao == '1':
                df.dropna(subset=[col], inplace=True)
                print(f"Linhas com nulos em {col} excluídas.")
            elif opcao == '2':
                if pd.api.types.is_numeric_dtype(df[col]):
                    media = df[col].mean()
                    df[col] = df[col].fillna(media)
                    print(f"Nulos em {col} preenchidos com a média: {media:.2f}")
                else:
                    moda = df[col].mode()[0]
                    df[col] = df[col].fillna(moda)
                    print(f"Nulos em {col} preenchidos com a moda: {moda}")
            elif opcao == '3':
                print(f"Nenhuma ação tomada para {col}.")
            else:
                print("Opção inválida. Ignorando.")
    else:
        print("\nNão foram encontrados valores nulos no dataset.")

    # Salvar alterações
    df.to_csv('trainCorrigido.csv', index=False)
    print("\nArquivo salvo como 'tranCorrido.csv'.")

    # Heatmap de correlação
    print("\nGerando Heatmap de correlações...")
    plt.figure(figsize=(10, 8))
    
    # Selecionar apenas colunas numéricas para correlação de Pearson
    df_numerico = df.select_dtypes(include=[np.number])
    
    if not df_numerico.empty:
        correlacao = df_numerico.corr(method='pearson')
        sns.heatmap(correlacao, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Mapa de Calor - Correlação de Pearson")
        plt.show()
    else:
        print("Não há colunas numéricas suficientes para gerar o heatmap.")

if __name__ == "__main__":
    analisar_dados()