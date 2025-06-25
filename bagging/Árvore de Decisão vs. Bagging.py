import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# Configurações gerais
np.random.seed(42)
plt.style.use('ggplot')

## 1. Implementação da Árvore de Decisão (Base) - Integrante 1
def implementar_arvore(X_train, y_train, X_test):
    """
    Implementa uma árvore de decisão com alta variância (propensa a overfitting)
    
    Parâmetros:
    X_train, y_train: Dados de treino
    X_test: Dados de teste para fazer predições
    
    Retorna:
    preds: Predições da árvore
    model: Modelo treinado
    """
    # Árvore com hiperparâmetros para alta variância
    tree = DecisionTreeRegressor(
        max_depth=None,          # Profundidade máxima ilimitada
        min_samples_leaf=1,      # Mínimo 1 amostra por folha
        splitter='random',       # Divisões aleatórias para aumentar variância
        random_state=42
    )
    
    tree.fit(X_train, y_train)
    preds = tree.predict(X_test)
    
    return preds, tree

## 2. Implementação do Bagging - Integrante 2
def implementar_bagging(base_estimator, X_train, y_train, X_test, n_estimators=100):
    """
    Implementa ensemble com Bagging
    
    Parâmetros:
    base_estimator: Modelo base (árvore de decisão)
    X_train, y_train: Dados de treino
    X_test: Dados de teste
    n_estimators: Número de árvores no ensemble
    
    Retorna:
    preds: Predições do Bagging
    model: Modelo treinado
    """
    bagging = BaggingRegressor(
    estimator=base_estimator,  # Mudança aqui
    n_estimators=n_estimators,
    max_samples=0.8,
    random_state=42
)
    
    bagging.fit(X_train, y_train)
    preds = bagging.predict(X_test)
    
    return preds, bagging

## 3. Análise Estatística (Viés) - Integrante 3
def calcular_vies(y_true, y_pred):
    """
    Calcula o viés quadrático médio
    
    Parâmetros:
    y_true: Valores reais
    y_pred: Predições do modelo
    
    Retorna:
    vies: Viés quadrático médio
    """
    return np.mean((y_true - np.mean(y_pred))**2)

## 4. Análise Estatística (Variância) - Integrante 4
def calcular_variancia(y_pred):
    """
    Calcula a variância das predições
    
    Parâmetros:
    y_pred: Predições do modelo
    
    Retorna:
    variancia: Variância das predições
    """
    return np.var(y_pred)

## 5. Visualização dos Resultados - Integrante 5
def plotar_resultados(resultados):
    """
    Gera visualizações comparando árvore e bagging
    
    Parâmetros:
    resultados: Dicionário com resultados das métricas
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Boxplot dos erros
    axs[0].boxplot([resultados['erros_arvore'], resultados['erros_bagging']], 
                   labels=['Árvore', 'Bagging'])
    axs[0].set_title('Distribuição dos Erros (MSE)')
    axs[0].set_ylabel('Erro Quadrático Médio')
    
    # Comparação de viés
    axs[1].bar(['Árvore', 'Bagging'], [resultados['vies_arvore'], resultados['vies_bagging']])
    axs[1].set_title('Comparação de Viés')
    axs[1].set_ylabel('Viés Quadrático')
    
    # Comparação de variância
    axs[2].bar(['Árvore', 'Bagging'], [resultados['var_arvore'], resultados['var_bagging']])
    axs[2].set_title('Comparação de Variância')
    axs[2].set_ylabel('Variância')
    
    plt.tight_layout()
    plt.savefig('resultados.png')
    plt.show()

## 6. Documentação e Discussão Teórica - Integrante 6
def gerar_relatorio(resultados):
    """
    Gera um relatório textual com os resultados e análise teórica
    """
    relatorio = f"""
    RELATÓRIO: ANÁLISE DE VIÉS E VARIÂNCIA
    
    1. Resultados Empíricos:
    - Árvore de Decisão:
      * Viés médio: {resultados['vies_arvore']:.4f}
      * Variância: {resultados['var_arvore']:.4f}
      * Erro médio (MSE): {np.mean(resultados['erros_arvore']):.4f}
    
    - Bagging (n={resultados['n_estimators']} árvores):
      * Viés médio: {resultados['vies_bagging']:.4f}
      * Variância: {resultados['var_bagging']:.4f}
      * Erro médio (MSE): {np.mean(resultados['erros_bagging']):.4f}
    
    2. Análise Teórica:
    - Como esperado, o Bagging reduziu a variância em {100*(1 - resultados['var_bagging']/resultados['var_arvore']):.1f}% 
      comparado à árvore única, confirmando que ensembles são eficazes para reduzir overfitting.
    - O viés aumentou ligeiramente ({100*(resultados['vies_bagging']/resultados['vies_arvore'] - 1):.1f}%), 
      demonstrando o tradeoff clássico viés-variância.
    - A redução geral no MSE de {100*(1 - np.mean(resultados['erros_bagging'])/np.mean(resultados['erros_arvore'])):.1f}% 
      mostra que o benefício da redução de variância superou o aumento de viés.
    """
    print(relatorio)
    
    with open('relatorio.txt', 'w') as f:
        f.write(relatorio)

## Fluxo principal de execução
def main():
    # Gerar dados sintéticos (regressão)
    X, y = make_regression(
        n_samples=1000, 
        n_features=20, 
        noise=0.5, 
        random_state=42
    )
    
    # Configurações da avaliação
    n_splits = 30
    n_estimators = 100
    
    # Armazenar resultados
    resultados = {
        'erros_arvore': [],
        'erros_bagging': [],
        'vies_arvore': [],
        'vies_bagging': [],
        'var_arvore': [],
        'var_bagging': [],
        'n_estimators': n_estimators
    }
    
    # Avaliação com múltiplas divisões de dados
    splitter = ShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=42)
    
    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 1. Treinar árvore de decisão (Integrante 1)
        preds_arvore, arvore = implementar_arvore(X_train, y_train, X_test)
        
        # 2. Treinar Bagging (Integrante 2)
        preds_bagging, bagging = implementar_bagging(arvore, X_train, y_train, X_test, n_estimators)
        
        # 3. Calcular métricas (Integrantes 3 e 4)
        resultados['erros_arvore'].append(mean_squared_error(y_test, preds_arvore))
        resultados['erros_bagging'].append(mean_squared_error(y_test, preds_bagging))
        
        resultados['vies_arvore'].append(calcular_vies(y_test, preds_arvore))
        resultados['vies_bagging'].append(calcular_vies(y_test, preds_bagging))
        
        resultados['var_arvore'].append(calcular_variancia(preds_arvore))
        resultados['var_bagging'].append(calcular_variancia(preds_bagging))
    
    # Calcular médias
    for key in ['vies_arvore', 'vies_bagging', 'var_arvore', 'var_bagging']:
        resultados[key] = np.mean(resultados[key])
    
    # 5. Visualização (Integrante 5)
    plotar_resultados(resultados)
    
    # 6. Relatório (Integrante 6)
    gerar_relatorio(resultados)

if __name__ == '__main__':
    main()
