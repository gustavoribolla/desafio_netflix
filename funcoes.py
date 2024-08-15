import numpy as np
import random
from scipy.linalg import svd, diagsvd

def sistema(A, A_nan):
    # Inicializa uma variável para verificar se a avaliação é válida
    avaliacao_valida = False

    # Loop para gerar um número e valor aleatórios para 'estragar' a matriz A
    while not avaliacao_valida:
        # Gera índices aleatórios para a posição na matriz A
        i = random.randint(0, A.shape[0]-1)
        j = random.randint(0, A.shape[1]-1)
        
        # Verifica se o valor correspondente na matriz A_nan não é NaN (não é um número)
        if not np.isnan(A_nan[i][j]):
            avaliacao_valida = True
        
        # Gera um número aleatório e calcula um valor dividindo-o por 2, para que o valor varie de 0.5 em 0.5 no intervalo de 0 a 5 
        numero_aleatorio = random.randint(0, 10)
        valor = numero_aleatorio / 2
    
    # Faz uma cópia da matriz A para B
    B = A.copy()

    # Define o valor gerado aleatoriamente na posição aleatória de B
    B[i][j] = valor
    
    # Retorna a matriz B modificada, o valor original em A_nan na posição modificada, e os índices i e j
    return B, A_nan[i][j], i, j

def remocao(s, K):
    # Cria uma cópia da lista s, atribuindo-a a s_
    s_ = s 
    
    # Multiplica os últimos K elementos de s_ por 0
    s_[-K:] *= 0
    
    # Retorna a lista modificada s_
    return s_

def previsao(matriz_com_ruido, k):
    # Executa a Decomposição em Valores Singulares (SVD) na matriz com ruído
    u_, s, vt_ = svd(matriz_com_ruido)
    
    # Remove os k maiores valores singulares da matriz singular s
    s_ = remocao(s, k)
    
    # Recria a matriz singular sigma com os valores singulares modificados
    sigma = diagsvd(s_, u_.shape[1], vt_.shape[0])
    
    # Reconstrói a matriz com ruído diminuido, multiplicando as matrizes U, Sigma e vt_
    matriz_com_ruido_diminuido = u_ @ sigma @ vt_
    
    # Arredonda os valores da matriz resultante para uma casa decimal
    matriz_com_ruido_diminuido = np.around(matriz_com_ruido_diminuido, 1)
    
    # Retorna a matriz com ruído diminuído
    return matriz_com_ruido_diminuido