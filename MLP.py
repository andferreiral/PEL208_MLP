# -*- coding: utf-8 -*-
"""
@author: Andrey Ferreira de Almeida
"""

#bibliotecas utilizadas
import numpy as np
from sklearn.datasets import load_iris

#carrega o dataset do iris
iris = load_iris()

#separa os dados de entrada
entrada = iris.data

#separa os dados target
saida = iris.target

#transpõe o vetor target
vetor_saida = saida.copy().reshape(150,1)

#pesos randômicos para a camada escondida e de saída para o backpropagation
pesos_camada_escondida = np.random.rand(4,150)
pesos_camada_saida = np.random.rand(150, 1)

#taxa de aprendizado
taxa_aprendizado = 0.0005

#tamanho do batch
batch_size = 200000

#função sigmoid
def sigmoid(entrada):
    return 1/(1 + np.exp(-entrada))

#derivada da função sigmoid
def derivada_sigmoid(entrada):
    return sigmoid(entrada) * (1 - sigmoid(entrada))

#função de treinamento
def train(entrada, saida, pesos_camada_escondida, pesos_camada_saida, taxa_aprendizado, batch_size):
    
#    variáveis que receberão os pesos a atualizar
    peso_camada_escondida = pesos_camada_escondida
    peso_camada_saida = pesos_camada_saida
    
#   cria a iteração para o treinamento
    for batch in range(batch_size):
        
        print(f'Treinamento: {batch} de {batch_size}')
        
#        entrada da camada oculta
        camada_oculta = np.dot(entrada, peso_camada_escondida)
#        saida da camada oculta
        saida_camada_oculta = sigmoid(camada_oculta)

#        entrada da camada de saída
        entrada_camada_saida = np.dot(saida_camada_oculta, peso_camada_saida)
#        saida da camada de saida
        saida_camada_saida = sigmoid(entrada_camada_saida)
        
#        calcula o erro
#        erro_camada_saida = ((1 / 2) * (np.power((saida_camada_saida - saida), 2)))
        
#        calcula a derivada da entrada da camada de saída por conta do backpropagation
        derivada_entrada = derivada_sigmoid(entrada_camada_saida)
        
#        (camada de saida transposta * 
#        (sigmoid(saida_camada_saida) - saida_target) * derivada_entrada)
        resultado_camada_saida = np.dot(saida_camada_oculta.T, \
                                    (saida_camada_saida - saida) * derivada_entrada)
        
#        calculo a nova matriz com base nos pesos de saída
        volta_camada_oculta = np.dot((saida_camada_saida - saida), peso_camada_saida.T)
        
#        calculo a derivada da sigmoid da camada oculta
        derivada_camada_oculta = derivada_sigmoid(camada_oculta)
        
#        (camada de entrada transposta * 
#        (sigmoid(saida_camada_saida) - saida_target) * "volta_camada_oculta"
        resultado_camada_oculta = np.dot(entrada.T, \
                                         derivada_camada_oculta * volta_camada_oculta)
        
#        atualiza os pesos da camada oculta com base na taxa de aprendizado
        peso_camada_escondida -= taxa_aprendizado * resultado_camada_oculta
        peso_camada_saida -= taxa_aprendizado * resultado_camada_saida
        
    return peso_camada_escondida, peso_camada_saida

#chamada da função de treinamento
pesos_camada_o, pesos_camada_s = train(entrada, vetor_saida, 
                                       pesos_camada_escondida,
                                       pesos_camada_saida,
                                       taxa_aprendizado,
                                       batch_size)
#função de predição
def predict(entrada):
    lista_entrada = np.array(entrada)
    predicao = sigmoid(np.dot(sigmoid(np.dot(lista_entrada, pesos_camada_o)), pesos_camada_s))
    return predicao

#chamada da função de predição
print(predict([5.1, 3.5, 1.4, 0.2]))