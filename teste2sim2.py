import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

print('Carregando Arquivo de teste')
arquivo = np.load('teste2.npy')
x = arquivo[0]
y = np.ravel(arquivo[1])


#Usando o teste de arquitetura camadas
for j in range(0, 10):

    regr = MLPRegressor(hidden_layer_sizes=(300, 1000), 
                        max_iter=1000,
                        activation='logistic', #{'identity', 'logistic', 'tanh', 'relu'} - tipo do grafico
                        solver='adam',
                        learning_rate = 'adaptive',
                        n_iter_no_change=50) 
    print('Treinando RNA')
    regr = regr.fit(x,y)



    print('Preditor')
    y_est = regr.predict(x)


    plt.figure(figsize=[14,7])

    #plot curso original
    plt.subplot(1,3,1)
    plt.plot(x,y)

    #plot aprendizagem
    plt.subplot(1,3,2)
    plt.plot(regr.loss_curve_)

    #plot regressor
    plt.subplot(1,3,3)
    plt.plot(x,y,linewidth=1,color='yellow')
    plt.plot(x,y_est,linewidth=2)

    erro = []

    for i in range(0,len(y_est)):
        erro.append(y_est[i]-y[i])

    arquivo = open("Teste2-Sim2/log.txt",'a')
    arquivo.write("Média do erro do teste "+str(j+1)+": "+str(np.average(erro))+"\n")

    arquivo.write("Desvio padrão do erro do teste "+str(j+1)+": "+str(np.std(erro))+"\n\n")

    plt.savefig("Teste2-Sim2/teste"+str(j+1)+".png", format='png')
    arquivo.close()