# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 17:54:19 2021

# Equipe:
# *   Juliane
# *   Rubens Lopes

"""

from numpy import linalg,apply_along_axis
from pylab import plot,show,pcolor,colorbar,bone
from minisom import MiniSom
import numpy as np
#Cria um objeto(bunch) com os dados da base e seu atributos
parkinson = np.genfromtxt('D:/OneDrive/Faculdade/S7/parkinson/parkinson_formated.csv', delimiter= ',')

#imprime os dados da base
type(parkinson)

features = parkinson[:,0:-1]

#normalizacao

data = apply_along_axis(lambda x: x/linalg.norm(x),1,features)

for epocas in [100, 200, 300, 1000]:
    
    ### Initialization and training ###
    som = MiniSom(20,20,754,sigma=1.0,learning_rate=0.01)
    som.random_weights_init(data)
    som.train_random(data,epocas)
       
    bone()
    pcolor(som.distance_map().T)
    colorbar()
    
    
    t = parkinson[:,-1].astype(int)
    
    # use different colors and markers for each label
    markers = ['o','s']
    colors = ['r','g']
    for cnt,xx in enumerate(data):
     w = som.winner(xx) # getting the winner
     # palce a marker on the winning position for the sample xx
     plot(w[0]+.5,w[1]+.5,markers[t[cnt]],markerfacecolor='None',
       markeredgecolor=colors[t[cnt]],markersize=12,markeredgewidth=2)
    
    show()
