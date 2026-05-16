# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 18:43:37 2026

@author: Usuari
"""

import numpy as np

#PARÀMETRES
N = 1000               # Nombre total de molècules del sistema.
epsilon = 1.0          # Unitat d'energia (ε).
T = 2.0                # Temperatura del sistema.
k_B = 1.0              # Constant de Boltzmann (per simplificar el càlcul).
beta = 1 / (k_B * T)   # Factor beta (β = 1/kBT).
passos = 1000000       # Nombre d'iteracions per garantir que el sistema arriba a l'equilibri.

# Definició dels nivells d'energia discretitzats segons l'enunciat:
# Nivell 1 = 0, Nivell 2 = ε, Nivell 3 = 10ε.
nivells = np.array([0, epsilon, 10 * epsilon])


# Seguint les instruccions de la Q6, cada partícula ha de tenir inicialment una 
# probabilitat d'1/3 d'estar en qualsevol dels tres nivells. 
# np.random.choice tria un índex (0, 1 o 2) de forma aleatòria i uniforme.

estats = np.random.choice([0, 1, 2], size=N)
#és un vector amb N números. Cada número és un índex (0, 1 o 2) que ens diu en quin nivell d'energia està cada partícula.

# Funció per calcular l'Hamiltonià del sistema.
# Multipliquem l'ocupació de cada nivell pel seu valor d'energia corresponent.
def calcular_energia_total(configuracio):
    return np.sum(nivells[configuracio])

#METROPOLIS

for a in range(passos):
    # 1. Triem una partícula a l'atzar
    i = np.random.randint(N)
    
    # 2. Mirem en quin nivell està i quina energia té ara
    estat_actual = estats[i]
    E_actual = nivells[estat_actual]
    
    # 3. Triem un nou nivell i mirem quina energia tindria
    # Fem que triï un nivell diferent de l'actual
    nou_estat = np.random.choice([n for n in [0, 1, 2] if n != estat_actual])
    E_nova = nivells[nou_estat]
    
    # 4. El canvi d'energia és la diferència entre el nou i el vell
   
    delta_E = E_nova - E_actual
    
    # Implementació de la Regla de Metropolis:
    # 1. Si l'energia disminueix (ΔE <= 0), el canvi s'accepta sempre (P = 1).
    # 2. Si l'energia augmenta (ΔE > 0), s'accepta amb probabilitat exp(-β*ΔE).
    q = np.exp(-beta * delta_E)
    Pacc = min(1, q)
    
    # Generem un número aleatori (u) entre 0 i 1 per decidir si acceptem el canvi.
    u = np.random.rand()
    if u <= Pacc:
        # Si u és menor o igual a la probabilitat d'acceptació, actualitzem el sistema.
       estats[i] = nou_estat

# En finalitzar el bucle, el sistema ha evolucionat des de la configuració 
# inicial (1/3 per nivell) cap a la distribució de Boltzmann a temperatura T.