# -*- coding: utf-8 -*-
"""
Created on Tue May 12 19:52:27 2026

@author: Usuari
"""

import numpy as np
import matplotlib.pyplot as plt

# PARÀMETRES
N = 1000
epsilon = 1.0
k_B = 1.0
nivells = np.array([0, epsilon, 10 * epsilon])
passos = 1000000

# GRÀFIC PER A T=0.5 (i T=10) 
T_prova = 10
beta_p = 1 / (k_B * T_prova)
estats_p = np.random.choice([0, 1, 2], size=N)
energies = []

for a in range(passos):
    i = np.random.randint(N)
    estat_actual = estats_p[i]
    
    # Selecció del nou estat (diferent de l'actual)
    nou_estat = np.random.choice([n for n in [0, 1, 2] if n != estat_actual])
    
    delta_E = nivells[nou_estat] - nivells[estat_actual]
    
    # Regla de Metropolis 
    q = np.exp(-beta_p * delta_E)
    Pacc = min(1, q)
    
    if np.random.rand() <= Pacc:
        estats_p[i] = nou_estat
    
    # Guardem l'energia cada 1000 passos pel gràfic per optimitzar el programa
    # i fer que el gràfic sigui més entenedor 
    # sense perdre la informació de l'arribada a l'equilibri.
    if a % 1000 == 0:
        energies.append(np.sum(nivells[estats_p]) / N)

plt.figure(figsize=(8, 5))
plt.plot(energies, color='blue', label=f"T = {T_prova}")
plt.xlabel(r"Passos ($\times 10^3$)")
plt.ylabel(r"Energia mitjana ($\langle e \rangle$)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

#CÀLCUL D'OCUPACIONS PER A DIFERENTS TEMPERATURES
temperatures = [0.25, 0.5, 1.0, 1.45, 5.0, 10.0, 100.0, 200.0, 500.0]

for T in temperatures:
    beta = 1 / (k_B * T)
    estats = np.random.choice([0, 1, 2], size=N)
    
    # Comptadors per a la mitjana temporal
    n0_sum, n1_sum, n2_sum = 0, 0, 0
    comptador_equilibri = 0

    for a in range(passos):
        i = np.random.randint(N)
        estat_actual = estats[i]
        
        nou_estat = np.random.choice([n for n in [0, 1, 2] if n != estat_actual])
        delta_E = nivells[nou_estat] - nivells[estat_actual]
        
        # Regla de Metropolis 
        q = np.exp(-beta * delta_E)
        Pacc = min(1, q)
        
        if np.random.rand() <= Pacc:
            estats[i] = nou_estat

        # Acumulem ocupacions a partir del pas d'equilibri (200.000)
        if a >= 200000:
           
            n0_sum += np.sum(estats == 0)
            n1_sum += np.sum(estats == 1)
            n2_sum += np.sum(estats == 2)
            comptador_equilibri += 1

    # Mitjanes temporals de les ocupacions
    m0 = n0_sum / comptador_equilibri
    m1 = n1_sum / comptador_equilibri
    m2 = n2_sum / comptador_equilibri
    
    # Energia mitjana per partícula
    e_mitjana = (m1 * epsilon + m2 * 10.0 * epsilon) / N
    
    print(f"T: {T:6.2f} | n0: {m0:7.2f}, n1: {m1:7.2f}, n2: {m2:7.2f} | <e>: {e_mitjana:7.4f}")