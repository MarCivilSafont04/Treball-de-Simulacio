# -*- coding: utf-8 -*-
"""
Created on Mon May 11 23:37:44 2026

@author: Usuari
"""

import numpy as np
import matplotlib.pyplot as plt

#PARÀMETRES
T = 10.0              
epsilon = 1.0          
k_B = 1.0              
beta = 1 / (k_B * T)  
passos = 500000        # Nombre total d'iteracions per cada N
equilibri = 200000     # Pas a partir del qual comencem a prendre dades 

# Definició dels nivells d'energia.
nivells = np.array([0, epsilon, 10 * epsilon])

# Llista del nombre de partícules que volem estudiar
llista_N = [50,100,200,300, 400, 500,600,700,800,900, 1000, 2000, 4000, 5000, 8000, 10000]
fluctuacions_relatives = []


for N in llista_N:
    # Inicialització de l'estat: cada partícula té 1/3 de probabilitat per nivell
    estats = np.random.choice([0, 1, 2], size=N)
    
    # Llista per guardar l'energia total en cada iteració (un cop a l'equilibri)
    energies_totals_equilibri = []
    
    # METROPOLIS
    for a in range(passos):
        # Triem una partícula a l'atzar
        i = np.random.randint(N)
        estat_actual = estats[i]
        
        # Triem un nou estat diferent de l'actual
        nou_estat = np.random.choice([n for n in [0, 1, 2] if n != estat_actual])
        
        # Calculem el canvi d'energia (delta E)
        delta_E = nivells[nou_estat] - nivells[estat_actual]
        
        # Regla de Metropolis
        q = np.exp(-beta * delta_E)
        Pacc = min(1, q)
        
        if np.random.rand() <= Pacc:
            estats[i] = nou_estat
            
        #Emmagatzematge de dades (Només en l'equilibri)
        # Prenem dades cada 100 passos per evitar correlacions.
        if a >= equilibri and a % 100 == 0:
            energia_total = np.sum(nivells[estats])
            energies_totals_equilibri.append(energia_total)
    

   
    #FLUCTUACIÓ RELATIVA (R)
    sigma_E = np.std(energies_totals_equilibri)
    E_mitjana = np.mean(energies_totals_equilibri)
    
    R = sigma_E / E_mitjana
    fluctuacions_relatives.append(R)
    
    print(f"N: {N:4} | R: {R:.5f} | R * sqrt(N): {R * np.sqrt(N):.4f}")

#Gràfic simulació
plt.figure(figsize=(8, 5))
plt.plot(llista_N, fluctuacions_relatives, 'ro-', label="Dades simulació")

# Gràfic corba teòrica 1/sqrt(N) 
teorica = [fluctuacions_relatives[0] * np.sqrt(llista_N[0]/n) for n in llista_N]
plt.plot(llista_N, teorica, 'k--', label=r"Tendència $1/\sqrt{N}$")

plt.xlabel("Nombre de partícules (N)")
plt.ylabel(r"Fluctuació relativa ($R = \sigma_E / \langle E \rangle$)")

plt.legend()
plt.grid(True, alpha=0.3)
plt.show()