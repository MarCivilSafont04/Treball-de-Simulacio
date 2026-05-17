from vpython import *
import matplotlib.pyplot as plt
import numpy as np
import random as rd

# --- 1. PARÀMETRES DE LA SIMULACIÓ ---
win = 500
Natoms = 200  
L = 1 
gray = color.gray(0.7)
mass = 4E-3/6E23 
Ratom = 0.03 
k = 1.4E-23 
T = 300 
dt = 1E-5 

# Q3: Gravetat
g_acc = 9.8      
sigma_at = sqrt(k * T / mass)
max_iteracions = 5000

llista_z = []
llista_vz = []

# --- 2. ESCENA ---
animation = canvas(width=win, height=win, align='left')
animation.range = L
animation.title = f'Gas d\'esferes dures - Q3 (g={g_acc})'

d = L/2+Ratom
r = 0.005
box = [curve(color=gray, radius=r) for _ in range(6)]
box[0].append([vector(-d,-d,-d), vector(-d,-d,d), vector(d,-d,d), vector(d,-d,-d), vector(-d,-d,-d)])
box[1].append([vector(-d,d,-d), vector(-d,d,d), vector(d,d,d), vector(d,d,-d), vector(-d,d,-d)])
box[2].append([vector(-d,-d,-d), vector(-d,d,-d)])
box[3].append([vector(-d,-d,d), vector(-d,d,d)])
box[4].append([vector(d,-d,d), vector(d,d,d)])
box[5].append([vector(d,-d,-d), vector(d,d,-d)])

Atoms = []; p = []; apos = []
pavg = sqrt(2*mass*1.5*k*T)

for i in range(Natoms):
    x, y, z = L*rd.random()-L/2, L*rd.random()-L/2, L*rd.random()-L/2
    Atoms.append(sphere(pos=vector(x,y,z), radius=Ratom, color=color.cyan if i==0 else gray, make_trail=(i==0)))
    apos.append(vec(x,y,z))
    theta, phi = pi*rd.random(), 2*pi*rd.random()
    p.append(vector(pavg*sin(theta)*cos(phi), pavg*sin(theta)*sin(phi), pavg*cos(theta)))

def checkCollisions():
    hitlist = []
    r2 = (2*Ratom)**2
    for i in range(Natoms):
        ai = apos[i]
        for j in range(i):
            dr = ai - apos[j]
            if mag2(dr) < r2: hitlist.append([i,j])
    return hitlist

nhisto = 0
print(f"Iniciant simulació: {max_iteracions} iteracions.")

# --- 3. BUCLE PRINCIPAL BLINDAT ---
while nhisto < max_iteracions:
    rate(500)
    

    # B. Gravetat i Moviment
    for i in range(Natoms):
        p[i].z -= mass * g_acc * dt 
        apos[i] = apos[i] + (p[i]/mass)*dt
        Atoms[i].pos = apos[i]
        
        if nhisto % 100 == 0:
            llista_z.append(apos[i].z)
            llista_vz.append(p[i].z / mass)

    # C. Col·lisions entre Àtoms (Versió Anti-Errors)
    hitlist = checkCollisions()
    for ij in hitlist:
        i, j = ij[0], ij[1]
        vrel = (p[j]-p[i])/mass
        rrel = apos[i]-apos[j]
        
        # Seguretat 1: Si ja s'allunyen, no recalcular (evita solapaments infinits)
        if vrel.dot(rrel) >= 0: continue 
        
        # Seguretat 2: Clamp per a l'asin (evita ValueError)
        dy = cross(rrel, vrel.hat).mag
        val_asin = dy/(2*Ratom)
        if val_asin > 1: val_asin = 1
        
        alpha = asin(val_asin)
        d_col = (2*Ratom)*cos(alpha) - dot(rrel, vrel.hat)
        deltat = d_col/vrel.mag
        
        # Moviment cap enrere, xoc i endavant
        posi = apos[i] - (p[i]/mass)*deltat
        posj = apos[j] - (p[j]/mass)*deltat
        mtot = 2*mass; ptot = p[i] + p[j]
        r_unit = norm(posi-posj)
        pcmi = p[i] - ptot*0.5
        pcmi = pcmi - 2*pcmi.dot(r_unit)*r_unit
        p[i] = pcmi + ptot*0.5
        p[j] = ptot - p[i]
        apos[i] = posi + (p[i]/mass)*deltat
        apos[j] = posj + (p[j]/mass)*deltat

    # D. Col·lisions amb Parets (abs() per seguretat extra)
    for i in range(Natoms):
        if abs(apos[i].x) > L/2: p[i].x = -abs(p[i].x) if apos[i].x > 0 else abs(p[i].x)
        if abs(apos[i].y) > L/2: p[i].y = -abs(p[i].y) if apos[i].y > 0 else abs(p[i].y)
        if abs(apos[i].z) > L/2: p[i].z = -abs(p[i].z) if apos[i].z > 0 else abs(p[i].z)
    
    nhisto += 1

# --- 4. GRÀFIQUES ---
print("Generant gràfiques finals...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(llista_z, bins=30, density=True, color='green', alpha=0.7, orientation='horizontal')
z_range = np.linspace(-L/2, L/2, 100)
teoria_z = np.exp(-mass * g_acc * z_range / (k * T))
teoria_z /= np.trapz(teoria_z, z_range)
plt.plot(teoria_z, z_range, 'r-', lw=2, label='Boltzmann')
plt.title(f"Llei Baromètrica (g={g_acc})"); plt.legend()

plt.subplot(1, 2, 2)
plt.hist(llista_vz, bins=30, density=True, color='blue', alpha=0.7)
v_range = np.linspace(min(llista_vz), max(llista_vz), 100)
teoria_v = (1/(sigma_at*np.sqrt(2*np.pi))) * np.exp(-v_range**2 / (2*sigma_at**2))
plt.plot(v_range, teoria_v, 'r-', lw=2, label='Maxwell-Boltzmann')
plt.title("Velocitats $v_z$"); plt.legend()
plt.show()