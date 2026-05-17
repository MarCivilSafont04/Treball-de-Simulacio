from vpython import *
import matplotlib.pyplot as plt
import numpy as np
import random as rd 

# --- PARÀMETRES INICIALS ---
win = 500
Natoms = 200  
L = 1 
gray = color.gray(0.7)
mass = 4E-3/6E23 
Ratom = 0.03 
k = 1.4E-23 
T = 300 
dt = 1E-5

# Configuració de la simulació no infinita
max_iter = 5000  # Número de passos abans d'aturar-se i fer el gràfic
comptador = 0

animation = canvas(width=win, height=win, align='left')
animation.range = L
animation.title = 'A "hard-sphere" gas'

# --- CONSTRUCCIÓ DE LA CAIXA ---
d = L/2+Ratom
r = 0.005
boxbottom = curve(color=gray, radius=r)
boxbottom.append([vector(-d,-d,-d), vector(-d,-d,d), vector(d,-d,d), vector(d,-d,-d), vector(-d,-d,-d)])
boxtop = curve(color=gray, radius=r)
boxtop.append([vector(-d,d,-d), vector(-d,d,d), vector(d,d,d), vector(d,d,-d), vector(-d,d,-d)])
vert1 = curve(color=gray, radius=r); vert1.append([vector(-d,-d,-d), vector(-d,d,-d)])
vert2 = curve(color=gray, radius=r); vert2.append([vector(-d,-d,d), vector(-d,d,d)])
vert3 = curve(color=gray, radius=r); vert3.append([vector(d,-d,d), vector(d,d,d)])
vert4 = curve(color=gray, radius=r); vert4.append([vector(d,-d,-d), vector(d,d,-d)])

# --- INICIALITZACIÓ D'ÀTOMS ---
Atoms = []
p = []
apos = []
pavg = sqrt(2*mass*1.5*k*T) 
    
for i in range(Natoms):
    x = L*random()-L/2
    y = L*random()-L/2
    z = L*random()-L/2
    if i == 0:
        Atoms.append(sphere(pos=vector(x,y,z), radius=Ratom, color=color.cyan, make_trail=True, retain=100, trail_radius=0.3*Ratom))
    else: Atoms.append(sphere(pos=vector(x,y,z), radius=Ratom, color=gray))
    apos.append(vec(x,y,z))
    theta = pi*random()
    phi = 2*pi*random()
    px = pavg*sin(theta)*cos(phi)
    py = pavg*sin(theta)*sin(phi)
    pz = pavg*cos(theta)
    p.append(vector(px,py,pz))

# --- HISTOGRAMA VPYTHON (V_MOD) ---
deltav = 100 
def barx(v): return int(v/deltav)
nhisto_bins = int(4500/deltav)
histo = [0.0] * nhisto_bins
histo[barx(pavg/mass)] = Natoms

accum = [[deltav*(i+.5),0] for i in range(int(3000/deltav))]
vdist = gvbars(color=color.red, delta=deltav )

def interchange(v1, v2):
    barx1 = barx(v1)
    barx2 = barx(v2)
    if barx1 == barx2: return
    if barx1 >= len(histo) or barx2 >= len(histo): return
    histo[barx1] -= 1
    histo[barx2] += 1
    
def checkCollisions():
    hitlist = []
    r2 = (2*Ratom)**2
    for i in range(Natoms):
        ai = apos[i]
        for j in range(i):
            dr = ai - apos[j]
            if mag2(dr) < r2: hitlist.append([i,j])
    return hitlist

# --- VARIABLES Q2 ---
nu = 0.05  
llista_energies = [] 
nhisto = 0 

print(f"Simulació en marxa per {max_iter} iteracions...")

# --- BUCLE PRINCIPAL (MODIFICAT PER NO SER INFINIT) ---
while comptador < max_iter:
    rate(500)
    
    # Termòstat d'Andersen
    sigma = sqrt(k * T / mass)
    for i in range(Natoms):
        if rd.random() < nu:
            v_old = p[i].mag / mass
            p[i] = vector(rd.gauss(0, sigma), rd.gauss(0, sigma), rd.gauss(0, sigma)) * mass
            interchange(v_old, p[i].mag / mass)
            
    # Histograma VPython
    for i in range(len(accum)): accum[i][1] = (nhisto*accum[i][1] + histo[i])/(nhisto+1)
    if nhisto % 10 == 0: vdist.data = accum
    nhisto += 1

    # Posicions
    for i in range(Natoms): Atoms[i].pos = apos[i] = apos[i] + (p[i]/mass)*dt
    
    # Col·lisions
    hitlist = checkCollisions()
    for ij in hitlist:
        i, j = ij[0], ij[1]
        ptot, posi, posj = p[i]+p[j], apos[i], apos[j]
        vi, vj = p[i]/mass, p[j]/mass
        vrel, rrel = vj-vi, posi-posj
        if vrel.mag2 == 0 or rrel.mag > Ratom: continue
        
        alpha = asin(cross(rrel, vrel.hat).mag/(2*Ratom)) 
        d_val = (2*Ratom)*cos(alpha)-dot(rrel, vrel.hat)
        deltat = d_val/vrel.mag 
        
        posi, posj = posi-vi*deltat, posj-vj*deltat
        mtot = 2*mass
        pcmi, pcmj = p[i]-ptot*mass/mtot, p[j]-ptot*mass/mtot
        rrel_norm = norm(rrel)
        pcmi = pcmi-2*pcmi.dot(rrel_norm)*rrel_norm
        pcmj = pcmj-2*pcmj.dot(rrel_norm)*rrel_norm
        p[i], p[j] = pcmi+ptot*mass/mtot, pcmj+ptot*mass/mtot
        apos[i], apos[j] = posi+(p[i]/mass)*deltat, posj+(p[j]/mass)*deltat
        interchange(vi.mag, p[i].mag/mass)
        interchange(vj.mag, p[j].mag/mass)

    # Captura dades Q2
    if comptador % 50 == 0:
        for i in range(Natoms):
            llista_energies.append(p[i].mag2 / (2 * mass))

    # Parets
    for i in range(Natoms):
        if abs(apos[i].x) > L/2: p[i].x = abs(p[i].x) if apos[i].x < 0 else -abs(p[i].x)
        if abs(apos[i].y) > L/2: p[i].y = abs(p[i].y) if apos[i].y < 0 else -abs(p[i].y)
        if abs(apos[i].z) > L/2: p[i].z = abs(p[i].z) if apos[i].z < 0 else -abs(p[i].z)

    comptador += 1

print("Simulació finalitzada. Generant gràfics...")

# --- GENERACIÓ DEL GRÀFIC MATPLOTLIB ---
plt.figure(figsize=(10, 6))
plt.hist(llista_energies, bins=50, density=True, alpha=0.7, color='skyblue', label="Simulació (Andersen)")

# Teoria
E_teorica = np.linspace(0, max(llista_energies), 200)
kT = k * T
P_E = (2/np.sqrt(np.pi)) * (1/kT)**1.5 * np.sqrt(E_teorica) * np.exp(-E_teorica/kT)

plt.plot(E_teorica, P_E, 'r-', lw=2, label="Teòrica (Boltzmann)")
plt.title(f"Distribució d'Energies (T={T}K, iteracions={max_iter})")
plt.xlabel("Energia (J)")
plt.ylabel("Probabilitat")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
