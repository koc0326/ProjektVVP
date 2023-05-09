import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
import scipy
from time import time as t
import json


import warnings

# potlaceni warningu
warnings.filterwarnings('ignore')


class Obrazec:
    def __init__(self, typ : str) -> None:
        self.typ = typ

    def nacti_matice(self) -> list:
        """
        Nacteni matic ze zadani 
        Return:
            matice - list matic

        Pouziti:
            nacti_matice()
        """
        if (self.typ == "cihly"):
            rho = np.load("data/cihly/RHO_C.npy")
            lam = np.load("data/cihly/LAM.npy")
            u = np.load("data/cihly/U_initial.npy")
        elif (self.typ == "spirala"):
            rho = np.load("data/spirala/RHO_C.npy")
            lam = np.load("data/spirala/LAM.npy")
            u = np.load("data/spirala/U_initial.npy")
        matice = [rho, lam, u] 
        return matice

    def nacti_parametry(self) -> list:
        """
        Nacteni parametru ze souboru json 
        Return:
            parametry - list parametru

        Pouziti:
            nacti_parametry()
        """
        if (self.typ == "cihly"):
            with open('data/cihly/parametry.json') as f:
                data = json.load(f)
            dx = data['dx']
            dy = data['dy']
            dt = data['dt']
        elif (self.typ == "spirala"):
            with open('data/spirala/parametry.json') as f:
                data = json.load(f)
            dx = data['dx']
            dy = data['dy']
            dt = data['dt']
        parametry = [dt, dx, dy]  
        return parametry      

    def vypocti_u(self,stop) -> list:
        """
        Vypocet zmen teplot pomoci vzorce U 
        Parametry:
            stop - konecny cas
            
        Return:
            U - list matic

        Pouziti:
            vypocti_u(stop)
        """
        [rho,lam,u0] = self.nacti_matice()
        [dt,dx,dy] = self.nacti_parametry()
        [a,b] = u0.shape
        U=[]
        U_new = u0.copy()
        for time in range(0, stop+1):
            u = U_new.copy()
            U.append(u) 
            for y in range(0, b):
                for x in range(0, a):
                    r_x_c = (2/rho[x,y])* dt
                    soucet = 0
                    if(x > 0):
                        soucet += (u[x-1,y] - u[x, y])/(((1/(lam[x,y])) + (1/(lam[x - 1, y])))*(dx**2))
                    if(y > 0):
                        soucet += (u[x, y - 1] - u[x, y])/(((1/(lam[x, y])) + (1/(lam[x, y - 1])))*(dy**2))
                    if(x < a-1):
                        soucet += (u[x + 1, y] - u[x, y])/(((1/(lam[x, y])) + (1/(lam[x + 1, y])))*(dx**2))
                    if(y < b-1):
                        soucet += (u[x, y + 1] - u[x, y])/(((1/(lam[x, y])) + (1/(lam[x, y + 1])))*(dy**2))
                    U_new[x, y] = u[x, y] + r_x_c*soucet
        return U
    

    def vypocti_u_sparse(self, stop) -> list:
        """
        Vypocet zmen teplot pomoci vzorce U 
        Parametry:
            stop - konecny cas

        Return:
            U - list matic

        Pouziti:
            vypocti_u_sparse(stop)
        """
        [rho,lam,u0] = self.nacti_matice()
        [dt,dx,dy] = self.nacti_parametry()
        [a,b] = u0.shape
        U=[]
        U_new = csc_matrix(u0.copy())
        soucet = csc_matrix(np.zeros((a,b)))
        r_x_c = csc_matrix(np.zeros((a,b)))
        for i in range(0,a):
            for j in range(0,b):
                r_x_c[i,j] =  (2/rho[i,j]) * dt 
        for time in range(0, stop+1):
            u = U_new.copy() 
            U.append(U_new) 
            soucet[1:a, :] += (u[:a-1, :] - u[1:a, :])/(((1/(lam[1:a, :])) + (1/(lam[:a-1, :])))*(dx**2))
            soucet[:, 1:b] += (u[:, :b-1] - u[:, 1:b])/(((1/(lam[:, 1:b])) + (1/(lam[:, :b-1])))*(dy**2))
            soucet[:a-1, :] += (u[1:a, :] - u[:a-1, :])/(((1/(lam[:a-1, :])) + (1/(lam[1:a, :])))*(dx**2))
            soucet[:, :b-1] += (u[:, 1:b] - u[:, :b-1])/(((1/(lam[:, :b-1])) + (1/(lam[:, 1:b])))*(dy**2))
            U_new = u + r_x_c.multiply(soucet)
        return U

    def vykresli_cas(self, stop : int) -> None:
        """
        Vykresli teplotu v danem case stop pomoci vypocti_u
        Parametry:
            stop - konecny cas
            
        Return:
            None

        Pouziti:
            vykresli_cas(stop)
        """
        U = self.vypocti_u(stop)
        color =sns.color_palette("coolwarm", as_cmap=True)
        fig = plt.figure()  
        plt.title(f"Teplota v čase t = {int(stop)}")
        ax = sns.heatmap(U[stop],cmap=color)
        plt.show()

    def vykresli_cas_sparse(self, stop : int) -> None:
        """
        Vykresli teplotu v danem case stop pomoci vypocti_u_sparse
        Parametry:
            stop - konecny cas
            
        Return:
            None

        Pouziti:
            vykresli_cas(stop)
        """
        U = self.vypocti_u_sparse(stop)
        color =sns.color_palette("coolwarm", as_cmap=True)
        fig = plt.figure()  
        plt.title(f"Teplota v čase t = {int(stop)}")
        ax = sns.heatmap(U[stop].todense(),cmap=color)
        plt.show()
    

def run_u(S: Obrazec, stop : int) -> None:
    """
    Vypocet matic U pomoci vypocti_u a ulozeni animace
    Parametry:
        S - trida Obrazec
        stop - konecny cas

    Return:
        None

    Pouziti:
        run_u(S,stop)
    """
    color =sns.color_palette("coolwarm", as_cmap=True)

    fig = plt.figure()
    U = S.vypocti_u(stop)
    
    def init():
        plt.clf()
        plt.title(f"Teplota v t = 0")
        ax = sns.heatmap(U[0], cmap=color)

    def animate(i):
        plt.clf()
        plt.title(f"Teplota v t = {int(i)}")
        ax = sns.heatmap(U[i], cmap = color)

    anim = animation.FuncAnimation(fig, animate, init_func=init, interval=100, frames = stop+1, cache_frame_data=False, repeat = False)
    pillowwriter = animation.PillowWriter(fps=10)
    if (S.typ == "spirala"):
            name = "spirala.gif"
    else:
            name = "cihla.gif"
    anim.save(name, writer=pillowwriter)
    plt.show()

def run_u_sparse(S : Obrazec, stop : int) -> None:
    """
    Vypocet matic U pomoci vypocti_u_sparse a ulozeni animace
    Parametry:
        S - trida Object
        stop - konecny cas

    Return:
        None

    Pouziti:
        run_u_sparse(S,stop)
    """
    color =sns.color_palette("coolwarm", as_cmap=True)

    fig = plt.figure()
    U = S.vypocti_u_sparse(stop)
    
    def init():
        plt.clf()
        plt.title(f"Teplota v t = 0")
        ax = sns.heatmap(U[0].todense(),cmap=color)

    def animate(i):
        plt.clf()
        plt.title(f"Teplota v t = {int(i)}")
        ax = sns.heatmap(U[i].todense(), cmap = color)

    anim = animation.FuncAnimation(fig, animate, init_func=init, interval=100, frames = stop+1, cache_frame_data=False, repeat = False)
    pillowwriter = animation.PillowWriter(fps=10)
    if (S.typ == "spirala"):
            name = "spirala_sparse.gif"
    else:
            name = "cihla_sparse.gif"
    anim.save(name, writer=pillowwriter)
    plt.show()

def porovnej(S : Obrazec, stop: int) -> None:
    """
    Porovnani metod vypocti_u  a vypocti_u_sparse
    Parametry:
        S - trida Object
        stop - konecny cas
                
    Return:
        None

    Pouziti:
        porovnej(S,stop)
    """
    begin = t()
    S.vypocti_u(stop)
    end = t()
    begin_S = t()
    S.vypocti_u_sparse(stop)
    end_S = t()
    print("Vypocet U:", end - begin)
    print("Vypocet sparse U:", end_S - begin_S)


