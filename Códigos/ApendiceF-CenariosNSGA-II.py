#! .\ven\scripts\python.exe
# ________________________________________________________________________________________________
from pymoo.indicators.hv import Hypervolume
#from pymoo.util.running_metric import RunningMetricAnimation
from pymoo.mcdm.pseudo_weights import PseudoWeights
from matplotlib import tight_layout
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import Workbook

# ________________________________________________________________________________________________
# Novos Plots com lib interna pymoo
from pymoo.visualization.scatter import Scatter
from pymoo.visualization.heatmap import Heatmap
from pymoo.visualization.petal import Petal
from pymoo.visualization.radar import Radar
from pymoo.util.misc import stack

# ________________________________________________________________________________________________
# NSGA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.crossover.expx import ExponentialCrossover
from pymoo.operators.crossover.ux import UniformCrossover
import time

# ________________________________________________________________________________________________
# Parametros
#Cenario 1 - 2040 (Exemplo)
C_EXT = [19.2, 11.4, 72, 41.4, 42.3]
C_NOVA = [16.8, 8.36, 72, 41.4, 42.3]
C_INV = [3640, 396, 1000, 1200, 1500]
FE = [10, 44, 551.6, 374.62, 86.21]
E_EXT = [144353.71, 288707.42, 6475473.10, 288707.42, 20443.77]
POT_NOVA = [44894, 34933.60, 64754.73, 2887.07, 204.44]
L = 2750143200
H = [4380, 4380, 8760, 8760, 8760]
I = 5

# ________________________________________________________________________________________________
# Definicao de variaveis extras
# def x(x, j):
#    return 1 * x[j]


def y(x, i):
    inicio = I  # para pular o espaço reservado para a variável 'x'
    return 1 * x[inicio + i]

# ________________________________________________________________________________________________
# Definicao do Problema


class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=self.totalVariaveis(),
                         n_obj=4,  # Numero de saidas de F
                         n_constr=self.totalRestricoes(),  # Numero de saidas de G
                         xl=np.array(self.limitesInferiores()),
                         xu=np.array(self.limitesSuperiores()))

    def totalVariaveis(self):
        return I + I  # I variaveis i e IxI variaveis i,j

    def totalRestricoes(self):
        return 11

    def limitesInferiores(self):
        lista = []
        n = self.totalVariaveis()
        for i in range(0, n):
            lista.append(0)

        return lista

    def limitesSuperiores(self):
        lista = []
        n = self.totalVariaveis()
        for i in range(0, n):
            lista.append(9999999999)

        return lista

    def gerarFuncaoObjetivo(self, x):

        f1 = 0
        for i in range(0, I):
            f1 += (x[i]*C_EXT[i]) + (H[i]*y(x, i)*C_NOVA[i])

        f2 = 0

        for i in range(0, I):
            f2 += (H[i]*y(x, i)*C_INV[i])

        f3 = 0

        for i in range(0, I):
            f3 += (FE[i]*(x[i]+(H[i]*y(x, i))))

        f4 = 0
        for i in range(0, I):
            f4 += (-x[i]) - (H[i]*y(x, i))

        return [f1, f2, f3, f4]

    def gerarRestricoes(self, x):
        listarest = []


# Restricao 1
        for i in range(0, I):
            listarest.append(x[i] - E_EXT[i])

# Resticao 2
        for i in range(0, I):
            listarest.append(y(x, i) - POT_NOVA[i])

# Restricao 3
        soma = 0
        for i in range(0, I):
            soma += (FE[i]*(x[i]+(H[i]*y(x, i))))

        listarest.append(soma - L)

        return listarest

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.gerarFuncaoObjetivo(x)
        out["G"] = self.gerarRestricoes(x)


problem = MyProblem()


# NSGA II set up
algorithm = NSGA2(
    pop_size=435,
    n_offsprings=435,
    sampling=LHS(),
    crossover=SBX(prob=1.0, eta=15),
    mutation=PM(eta=1),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 500)

# Caculation and outputs results of the NSGA II
res = minimize(problem,
               algorithm,
               termination,
               seed=2000,
               save_history=True,
               verbose=False)

X = res.X
F = res.F
print("Best solution found: %s" % res.X)
print("Function value: %s" % res.F)

def countList(lst):
    return len(lst)

print("Qtd results: %s" % countList(res.X))

# ________________________________________________________________________________________________
# Identification of min and max value
approx_ideal = fl = F.min(axis=0)
approx_nadir = fu = F.max(axis=0)
print(f"Scale f1: [{fl[0]}, {fu[0]}]")
print(f"Scale f2: [{fl[1]}, {fu[1]}]")
print(f"Scale f3: [{fl[2]}, {fu[2]}]")
print(f"Scale f3: [{fl[3]}, {fu[3]}]")

'''plot = Scatter(tight_layout=True, figsize=(20, 15))
plot.add(F, s=50)
plot.add(approx_ideal, facecolors='none', edgecolors='red',
         marker="*", s=80, label="Ideal Point (Approx)")
plot.add(approx_nadir, facecolors='none', edgecolors='black', marker="p", s=80, label="Nadir Point (Approx)")
plot.show()'''

# ________________________________________________________________________________________________
# Normalization
nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

fl = nF.min(axis=0)
fu = nF.max(axis=0)
print(f"Scale f1 - Normalized: [{fl[0]}, {fu[0]}]")
print(f"Scale f2 - Normalized: [{fl[1]}, {fu[1]}]")
print(f"Scale f3 - Normalized: [{fl[2]}, {fu[2]}]")
print(f"Scale f3 - Normalized: [{fl[3]}, {fu[3]}]")

'''plot = Scatter(tight_layout=True, figsize=(20, 15))
plot.add(nF, s=50)
plot.add(fl, facecolors='none', edgecolors='red', marker="*", s=80, label="Ideal Point (Approx)")
plot.add(fu, facecolors='none', edgecolors='black', marker="p", s=80, label="Nadir Point (Approx)")
plot.show()'''

# ________________________________________________________________________________________________
# Select a result function

weights = np.array([0.7, 0.5, 0.6, 0.8])
i = PseudoWeights(weights).do(nF)

print("Best regarding Pseudo Weights: Point \ni = %s\nF = %s" % (i, F[i]))
print("Best regarding Pseudo Weights: Point \ni = %s\nX = %s" % (i, X[i]))

'''plot = Scatter(tight_layout=True, figsize=(20, 15))
plot.add(F, s=50)
plot.add(F[i], marker="x", color="red", s=100)
plot.show()'''

# ________________________________________________________________________________________________
# Exportar Excel

if str(res.X) == "None":
    #pass
    void_solution += 1
else:
    #Build up to export to Excel file each result
    listX1 = X[:, 0]
    listX2 = X[:, 1]
    listX3 = X[:, 2]
    listX4 = X[:, 3]
    listX5 = X[:, 4]
    listY1 = X[:, 5]
    listY2 = X[:, 6]
    listY3 = X[:, 7]
    listY4 = X[:, 8]
    listY5 = X[:, 9]

    listF1 = F[:, 0]
    listF2 = F[:, 1]
    listF3 = F[:, 2]
    listF4 = F[:, 3]

    colX1 = "solution found X1"
    colX2 = "solution found X2"
    colX3 = "solution found X3"
    colX4 = "solution found X4"
    colX5 = "solution found X5"
    colX6 = "solution found Y1"
    colX7 = "solution found Y2"
    colX8 = "solution found Y3"
    colX9 = "solution found Y4"
    colX10 = "solution found Y5"

    colF11 = "Function Value F1"
    colF12 = "Function Value F2"
    colF13 = "Function Value F3"
    colF14 = "Function Value F4"

    data = pd.DataFrame(
        {colX1:listX1,colX2:listX2,colX3:listX3,colX4:listX4,colX5:listX5,colX6:listY1,colX7:listY2,colX8:listY3,colX9:listY4,colX10:listY5,
        colF11:listF1,colF12:listF2,colF13:listF3,colF14:listF4}
        )
    
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    filepath = '/content/drive/MyDrive/Cenarios/Cenario_2040.xlsx'
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer: 

        # Write each dataframe to a different worksheet.
        data.to_excel(writer, sheet_name='Dados', index=False)
