#! .\ven\scripts\python.exe
# _________________________________________________________________
#Importacao de bibliotecas para a aplicacao
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import Workbook

# _________________________________________________________________
# Parametros
C_EXT = [5.06, 45.31, 58.40, 10.00, 4.00]
C_NOVA = [4.06, 44.31, 57.40, 9.00, 3.00]
C_INV = [1.50, 1.50, 1.50, 1.50, 1.50]
H = [4380, 4380, 4380, 4380, 4380]
FE = [600, 610, 850, 0, 200]
I = 5
E_EXT = [4000, 6000, 4000, 3000, 4500]
POT_NOVA = [1430, 9000, 1750, 4100, 7714]
L = 3667800


# Definicao de variaveis extras

def y(x, i):
    inicio = I  # Parametro para deslocar para o espaço reservado para a variável 'x'
    return 1 * x[inicio + i]

# _________________________________________________________________
# Definicao do Problema

class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=self.totalVariaveis(),
                         n_obj= 4,  # Numero de saidas de F
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


# Resticao 1
        for i in range(0, I):
            listarest.append(x[i] - E_EXT[i])

# Restricao 2
        for i in range(0, I):
            listarest.append(y(x, i) - POT_NOVA[i])


# Restricao 3
        soma = 0
        for i in range(0, I):
            soma += (FE[i]*(x[i]+(H[i]*y(x, i))))

        listarest.append(soma - L)

        return listarest

# Saidas da classe MyProblem

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.gerarFuncaoObjetivo(x)
        out["G"] = self.gerarRestricoes(x)


problem = MyProblem()

# Configuracoes do algoritmo NSGA-II

algorithm = NSGA2(
    pop_size=435,
    n_offsprings=435,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.5, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 182)

# Aplicacao do problema utilizando o NSGA-II em um problema de minimizacao

res = minimize(problem,
               algorithm,
               termination,
               seed=2000,
               save_history=True,
               verbose=False)

X = res.X
F = res.F


# Saida no terminal dos resultados

print("Best solution found: %s" % res.X)
print("Function value: %s" % res.F)

# Exporta resultados para arquivo Excel em pasta especifica

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
data.to_excel('D:\MESC\Projeto_dissertacao\experimento1.xlsx', sheet_name='sheet1', index=False)
