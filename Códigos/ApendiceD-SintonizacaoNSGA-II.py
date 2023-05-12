#! .\ven\scripts\python.exe
# _________________________________________________________________
#Importacao de bibliotecas para a aplicacao
from statistics import mode
import numpy as np
import openpyxl
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import Workbook
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from pymoo.visualization.scatter import Scatter
import time
from math import ceil

# _________________________________________________________________
#NSGA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.crossover.expx import ExponentialCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.indicators.hv import Hypervolume

# _________________________________________________________________
# Parametros
C_EXT = [5.06, 45.31, 58.40, 10.00, 4.00]
C_NOVA = [4.06, 44.31, 57.40, 9.00, 3.00]
C_INV = [550, 1550, 550, 550, 550]
H = [4380, 4380, 4380, 4380, 4380]
FE = [600, 610, 850, 100, 200]
I = 5
E_EXT = [4000, 6000, 4000, 3000, 4500]
POT_NOVA = [1430, 9000, 1750, 4100, 7714]
L = 3667800

# _________________________________________________________________
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

# _________________________________________________________________
# Funcao para contar o numero de lista de lista de resultados de saida do algoritmo
def countList(lst):
    return len(lst)

# _________________________________________________________________
# Contadores
n_iteraction = 0
void_solution = 0

# _________________________________________________________________
# Insire os resultados do Excel dos resultados positivos

from tqdm import tqdm

xls = pd.ExcelFile(r'D:\MESC\compilado.xlsx')
df_input = pd.read_excel(xls, 'ApendiceD')

df_final = pd.DataFrame()

for j in tqdm(df_input.index):
    pop = df_input.loc[j,"Tamanho da população"]
    child = df_input.loc[j,"Filhos"]
    sample_raw= df_input.loc[j,"Amostragem"]
    x_method_raw = df_input.loc[j,"Cruzamento"]
    mutation_raw = df_input.loc[j,"Mutações"]
    gen = df_input.loc[j,"Gerações"]
    n_seed = df_input.loc[j,"Semente"]
    sample_method = eval(sample_raw)
    x_method = eval(x_method_raw)
    mutation_method = eval(mutation_raw)

    # Reinicia temporizador
    start_time = 0
    end_time = 0

    # Inicia temporizador
    start_time = time.time()

    #Contador de iteracoes
    n_iteraction += 1

    #Configura o metodo NSGA II
    algorithm = NSGA2(
        pop_size=pop,
        n_offsprings=ceil(pop/child),
        sampling=sample_method,
        crossover=x_method,
        mutation=mutation_method,
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", gen)

    #Calculo e resultados de saída do NSGA II
    res = minimize(problem,
                algorithm,
                termination,
                seed= n_seed,
                save_history=True,
                verbose=False)

    X = res.X
    F = res.F
    
    #Calcule o tempo total para executar essas funcoes
    end_time = time.time()
    duration = end_time - start_time

    if str(res.X) == "None":
        #skip
        void_solution += 1

    else:

        # _________________________________________________________________
        #Identifica os limites de fronteira
        approx_ideal = fl = F.min(axis=0)
        approx_nadir = fu = F.max(axis=0)

        # _________________________________________________________________
        #Normalizacao
        nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

        fl = nF.min(axis=0)
        fu = nF.max(axis=0)

        # _________________________________________________________________
        #Selecao de resultado por pseudopeso

        weights = np.array([0.7, 0.5, 0.6, 0.8])
        k = PseudoWeights(weights).do(nF)
        best_selection_weight = F[k]

        # _________________________________________________________________
        #Aplicacao do metodo de convergencia por Hipervolume
        n_evals = []             # corresponding number of function evaluations\
        hist_F = []              # the objective space values in each generation
        hist_cv = []             # constraint violation in each generation
        hist_cv_avg = []         # average constraint violation in the whole population

        for algo in res.history:

            # armazenar o numero de funcoes de avaliacao
            n_evals.append(algo.evaluator.n_eval)

            # recuperar o valor otimo do algoritmo
            opt = algo.opt
            
            # armazenar o menor valor da restrição de violacao e a media em cada população
            hist_cv.append(opt.get("CV").min())
            hist_cv_avg.append(algo.pop.get("CV").mean())

            # filtra somente valores viaveis e unicos do espaco objetivo
            feas = np.where(opt.get("feasible"))[0]
            hist_F.append(opt.get("F")[feas])
     
        metric = Hypervolume(ref_point= np.array([1, 1, 1, 1]),
                            norm_ref_point=False,
                            zero_to_one=True,
                            ideal=approx_ideal,
                            nadir=approx_nadir)

        hv = [metric.do(_F) for _F in hist_F]

        df_info = pd.DataFrame (
            {'ID Iteração':n_iteraction,
            'Tamanho da população': pop,
            'Filhos': child,
            'Amostragem': sample_raw,
            'Cruzamento': x_method_raw,
            'Mutações': mutation_raw,
            'Gerações': gen,
            'Semente': n_seed,
            'Tempo de execução': [duration],
            'Resultados X': [countList(res.X)],
            'Resultados F': [countList(res.F)],
            'approx_ideal':str(approx_ideal),
            'approx_nadir':str(approx_nadir),
            'Linha melhor resultado Pseudo Peso':k,
            'Melhor resultado Pseudo Peso':str(best_selection_weight),
            'Resultado Hypervolume': max(hv)
            }
            )
        
        # Caminho do arquivo de Excel existente
        file_path = r'D:\\results_2\\Cenario_2.xlsx'

        # Escreve o dataframe na próxima linha vazia da planilha existente
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
            # Carrega o arquivo Excel existente
            writer.book = load_workbook(file_path)

            # Seleciona a planilha a ser escrita
            writer.sheets = {ws.title: ws for ws in writer.book.worksheets}
            worksheet = writer.sheets['Plan1']

            # Escreve as informações na próxima linha vazia da planilha
            for row in dataframe_to_rows(df_info, index=False, header=False):
                worksheet.append(row)

print("Numero de itercoes: %i" % n_iteraction)
print("Funcoes vazias: %i" % void_solution)
