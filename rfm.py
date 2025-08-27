
import pandas as pd
import numpy as np
import datetime
from functools import reduce

#Saldos = pd.read_csv(f'/Users/***.txt', sep = '\t', names = ['FECHAAN', 'NEGOCIO', 'CUENTA', 'SDO', 'TIPOCTA', 'STATCTA', 'MONPAGAR', 'FCREACTA', 'FULTPAGO', 'MULTPAGO', 'FULTCOMPRA', 'MULTCOMPRA', 'CAPPAGO', 'SDO2', 'IFIN', 'IMOR', 'REGIONOPER'])
#Saldos['FECHAAN'] = pd.to_datetime(Saldos['FECHAAN'], format='%Y/%m/%d')
#Saldos['FULTCOMPRA'] = pd.to_datetime(Saldos['FULTCOMPRA'], format='%Y/%m/%d')

def dfCargos(file, año, neg):
    """
    Procesa archivo de cargos de clientes y agrupa por mes y cuenta.

    Args:
        file (str): Nombre del archivo de cargos.
        año (int): Año de análisis.
        neg (int): Código del negocio.

    Returns:
        DataFrame: Tabla consolidada por cuenta y mes.
    """

    Cargos = pd.read_csv(f'/Users/andrea.rodriguez/Documents/VSM-churn/Churn PoC/{file}', sep = '\t', names = ['FECMOVTO', 'NEGOCIO', 'CUENTA', 'IMPMOV', 'IDMOVTO'])
    Cargos['FECMOVTO'] = pd.to_datetime(Cargos['FECMOVTO'], format='%Y/%m/%d')
    Cargos = Cargos[Cargos.NEGOCIO == neg]
    if neg == 5:
        Cargos['NEGOCIO'] = Cargos['NEGOCIO'].replace(5, 1) 

    # Solo tipo de movimiento 10 o 20
    Cargos = Cargos[Cargos.IDMOVTO.isin(['10', '20'])]
    Cargos = Cargos.groupby(pd.Grouper(key='FECMOVTO', freq='M'))
    Cargos = [group for _,group in Cargos]    #Agrupar por mes de cargo

    for i in range(len(Cargos)):    #Suma monto de cargo de cada cuenta por mes
        globals()[f"Cargos{i}"] = Cargos[i].groupby(['CUENTA'], sort=True, as_index=False).sum('IMPMOV')
   
    dfs = []
    for i in range(len(Cargos)): #Renombrar columnas, la primera es el numero de cargos y la segunda el monto total del mes
        j = i+1
        globals()[f"Cargos{i}"].rename(columns={'NEGOCIO': pd.to_datetime(f'{año}-'f'{j}-01'), 'IMPMOV' : pd.to_datetime(f'{año}-'f'{j}-01')}, inplace=True)  #En groupby el no de neg se sumó por lo tanto ahora tenemos el no de cargos por cuenta                 
        dfs.append(globals()[f"Cargos{i}"])   #Crea una lista de los dataframes para posteriormente unirlos usando merge

    Cargos = reduce(lambda  left,right: pd.merge(left,right,on=['CUENTA'], how='outer'), dfs) #Merge los 12 dataframes obtenidos en los ciclos for
    return Cargos

cargos_2020 = dfCargos("Cargos 01-01-2020 - 31-12-2020.txt", 2020, 1)
cargos_2021 = dfCargos("Cargos 01-01-2021 - 31-12-2021.txt", 2021, 1)
cargos_2022 = dfCargos("Cargos 01-01-2022 - 31-12-2022.txt", 2022, 1)

a = cargos_2020.merge(cargos_2021, how='outer', on = 'CUENTA')
cargos = a.merge(cargos_2022, how='outer', on = 'CUENTA')

cargos = cargos.set_index('CUENTA')
cargos['Recency'] = " "
cargos['Frequency'] = " "
cargos['Monetary'] = " "

def difdate (d1,d2):       #Función que calcula número de meses entre dos fechas
    x = (d1-d2)
    return round(x/np.timedelta64(1, 'M'))

for j in range(len(cargos)):         #ciclo sobre todos los renglones (cuentas)
    for i in range(len(cargos.columns)-2):         #ciclo sobre las columnas
        k=len(cargos.columns)-2-i
        flag = cargos.iloc[j,k-1]
        result = cargos.iloc[j,k]
        if pd.isna(result) and pd.notna(flag):
            d1 = cargos.columns[-4]
            d2 = cargos.columns[k]
            df = cargos.columns[k-1]
            x = difdate(d1,df)
            cargos.at[cargos.index[j], 'Recency'] = x
            break
        if pd.notna(cargos.iloc[j,71]):
            cargos.at[cargos.index[j], 'Recency'] = 0
            break

#######################################
cargos.to_csv('/Users/andrea.rodriguez/Documents/Mejores clientes/recency_1')
cargos_r = pd.read_csv('/Users/andrea.rodriguez/Documents/Mejores clientes/recency_1')

cuentas = cargos_r.dropna().drop(['Frequency', 'Monetary'], axis=1)
cuentas['CUENTA'].to_csv('/Users/andrea.rodriguez/Documents/Mejores clientes/cuentas20202022_1')

import matplotlib.pyplot as plt
cargos["Recency"].hist(bins=14, edgecolor='black', grid=False)
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.show()
######################################

cargos = cargos.fillna(0)

def fm(file, n):
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=14)

    x = len(file.columns)-(3+2*n)
    y = len(file.columns)-3

    def freq(i,j):
        result_f=0
        result_m=0
        for i in range(x, y, 2):
            result_f += file.iloc[j,i]
            result_m += file.iloc[j,i+1]
            file.at[file.index[j], 'Frequency'] = result_f
            file.at[file.index[j], 'Monetary'] = result_m

    for j in range(len(file)):      #ciclo sobre todos los renglones (cuentas)
        for i in range(x, y, 2):        #últimas 9 columnas
            executor.submit(freq,i,j)
    
    return file

cargos_rfm = fm(cargos,9)

cargos_rfm

cargos_rfm.to_csv('/Users/andrea.rodriguez/Documents/Mejores clientes/rfm_1')




cargos_rfm = pd.read_csv('/Users/andrea.rodriguez/Documents/Mejores clientes/rfm_5')

cargos_rfm = cargos_rfm.set_index('CUENTA')
rec = cargos_rfm["Recency"].sort_values(ascending=True).reset_index()
mon = cargos_rfm["Monetary"].sort_values(ascending=False).reset_index()
freq = cargos_rfm["Frequency"].sort_values(ascending=False).reset_index()


def rfm(file):
    j=0
    i=5
    n=1
    while j < len(file):
        file.at[file.index[j], 'Result'] = i
        j+=1
        if j > (len(file)/5)*n:
            i-=1
            n+=1
    return file

rfm(rec)
rfm(freq)
rfm(mon)

rfm = rec.merge(freq, how='outer', on = 'CUENTA')
rfm = rfm.merge(mon, how='outer', on = 'CUENTA')

rfm.drop(columns=['Recency','Frequency','Monetary'], inplace=True, axis=1)
rfm.rename(columns={'Result_x': 'Recency', 'Result_y' : 'Frequency', 'Result' : 'Monetary'}, inplace=True)


rfm.to_csv('/Users/andrea.rodriguez/Documents/Mejores clientes/rfm_5_values')




rfm = pd.read_csv('/Users/andrea.rodriguez/Documents/Mejores clientes/rfm_5')
rfm
rfm.describe()
import matplotlib.pyplot as plt
plt.hist(rfm)
plt.show()

rfm['Recency'].mean()
rfm['Frequency'].mean()
rfm['Monetary'].mean()




##################
#   Clustering   #
##################

rfm = pd.read_csv('/Users/andrea.rodriguez/Documents/Mejores clientes/rfm_5_score', index_col=[0])
x = np.array(rfm[['Recency', 'Frequency', 'Monetary']])
y = np.array(rfm['CUENTA'])

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def find_best_k():
    sum_of_squared_distances = []
    K=range(5,65)
    for k in K:
        km=KMeans(n_clusters=k)
        km=km.fit(x)
        sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('sum_of_squared_distances')
    plt.title('Elbow method for optimal k neg 5')
    plt.show()  
find_best_k()

kmeans = KMeans(n_clusters=8).fit(x)
centroids = kmeans.cluster_centers_
print(centroids)

rfm['cluster'] = kmeans.labels_
rfm['cluster'].value_counts()

for i in range(13):
    print(i)
    rfm[rfm['cluster'] == i].mean()

import pickle
pickle.dump(kmeans, open("kmeans_5_8clusters.pkl", "wb"))
rfm.to_csv('/Users/andrea.rodriguez/Documents/Mejores clientes/rfm_5_8clusters')


#################
# Another approach

#################

rfm = pd.read_csv('/Users/andrea.rodriguez/Documents/Mejores clientes/rfm_5')
rfm = rfm[['CUENTA', 'Recency', 'Frequency', 'Monetary']]
rfm.to_csv('/Users/andrea.rodriguez/Documents/Mejores clientes/rfm_5_values')

rfm2 = pd.read_csv('/Users/andrea.rodriguez/Documents/Mejores clientes/rfm_5_values', index_col=[0])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
rfm2[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(rfm2[['Recency', 'Frequency', 'Monetary']])

x = np.array(rfm2[['Recency', 'Frequency', 'Monetary']])
y = np.array(rfm2['CUENTA'])

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def find_best_k():
    sum_of_squared_distances = []
    K=range(1,35)
    for k in K:
        km=KMeans(n_clusters=k)
        km=km.fit(x)
        sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('sum_of_squared_distances')
    plt.title('Elbow method for optimal k neg 5')
    plt.show()  
find_best_k()

kmeans = KMeans(n_clusters=8).fit(x)

rfm2['cluster'] = kmeans.labels_
rfm2['cluster'].value_counts()

for i in range(8):
    print(i)
    rfm[rfm['cluster'] == i].mean()


#########################

rfm
data = rfm2.merge(rfm, on = 'CUENTA')
data

x = np.array(rfm[['Recency_x', 'Frequency_x', 'Monetary_x', 'Recency_y', 'Frequency_y', 'Monetary_y']])
y = np.array(rfm['CUENTA'])

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def find_best_k():
    sum_of_squared_distances = []
    K=range(1,45)
    for k in K:
        km=KMeans(n_clusters=k)
        km=km.fit(x)
        sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('sum_of_squared_distances')
    plt.title('Elbow method for optimal k neg 5')
    plt.show()  
find_best_k()