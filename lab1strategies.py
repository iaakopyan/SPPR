import numpy as np
answer={}
                                #7
matrix = np.array([[8, 11, 9, 3, 5],
                  [3, 8, 3, 0, 6],
                  [9, 11, 4, 12, 6], #4
                  [11, 6, 11, 11, 1],
                  [8, 4, 0, 7, 7]])
P = (0.55, 0.0, 0.18, 0.0, 0.26)
Q = (0.0, 0.0, 0.18, 0.01, 0.8)
lower_price = max([min(x) for x in matrix])
upper_price = min([max(x) for x in np.rot90(matrix)])
a= {lower_price, upper_price}
if len(a)!=1:
    print("\033[32mСедловой точки нет\033[39m")
else:
    print(f"\033[31mСедловая точка:{a}")
    exit(1)
buff=0
for i,pin in zip(matrix, P):
    # H_A(P, Q) = sum_i=1,m  *   sum_j=1,n * a_ij * p_i * p_j
    buff+=pin*sum([x*y for x,y in zip(i,Q)])
answer["H(P,Q)"]=buff #выигрыш в первой ситуации
for k,i in enumerate(np.rot90(matrix),1):
    answer[f"H(P,B{k})"]=sum([x*y for x,y in zip(i,P)])
print('Вероятности выигрыша игрока А в данных ситуациях:')


print('\033[35m',*(answer.items()), sep='\n')

