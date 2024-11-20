def quick_sort(A, p, r):
    if p < r:
        j = pivot(A, p, r)
        quick_sort(A, p, j - 1)
        quick_sort(A, j + 1, r)

def pivot(A, p, r):
    piv = A[p][1]
    i = p + 1
    j = r

    while i < j:
        while i < r and A[i][1] <= piv:
            i += 1
        while A[j][1] > piv:
            j -= 1
        if i < j:
            A[i], A[j] = A[j], A[i]
    A[p], A[j] = A[j], A[p]
    
    return j

# Lista estática de números a ordenar, incluyendo los índices originales
A = [(i, val) for i, val in enumerate([10, 2, 7, 8, 9, 1, 5, 11, 75])]
n = len(A)

quick_sort(A, 0, n - 1)

# Imprimir el arreglo ordenado con los índices originales
print("El arreglo ordenado con índices originales:")
for index, value in A:
    print(f"Index: {index}, Value: {value}")
