def bubble_sort(arr):
    n = len(arr)
    # Recorrer todos los elementos del arreglo
    for i in range(n):
        # Últimos i elementos ya están en su lugar
        for j in range(0, n - i - 1):
            # Si el elemento actual es mayor que el siguiente, intercámbialos
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

# Ejemplo de uso
lista = [64, 34, 25, 12, 22, 11, 90]
bubble_sort(lista)
print("Lista ordenada:", lista)
