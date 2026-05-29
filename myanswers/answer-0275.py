import numpy as np
from sklearn.feature_extraction.text import CountVectorizer # type: ignore

def crear_matriz_firmas(df, text_col, num_hashes, random_state=42):
    """
    Crea una matriz de firmas MinHash a partir de una columna de textos en un DataFrame.
    """
    # 1 y 2. Extraer la columna de textos y vectorizar (Filas = docs, Columnas = vocabulario)
    cv = CountVectorizer(binary=True)
    matriz_docs_shingles = cv.fit_transform(df[text_col]).toarray()
    
    # 3. Transponer la matriz para obtener Filas = shingles, Columnas = documentos
    matriz_shingles_docs = matriz_docs_shingles.T
    n_shingles, n_documentos = matriz_shingles_docs.shape
    
    # 4. Inicializar generador de números aleatorios y la matriz de firmas
    rng = np.random.RandomState(random_state)
    firmas = np.zeros((num_hashes, n_documentos), dtype=int)
    
    # Generar permutaciones y calcular el MinHash
    for i in range(num_hashes):
        # Generar la permutación de índices para los shingles
        perm = rng.permutation(n_shingles)
        
        # Aplicar la permutación a las filas de la matriz
        matriz_permutada = matriz_shingles_docs[perm, :]
        
        # Para cada documento, buscar el índice del primer '1'
        for d in range(n_documentos):
            # np.where encuentra los índices donde la condición es verdadera
            # [0][0] extrae el primer índice encontrado
            primer_indice_uno = np.where(matriz_permutada[:, d] == 1)[0][0]
            firmas[i, d] = primer_indice_uno
            
    # 5. Devolver la matriz de firmas
    return firmas