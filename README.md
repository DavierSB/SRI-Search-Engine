# Proyecto de Recuperación de Información

## Autores:
Davier Sánchez Bello

Maykol Luis Martínez Rodríguez

## Modelo Booleano Extendido. Definición:
**D**: Vectores de los pesos asociados a los terminos de los documentos, se calculan usando tf-idf.

**Q**: Vectores de los pesos asociados a los tˊerminos de los documentos, se calculan usando tf-idf.

**F**: Espacio n-dimensional, propiedades del algebra booleana y operaciones del algebra lineal entre vectores.

**R**:$$ \text{sim}(q_{or},d_j) = (\frac{\sum_{i=1}^n (q_iw_i)^p}{\sum_{i=1}^n q_i })^\frac{1}{p} $$
$$ \text{sim}(q_{and},d_j) = 1-(\frac{\sum_{i=1}^n (q_i(1-w_i))^p}{\sum_{i=1}^n q_i })^\frac{1}{p} $$

## Consideraciones tomadas a la hora de desarrollar la solución:
Se implementó el modelo booleano extendido, usando una representación tf-idf de los documentos. Se usó cranfield como dataset por defecto pero se puede aplicar sobre cualquier otro dataset de la librería ir-datasets o sobre un conjuntos de documentos ubicado en la carpeta correspondiente. Se obtuvieron métricas poco satisfactorias en las queries de prueba del dataset empleado, pero tras compararse con los resultados no solo del modelo booleano sino también del modelo vectorial, así como datos recopilados de otros usuarios en situaciones similares, se determinó que esto no era razón para dudar de la correctitud del algoritmo, sino que Cranfield, a pesar de ser un documento clasico para el benchmarking de sistemas de recuperacion de informacion, peca de tener una cantidad relativamente pequenha de documentos, todos los cuales giran alrededor de un tema muy muy especifico.

## Ejecución del proyecto y definición de consulta
Para ejecutar el motor de busqueda se usa:
streamlit run src/code/gui.py
Para ejecutar el tester se usa:
streamlit run src/code/gui_tester.py

Las consultas se realizarán en lenguaje natural, preferiblemente en inglés.
## Explicación de la solución desarrollada:
Se procesan los documentos del dataset Cranfield, incluyendo la tokenización, eliminación de ruido, eliminación de stopwords, lematización, se representan los documentos usando vectores tf-idf, se calcula la matriz de co-ocurrencia, y se determinan los autores de cada documento. 
Naturalmente, estos datos se calculan una vez y se almacenan en archivos para su posterior uso. 
La matriz de co-ocurrencia tiene varias versiones, en la empleada en este trabajo se calcula la cercanía entre dos palabras aumentándose esta cada vez que aparecen en la misma vecindad de longitud 2 dentro del corpus. 
Pasando a la query, se realiza un procesamiento similar al realizado a los documentos, se implementa además una expansión de consulta que utiliza la matriz de co-ocurrencia para determinar las palabras más relacionadas con la query, usando:
$$ \text{coo}(q,w)=\sum_{i=0}^nmatrix[wq_i]$$
Usamos la librería wordnet para agregar los sinónimos e hiperónimos, naturalmente existen otros tipos (hasta 7) de palabras relacionadas semánticamente con la query, pero agregar demasiadas palabras puede ser contraproducente. La query extendida, consiste en una concatenación usando ANDs de grupos de palabras semánticamente similares relacionadas entre sí como un OR. Los grupos de palabras, compuestos por sinónimos e hiperónimos son computado para cada una de las palabras de la query original, así como para aquellas que presentaban la mayor similitud total con las palabras de la query pero se encontraban fuera de ella, entiéndase por similitud total a la suma de las cercacias de esa palabra con cada una de las palabras en la query. Luego, tenemos la precaución de calcular el tfidf para cada una de las palabras en la query expandida, contando correctamente las apariciones que tenían las palabras de la query original, que a diferencia de las expandidas sí les es admisible tener un tf mayor que uno. No obstante, el peso tfidf es reducido para las palabras anidadas ya sea por similitud semántica o por cercanía en el corpus, y los factores por los cuales son reducidas son ajustables. Una vez tenemos este vector, usamos para el calculo de la similitud usando las fórmulas descritas anteriormente.

También se implementó sistema de recomendación, este está basado en los autores de cada documento, para esto se almacena un ranking de los autores más referenciados usando los resultados de las búsquedas del usuario, esto se logra agregando los scores de los documentos, ya sea que se hallan devuelto o no, al valor almacenado anteriormente, esto hace que la importancia de la query actual sea naturalmente elevada, esto se debe a que el valor almacenado está normalizado y al mismo tiempo, le da importancia a las queries anteriores.

## Insuficiencias y mejoras:
Si bien los resultados obtenidos son poco alentadores, no consideramos que esto se deba a deficiencias en el algoritmo por lo que sin hacer cambios radicales que incluyan la utilización de machine learning no creemos posible lograr una mejora sustancial, con los campos de conocimientos que abarca este trabajo los resultados se pueden considerar satisfactorios. Sin embargo, sí creemos que la retroalimentación podría permitir cierta mejora. Una mejora no relacionada con la recuperación de información en sí sino con la utilidad y factibilidad del programa desde el punto de vista del usuario sería la capacidad de acceder a los documentos recuperados, esto no solo mejoraría las ventajas prácticas sino que nos permitiría implementar una retroalimentación tomando la información que se da implícitamente por el usuario según los documentos accedidos.
