# Segmentación de Clientes y Predicción de Churn Bancario con LightGBM


Este repositorio documenta el desarrollo de un modelo de predicción de churn bancario, realizado como trabajo final para la competencia de Kaggle organizada por la materia de Data Mining en Economía y Finanzas de la “Maestría en Explotación de Datos y Descubrimiento del Conocimiento” de la UBA.

[Competencia de Kaggle - DMEyF 2024 Tercera](https://www.kaggle.com/competitions/dm-ey-f-2024-tercera/leaderboard)

## Descripción del Trabajo

El proyecto implementa técnicas de **reducción de dimensionalidad**, **clustering**, **feature engineering**, y **boosting** (LightGBM) para predecir qué clientes se darán de baja en un plazo de dos meses tomando como mes de test el de septiembre de 2021. 

Primero se procedió a crear la variable target ('clase_ternaria') modificando el dataset original. Ésta sería una variable de tipo categórica y dado que las clases posibles serían los clientes que en dos meses para un determinada observación todavía permaneciesen en el banco ('CONTINUA'), los clientes que en un mes se darían de baja ('BAJA+1') y los clientes que en dos meses se darían de baja ('BAJA+2'), el problema del modelo a aplicar es uno del tipo clasificación multicategórica. En el caso particular de la competencia el mes a predecir fue *septiembre del año 2021*.

El proyecto se implemento usando lenguaje Python en múltiples scripts de tipo Jupyter Notebook (.ipynb), con la finalidad de que cada script resolviera algún problema específico del modelo o la competencia de Kaggle. Consecuentemente los scripts usados fueron:

- **Recurrentes** y **Funciones**: Scripts generales que serán llamados al script que se use mediante el comando mágico %run serían incorporados en todos los subsecuentes scripts para facilitar la articulación de código y legibilidad.
- **Cluster_AE**: Utilizado para realizar el análisis exploratorio, primero se subsamplea la cantidad de observaciones de la clase negativa ("CONTINUA"), esto se realiza porque hay un desbalance significativo entre clases, buscando evitar que las observaciones de la clase mayoritaria puedan dominar la estructura del bosque aleatorio, haciendo más difícil identificar patrones en las clases minoritarias: "BAJA+1" y "BAJA+2". Posteriormenta se entrena un bosque aleatorio y luego se construye una matriz de distancias entre las muestras, basada en qué tan frecuentemente caen en las mismas hojas del conjunto de árboles. De esta forma se obtiene una medida de disimilaridad entre las muestras sin basarse directamente en la distancia euclídea, sino en la estructura creada por el bosque aleatorio. Posteriormente se utiliza UMAP para hacer reducción de dimensionalidad a dos dimensiones, y se aplica DBSCAN para clusterizar y segmentar los clientes. Se emplea esta estrategia contemplando la complejidad de las relaciones entre las variables y la falta de robustez de ciertos algoritmos de clustering como KNN frente a relaciones no lineales en los datos.
- **Back Testing**: Analisa el desempeño real del modelo centrandose en detectar en el mes de julio la cantidad de clientes BAJA+2
- **Data Quality**: Permitió observar la cantidad de nulls en las variables y hacer análisis exploratorio sobre el dataset. Utiliza PSI (Population Stability Index) para detectar feature drifting (data drift) en el dataset.
- **Feature Engineering (FE)**: Crea variables de lags y deltas en el dataset. Es decir, los valores para determinada variable en 1 y 2 meses anteriores a la observación, y los valores producto de restar los valores de 2 meses anteriores con el de 1 mes anterior y de restar los de 1 mes anterior con los actuales para visibilizar la tendencia de ese cliente en particular. 
- **Punto de Corte**: Busca analizar a través del teorema central del límite cuál sería la cantidad idónea de estimulos que se enviarían a las predicciones realizadas por el modelo. Es decir, cuáles predicciones serían concebidas como 1 y que consecuentemente serían enviadas a Kaggle como predicción final del modelo.
- **VM Optuna**: Encargado de la optimización. Debido al tamaño del dataset luego de realizar FE se utilizó una Virtual Machine de Google con una configuración de 256gb de ram y 24 nucleos virtuales (12 real cores) a los efectos de realizar la optimización bayesiana de hiperparámetros del algoritmo con la biblioteca Optuna. 
- **Predicción final**: Script utilizado para realizar un ensamble de voting que promediaría la predicción final a partir de un DataFrame de Pandas construido del entrenamiento y predicción del modelo pero cambiando multiples semillas del modelo que incorpora los hiperparámetros obtenidos por VM Optuna. Esto se realizó a los efectos de controlar la varianza de las probabilidades de las predicciones y que los puntajes obtenidos no fuesen producto del azar en la semilla utilizada.


## Técnicas Utilizadas

- **Preprocesamiento y Manipulación de Datos**:
  - Limpieza y preparación de datos con Pandas.
  - Concatenación de diferentes modelos y conjuntos de datos.
  - Subsampleo

- **Codificación de Variables Categóricas**:
  - **One-Hot Encoding**.
  - **Ordinal Encoding**.

- **Ingeniería de Características (Feature Engineering)**:
  - Creación de nuevas variables relevantes (LAGS y DELTAS).
  - Transformación y reescalado de la variable objetivo ('clase_ternaria').

- **Validación y Optimización del Modelo**:
  - Validación cruzada con **KFold** en la función .cv de lightgbm.
  - Búsqueda de hiperparámetros óptimos con **Optuna**.



## Conclusiones

### Segmentación de clientes

- **Características comunes**: Los clientes a darse de baja poseen como característica común una menor actividad en relación con el cliente que continúa. Se los puede subdividir en 3 grupos con características específicas:

  - **Grupo 1**:
    - Se caracteriza por una baja participación con los servicios prestados por la empresa a lo largo de su relación con el banco.
    - Utilizan principalmente los servicios relativos al débito y a la aplicación móvil, es decir, realizan:
      - Escuetamente, extracciones de dinero.
      - Pago de servicios.
      - Ocasionalmente, operaciones de cheques.
    - Hacen un bajo uso de las tarjetas de crédito al momento de la desvinculación.
    - Preparan progresivamente su salida del banco con anticipación, acompañando una baja escalonada de los sueldos depositados por sus empleadores.
    - Se infiere que este perfil corresponde al trabajador promedio que:
      - Recibe el pago de su sueldo en la entidad.
      - Deja depositado el dinero.
      - Cambia de banco al cambiar de empleo.

  - **Grupo 2**:
    - Utilizan el banco para recibir el sueldo e inmediatamente debitan la totalidad del mismo.
    - En el caso de trabajos no registrados, reciben su sueldo en mano y solo una parte mínima se deposita en el banco.

  - **Grupo 3**:
    - Contrapuesto a los clientes "low-engagement" (Grupo 1) y "low-revenue" (Grupo 2), este segmento representa clientes de **alto riesgo**.
    - Características principales:
      - Al momento de la desvinculación, el saldo de la cuenta oscila en 0 o en valores de deuda bajos para la mayoría.
      - Sin embargo, una porción significativa tiene un saldo deudor promedio en créditos y cuentas negativas de 5 a 10 veces mayor que los Grupos 1 y 2.
    - Centran su actividad en el uso de servicios crediticios del banco, con un uso escueto de:
      - Emisión de cheques.
      - Pago de servicios en casos puntuales.
    - La mayoría de los clientes morosos al momento de la desvinculación se agrupan en este perfil:
      - Muchos no completan el pago mínimo de sus tarjetas de crédito en los meses anteriores a la desvinculación.
      - Es común que inicien el proceso de cierre de tarjetas, aunque esto no es excluyente.
      - Los montos del pago mínimo para evitar la mora son el doble en promedio que los de clientes que continúan.
    - Cuanto más avanzada está la cuenta en mora o cerrada, mayor es la probabilidad de que el cliente se dé de baja.

### Modelo de predicción

- Lightgbm resulta un algoritmo util para tratar las variables con valores nulos de manera tal que realizar imputaciones con tecnicas como Mice, Knn o Interpolación lineal pueden empeorar su rendimiento.
- La creación de variables como feature engineering contribuyo ampliamente al mejoramiento de la perfomance del modelo.
- Realizar back-testing fundamental para medir la perfomance del modelo sobre datos nunca vistos.
- La optimización de hiperparámetros mediante Optuna permite mejorar y medir la perfomance real del modelo.
- La utilización de voting a partir del uso de multiples semillas en el modelo para el entrenamiento y predicción del modelo resulta indispensable para lidiar con lidiar con el azar.


El script fue desarrollado en Python y utiliza las siguientes **librerías** principales:

**Visualización**:
  - Plotly
  - Seaborn
  - Matplotlib

 **Modelo**:
  - Pandas
  - Numpy
  - Lightgbm
  - Sklearn
  - Openpyxl
  - Optuna
  - Datetime
  - Imblearn.undersampling

**Análisis exploratorio**
  - DBSCAN
  - UMAP
