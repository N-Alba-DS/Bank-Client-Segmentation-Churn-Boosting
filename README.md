# Segmentación de Clientes y Predicción de Churn Bancario con LightGBM


Este repositorio documenta el desarrollo de un modelo de predicción de churn bancario, realizado como trabajo final para la competencia de Kaggle organizada por la materia de Data Mining en Economía y Finanzas de la “Maestría en Explotación de Datos y Descubrimiento del Conocimiento” de la UBA.

[Competencia de Kaggle - DMEyF 2024 Tercera](https://www.kaggle.com/competitions/fcen-dm-2024-prediccion-precio-de-propiedades/overview](https://www.kaggle.com/competitions/dm-ey-f-2024-tercera/leaderboard)

## Descripción del Trabajo

El proyecto implementa técnicas de **reducción de dimensionalidad**, **clustering**, **feature engineering**, y **boosting** (LightGBM) para predecir qué clientes se darán de baja en un plazo de dos meses. Se utilizaron múltiples herramientas como Optuna para optimización de hiperparámetros y UMAP para análisis exploratorio, destacando el impacto práctico en la segmentación de clientes. 

Primero se procedió a crear la variable target ('clase_ternaria') modificando el dataset original. Ésta sería una variable de tipo categórica y dado que las clases posibles serían los clientes que en dos meses para un determinada observación todavía permaneciesen en el banco ('CONTINUA'), los clientes que en un mes se darían de baja ('BAJA+1') y los clientes que en dos meses se darían de baja ('BAJA+2'), el problema del modelo a aplicar es uno del tipo clasificación multicategórica. En el caso particular de la competencia el mes a predecir fue *septiembre del año 2021*.

El proyecto se implemento usando lenguaje Python en múltiples scripts de tipo Jupyter Notebook (.ipynb), con la finalidad de que cada script resolviera algún problema específico del modelo o la competencia de Kaggle. Consecuentemente los scripts usados fueron:

- **Recurrentes** y **Funciones**: Scripts generales que serán llamados al script que se use mediante el comando mágico %run serían incorporados en todos los subsecuentes scripts para facilitar la articulación de código y legibilidad.
- **cluster_AE**: Utilizado para realizar el análisis exploratorio, primero se subsamplea la cantidad de observaciones de la clase negativa ("CONTINUA"), esto se realiza porque hay un desbalance significativo entre clases, buscando evitar que las observaciones de la clase mayoritaria puedan dominar la estructura del bosque aleatorio, haciendo más difícil identificar patrones en las clases minoritarias: "BAJA+1" y "BAJA+2". Posteriormenta se entrena un bosque aleatorio y luego se construye una matriz de distancias entre las muestras, basada en qué tan frecuentemente caen en las mismas hojas del conjunto de árboles. De esta forma se obtiene una medida de disimilaridad entre las muestras sin basarse directamente en la distancia euclídea, sino en la estructura creada por el bosque aleatorio. Posteriormente se utiliza UMAP para hacer reducción de dimensionalidad a dos dimensiones, y se aplica DBSCAN para clusterizar y segmentar los clientes. Se emplea esta estrategia contemplando la complejidad de las relaciones entre las variables y la falta de robustez de ciertos algoritmos de clustering como KNN frente a relaciones no lineales en los datos.
- **Back Testing**: Analisa el desempeño real del modelo centrandose en detectar en el mes de julio la cantidad de clientes BAJA+2
- **Data Quality**: Permitió observar la cantidad de nulls en las variables y hacer análisis exploratorio sobre el dataset
- **Drifting**: Utiliza PSI (Population Stability Index) para detectar feature drifting (data drift) en el dataset.
- **Feature Engineering (FE)**: Crea variables de lags y deltas en el dataset. Es decir, los valores para determinada variable en 1 y 2 meses anteriores a la observación, y los valores producto de restar los valores de 2 meses anteriores con el de 1 mes anterior y de restar los de 1 mes anterior con los actuales para visibilizar la tendencia de ese cliente en particular.
- **Punto de Corte**: Busca analizar a través del teorema central del límite cuál sería la cantidad idónea de estimulos que se enviarían a las predicciones realizadas por el modelo. Es decir, cuáles predicciones serían concebidas como 1 y que consecuentemente serían enviadas a Kaggle como predicción final del modelo.
- **VM Optuna**: Debido al tamaño del dataset luego de realizar FE se utilizó una Virtual Machine de Google con una configuración de 256gb de ram y 24 nucleos virtuales (12 real cores) a los efectos de realizar la optimización bayesiana de hiperparámetros del algoritmo con la biblioteca Optuna.
- **Kaggle Uploader**: Script utilizado para realizar un ensamble de voting que promediaría la predicción final a partir de un DataFrame de Pandas construido a partir del entrenamiento y predicción cambiando la semilla del modelo que incorpora los hiperparámetros obtenidos por VM Optuna. Esto se realizó a los efectos de controlar la varianza de las probabilidades de las predicciones y que los puntajes obtenidos no fuesen producto del azar en la semilla utilizada. Finalmente el script procedía a subir el modelo a Kaggle.

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


- Segmentación de clientes: Los clientes a darse de baja poseen como característica común una menor actividad en relación con el cliente que continua. Se los puede subdividir en 3 grupos con características específicas:
    -- Grupo 1 : se caracteriza por una baja participación con los servicios prestados por la empresa a lo largo de su relación con el banco. Utilizando principalmente los servicios relativos al débito y a la aplicación movil, es decir se caracterizan por realizar escuetamente extracciones de dinero, pago de servicios y ocacacionalmente operaciones de cheques, pero sobre todo, hacen un bajo uso de las tarjetas de credito al momento de la desvinculación.
      En su accionar preparan progresivamente su salida del banco con anticipación. ACOMPAÑANANDO de esta manera la baja escalonada de los sueldos depositados por sus empleadores en el banco. Motivo de ello se infiere que resulta este el       perfil tipico del trabajador promedio que recibe el pago de sueldo en la entidad, lo deja depositado y ante el cambio de trabajo cambia de entidad bancaria.
    -- Grupo 2: utiliza el banco para recibir el sueldo e inmediatamente debita la totalidad de su sueldo del mismo, o al ser trabajo no registrado recibe su sueldo en mano y una parte mínima en el banco.
    -- Grupo 3: contraposición al cliente low-engagement y el tipo 2 low-revenue que al momento de la desvinculación el saldo de la cuenta de los mismos oscila en 0 o en valores de deuda bajos, existe un cliente de alto riesgo. Pues, sumado a la perdida de ganancia que supone la salida del ciente se adiciona como característica típica de este segmento el más alto saldo deudor promedio en crédito y un saldo negativo de cuenta en promedio de 5 a 10 veces mayor que el cliente tipo 1 y 2. Centra su actividad en el uso de los servicios crediticios del banco, con un muy escueto uso de los servicios de emisión de cheques y pago de servicios en casos puntuales. La mayor cantidad de supuestos de clientes deudores morosos al momento de la desvinculación y de casos avanzados en el proceso de cierre de tarjetas de crédito se conglomeran en este perfil. Es decir, el cliente se da a la baja por no llegar a completar el pago mínimo de su tarjeta de crédito en los meses anteriores a la desvinculación. Puede también caracterizarse por empezar el proceso de cierre de la tarjeta pero no resulta excluyente. Si resulta indicativo que los montos del pago minimo necesario para no ser moroso de la tarjeta de crédito serán el doble en promedio a los clientes que continúan y que mientras más avanzada este la cuenta o cerrada más probabilidades hay de que esté en mora y por ende de darse de baja.
- Lightgbm resulta un algoritmo util para tratar las variables con valores nulos de manera tal que realizar imputaciones con tecnicas como Mice, Knn o Interpolación lineal pueden empeorar su rendimiento
- La creación de variables como feature engineering contribuyo ampliamente al mejoramiento de la perfomance del modelo
- Realizar back-testing fundamental para medir la perfomance del modelo sobre datos nunca vistos.
- La optimización de hiperparámetros mediante Optuna permite mejorar y medir la perfomance real del modelo.
- La utilización de voting a partir del uso de multiples semillas en el modelo para el entrenamiento y predicción del modelo resulta indispensable para lidiar con lidiar con el azar.

