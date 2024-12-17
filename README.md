# Churn-prediction-with-boosting



# Predicción de Precios de Propiedades en Buenos Aires utilizando Random Forest
Este repositorio contiene el código y la documentación del modelo de predicción de churning bancario, desarrollado como trabajo final para la competencia de Kaggle organizada por la materia de Data Mining en Economía y Finanzas de la “Maestría en Explotación de Datos y Descubrimiento del Conocimiento” de la UBA.

[Competencia de Kaggle - DMEyF 2024 Tercera]([https://www.kaggle.com/competitions/fcen-dm-2024-prediccion-precio-de-propiedades/overview](https://www.kaggle.com/competitions/dm-ey-f-2024-tercera/leaderboard)

## Descripción del Trabajo

Este proyecto aplica técnicas de análisis exploratorio, feature engineering y optimización de datos impartidas en el segundo semestre de la **Maestría en Explotación de Datos y Descubrimiento del Conocimiento** de la UBA. El objetivo del modelo creado fue implementar ensambles de tipo boosting, específicamente haciendo uso del algoritmo light gbm para predecir los sujetos que se darán a la baja en un periodo de dos meses a futuro, en el caso particular de la competencia el mes a predecir fue septiembre del año 2021.

Primero se procedió a crear la variable target ('clase_ternaria') modificando el dataset original. Ésta sería una variable de tipo categórica y dado que las clases posibles serían los clientes que en dos meses para un determinada observación todavía permaneciesen en el banco ('CONTINUA'), los clientes que en un mes se darían de baja ('BAJA+1') y los clientes que en dos meses se darían de baja ('BAJA+2'), el problema del modelo a aplicar es uno del tipo clasificación multicategórica. 

El proyecto se implemento usando lenguaje Python en múltiples scripts de tipo Jupyter Notebook (.ipynb), con la finalidad de que cada script resolviera algún problema específico del modelo o la competencia de Kaggle. Consecuentemente los scripts usados fueron:

- **Recurrentes** y **Funciones**: Scripts generales que se mediante el comando mágico %run serían incorporados en todos los subsecuentes scripts para facilitar la articulación de código y legibilidad.
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
  - pandas
  - numpy
  - lightgbm
  - sklearn
  - openpyxl
  - optuna
  - datetime
  - imblearn.undersampling


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

- La segmentación por tipo de propiedad (Casas, departamentos y cocheras) y jerárquica (entre departamentos de Puerto Madero y el resto de CABA), permitió reducir el sesgo y mejorar la precisión de las predicciones.
- El uso de técnicas de imputación avanzadas (MICE y KNN) contribuyó a manejar eficazmente los datos faltantes.
- La optimización de hiperparámetros mediante GridSearchCV y la validación cruzada fueron fundamentales para evitar el sobreajuste. Remarcando que debido a la inexperiencia al momento del proyecto no se contaba con conocimientos de otras herramientas más eficientes de optimización que implementaran optimización bayesiana. 
