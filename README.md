# UFC
Este proyecto aborda el análisis de datos de combates de la UFC en dos fases complementarias: primero, una exploración y visualización de la evolución histórica de los formatos, métodos de finalización y características de los luchadores; y segundo, la construcción y evaluación de varios modelos de clasificación (Regresión Logística, LDA, SVM, etc.) para predecir el ganador de un combate.

## 1. Objetivos principales
- Explorar y entender la estructura del dataset

- Inspeccionar forma, tipos de datos y estadísticas descriptivas iniciales.

- Cuantificar y tratar valores faltantes

- Identificar columnas con datos incompletos y aplicar imputación (mediana para numéricas, “NO DATA” para categóricas).

- Visualizar patrones históricos

- Analizar la evolución anual de formatos de combate y métodos de victoria (nocaut, sumisión, decisión).

- Describir distribuciones clave

- Estudiar la distribución de edades y estadísticas de golpeo de luchadores en esquina roja vs. azul.

- Mapear asociaciones y procedencias

- Identificar las principales escuelas/asociaciones de luchadores y su distribución por país.

- Preparar los datos para modelado

- Seleccionar variables numéricas, codificar categóricas (p. ej. color de esquina) y escalarlas.

- Evaluar distintos clasificadores

- Comparar precisión (accuracy) de Regresión Logística, LDA, KNN, Árboles de decisión, Naive Bayes y SVM con validación cruzada a 10 folds.

- Analizar curvas ROC-AUC

- Medir la capacidad discriminativa (ROC-AUC) de los tres mejores modelos: LR, LDA y SVM.

- Ajustar hiperparámetros del mejor modelo

- Utilizar GridSearchCV para optimizar parámetros de SVM (kernel RBF) y maximizar la precisión.

- Serializar el modelo final

- Guardar el clasificador óptimo (modeloUFC_finalizado_joblib.sav) para su posterior despliegue.

## 2. Principales conclusiones
- Volumen y calidad de datos: Tras la limpieza, quedaron 5 144 combates con 134 variables numéricas, y en fighters_data persistieron ~93 800 valores nulos en atributos no críticos.

- Evolución de formatos: A lo largo de las últimas décadas, los combates sin título (“Non-Title Fight”) dominan en número, con un ligero aumento proporcional de los combates de título.

- Métodos de victoria: La decisión mayoritaria crece de forma constante, mientras que los nocauts y sumisiones mantienen participaciones más estables pero secundarias.

- Distribución de edades: Tanto en esquina roja como azul, la edad de los luchadores se concentra alrededor de los 28–32 años, con una dispersión similar en ambas esquinas.

- Asociaciones y geografía: EEUU y Brasil lideran el número de luchadores registrados, seguidos a distancia por asociaciones de Canadá y Reino Unido.

- Desempeño de modelos:

- Regresión Logística: 68,18 % ± 2,9 % de accuracy.

- SVM (sin ajustar): 68,18 % ± 3,5 %.

- LDA: 67,66 % ± 3,8 %.

- Modelos basados en árboles y Naive Bayes quedan por debajo del 63 %.

- Modelo óptimo: SVM con kernel RBF y parámetros (C=1, γ=1) alcanzó 68,02 % de accuracy en validación cruzada y fue seleccionado y guardado.

## 3. Recomendaciones
- Ampliar y depurar el dataset: Incorporar combates más recientes y variables adicionales (grappling vs. striking, datos biométricos) para reducir gaps de información.

- Realizar selección de características: Aplicar métodos de feature-selection (p. ej. Recursive Feature Elimination) para quedarnos con las variables más predictivas y reducir dimensionalidad.

- Probar algoritmos de ensamblado: Ensayar Random Forest, XGBoost o LightGBM y comparar su ROC-AUC frente a SVM.

- Optimizar curvas ROC: Ajustar el umbral de decisión en función del análisis de coste de falsos positivos/negativos según el uso final.

- Implementar pipeline reproducible: Unir limpieza, imputación, escalado y modelado en un pipeline de scikit-learn para facilitar reentrenamientos periódicos.

- Validación temporal: Usar validación cruzada basada en fechas (TimeSeriesSplit) para evitar fuga de información y simular predicción en combates futuros.

- Monitorizar desempeño en producción: Registrar métricas en nuevos combates y reevaluar el modelo cada cierto periodo para detectar degradación.

- Explorar técnicas de balanceo: En caso de clases desbalanceadas (por método de victoria), aplicar SMOTE o submuestreo.

- Visualizar importancia de variables: Con Extratrees o coeficientes de LR, identificar las características clave que más influyen en la predicción.

- Documentar criterios de negocio: Coordinar con entrenadores/analistas de la UFC para integrar variables cualitativas que el modelo aún no considera.
