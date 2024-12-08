---
marp: true
theme: default
paginate: true
footer: 19 de Noviembre, 2024
math: katex
---

<style>
    
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}

.center-align {
    margin-right: auto !important;
    margin-left: auto !important;
    text-align: center;
}

.right-align {
    text-align: right;
}

.figure-container {
  display: flex;
  justify-content: space-evenly;
  margin: 0 0 0 0;
  padding: 0 0 0 0;
}

</style>

# Text-like Encoding of Collaborative Information in Large Language Models for Recommendation


#
Diego Alejandro Quezada Campos
Diego Ignacio Campusano Schialer
Nicolás Javier Espinoza Silva


---
# Contexto

* Creciente interés en adaptar modelos de lenguaje grandes (LLMs) para sistemas de recomendación (LLMRec)
* La información colaborativa es crucial para modelar los intereses de los usuarios, pero difiere en modalidad de los datos textuales

---
# Marco teórico


* LLM: Modelos de Deep Learning basados en la arquitectura *transformer* (Vaswani et al., 2017) la cual les permite extraer información semántica de secuencias de palabras

* RecSys: Algoritmos que dado un usuario y una lista de ítems recomiendan uno o varios ítems que idealmente sean de preferencia para el usuario

* LLMRec: Área que estudia la aplicación y adaptación de LLMs para que sean utilizados por sistemas recomendadores

---

* Tokens: Unidades en las que LLMs descomponen cadenas de texto, pueden ser palabras, caracteres, signos de puntuación o combinaciones de estas

* Embeddings: Vectores de números reales que representan palabras, imágenes u otras formas de información de manera que puedan ser procesadas por modelos de Machine Learning de manera eficaz
---
# Problema

Cómo codificar la información colaborativa en LLMs para recomendaciones


---
# Estado del arte

## Estrategia 1

* Incorporar tokens adicionales en LLMs para representar usuarios y productos, ajustando los datos de interacción para capturar información colaborativa en los embeddings ([Zheng et al., 2024](https://arxiv.org/abs/2311.09049); [Hua et al., 2023](http://dx.doi.org/10.1145/3624918.3625339))
* Altera el espacio generativo del LLM, potencialmente comprometiendo las funcionalidades originales
* Baja eficacia debido a la naturaleza de bajo rango de la información, lo que genera redundancia en la tokenización


---

## Estrategia 2

* Usar un modelo externo de factores latentes para capturar la información colaborativa y luego mapearla al espacio de embeddings de los tokens, evitando la necesidad de aprenderlos desde cero ([Zhang et al., 2024](https://arxiv.org/abs/2310.19488); [Li et al., 2023](https://arxiv.org/abs/2312.02443); [Liao et al., 2024](https://arxiv.org/abs/2312.02445))
* Es efectivo, pero requiere entrenar un modelo adicional



---
# Contribución

Tomando inspiración en:

* La viabilidad de binarizar embeddings colaborativos sin comprometer el rendimiento ([Tan et al., 2020](https://arxiv.org/abs/2003.01917))
* LLMs pueden realizar operaciones a nivel de bits de forma natural o después de un ajuste por instrucciones ([Savelka et al., 2023](http://dx.doi.org/10.1145/3568813.3600142))


La contribución de los autores es:

* **BinLLM**: un enfoque innovador de LLMRec que integra información colaborativa en los LLMs al codificar los embeddings colaborativos como secuencias binarias.

---
# BinLLM

<div style="display: flex; justify-content: space-around; margin-top: 20px;">
  
  <img src="./images/bin-llm.png" alt="Interacciones por usuario (Prueba)" style="width: 75%; height: auto;"/>

</div>

---

## Plantilla de prompt

#Question: A user has given high ratings to the following books: ``ItemTitleList``. Additionally, we have information about the user's preferences encoded in the feature ``UserID``. Using all available information, make a prediction about whether the user would enjoy the book titled ``TargetItemTitle`` with the feature ``TargetItemID``? Answer with "Yes" or "No".
#Answer:

---

## Codificación de Información Colaborativa

Dado un usuario $u$ y un item $i$, obtendremos sus embeddings a través de un modelo $f$:

$$
e_u = f (u; \theta)
$$

$$
e_i = f (i; \theta)
$$


Para binarizar los embeddings, se utiliza una capa totalmente conectada:

$$
\bm{h}_u = sign(\sigma(We_u + b))
$$

$$
\bm{h}_i = sign(\sigma(We_i + b))
$$

---

En donde:

* $e_u \in \mathcal{R}^d$
* $e_i \in \mathcal{R}^d$
* $\bm{h}_u \in \{0, 1\}^d$
* $\bm{h}_i \in \{0, 1\}^d$
* $W \in \mathcal{R}^{d \times d}$
* $b \in \mathcal{R}^d$.

---

## Configuración

* Tarea: Predicción de click
* Conjunto de datos: MovieLens-1M y Amazon-Book
* LLM: Vicuna-7B
* Largo representación binaria: 32
* Métrica de evaluación: AUC
* Función de activación: Tanh
* Función de pérdida: Binary cross-entropy

---

# Entrenamiento

El entrenamiento se compone de dos partes fundamentales:
* **Pre-training for Text-like Encoding**
* **LoRA Tuning**

---

## Pre-training for Text-like Encoding

En esta etapa, se entrena el módulo encargado de codificar la información colaborativa en un formato binario que pueda ser procesado por LLMs.

Se realiza de la siguiente manera:
1. **Obtención de embeddings colaborativos**: A partir de usuarios e ítems, se generan vectores latentes.
2. **Binarización**: Los vectores se convierten en secuencias binarias mediante una capa completamente conectada y la función de activación `sign`:
   $$h_u = \text{sign}(\sigma(W e_u + b))$$

---

3. **Optimización**: Se minimiza una loss function definida como binary cross-entropy para ajustar las predicciones con los datos de entrenamiento. Para superar la falta de derivada de la función `sign`, se usa el Straight-Through Estimator (STE) para aproximar gradientes. El STE consiste en no derivar aquellas funciones que no tengan una derivada definida en todo el espacio.

---

## LoRA

![bg right width:500px](./images/lora_image.png)

El método **Low-Rank Adaptation (LoRA)** adapta eficientemente los pesos de las capas densas de un modelo pre-entrenado, manteniendo los pesos originales congelados. Esto se hace optimizando matrices de baja dimensión que modelan los cambios durante la adaptación.

---

## Profundización LoRA

En LoRA, se realiza una adaptación de los pesos de capas densas mediante dos matrices clave, \(A\) y \(B\):

### Componentes del Diagrama
1. **Pesos pre-entrenados (\(W \in \mathbb{R}^{d \times d}\))**:
   - Son los pesos originales del modelo pre-entrenado, que se mantienen **congelados** durante el proceso de ajuste.

---

2. **Matriz \(A\)**:
   - Representa un cambio proyectado hacia una dimensionalidad baja.
   - Se inicializa con una distribución normal \(A \sim \mathcal{N}(0, \sigma^2)\) porque esto facilita una exploración más eficiente del espacio de soluciones en las primeras etapas del entrenamiento.
   - Es entrenable.


3. **Matriz \(B\)**:
   - Proyecta de vuelta al espacio de alta dimensionalidad.
   - Se inicializa como cero (\(B = 0\)) para que, al inicio del entrenamiento, los cambios sean mínimos y no afecten los pesos originales \(W\).
   - También es entrenable.

---

4. **Combinación de \(A\) y \(B\)**:
   - Las matrices \(A\) y \(B\) juntas permiten capturar los cambios requeridos en los pesos del modelo sin modificar los pesos pre-entrenados \(W\). El cambio total se calcula como:
     $$\Delta W = A \cdot B$$

5. **Entrenamiento**:
   - Solo se entrenan las matrices \(A\) y \(B\), lo que reduce significativamente la cantidad de parámetros que deben ser optimizados en comparación con un ajuste completo del modelo.
   - Esto es computacionalmente eficiente y reduce los requisitos de hardware, manteniendo el desempeño del modelo.

---

## LoRA Tuning

### Métodos
1. **Intuitive Tuning**:
   - Ajusta directamente las matrices de baja dimensión utilizando prompts enriquecidos con información colaborativa binarizada.

---

2. **Two-Step Tuning**:
   - **Paso 1**: Entrenamiento inicial con prompts que excluyen la información colaborativa.
   - **Paso 2**: Mayor ajuste con prompts completos que incluyen información colaborativa.
   - Este enfoque evita que el modelo dependa excesivamente de las representaciones binarias y mejora el balance con otras características del dataset.

---


## Inferencia

Considerando un prompt $p$, la predicción se puede formular como:

$$
\hat{y} = LLM_{\hat{\Phi} + \Phi'}(p)
$$

En donde $\hat{\Phi}$ denota los parámetros del LLM y $\Phi'$ los del módulo LoRA.


---

# Evaluación

<div style="display: flex; justify-content: space-around; margin-top: 20px;">
  
  <img src="./images/overall-performance.png" alt="Overall performance" style="width: 75%; height: auto;"/>

</div>

---
# Experimentos

## Rendimiento en warm / cold

![bg right width:500px](./images/warm-cold.png)

Los datos de prueba se dividen en "warm" o "cold" en base al número de interacciones. 


---

## La influencia de los componentes


![bg right width:600px](./images/benefits.png)


**BinMF**: Solo las representaciones binarias y MF
**BinLLM-TO**: Solo los textos en el prompt
**BinLLM-IO**: Solo los ids en el prompt
**BinLLM-IT**: Intuitive tuning (IT)


---

## La influencia de comprimir

![bg right height:270px](./images/compression.png)

Una limitación de las secuencias binarias es su largo.

Los autores evaluan el efecto de comprimirlas, convirtiendo cada ocho dígitos binarios en un número decima y utilizando el punto como carácter de separación.


---
# Conclusiones

* BinLLM es un método robusto para la tarea de predicción de click
* Dada una matriz de interacción "cold", los métodos LLMRec son los más adecuados
* BinLLM aprovecha tanto la información semántica como la colaborativa para destacarse sobre los otros métodos
* La compresión puede reducir el largo de la secuencia binaria manteniendo en gran medida el desempeño de BinLLM

---
# Referencias

1. Zheng, B., Hou, Y., Lu, H., Chen, Y., Zhao, W. X., Chen, M., & Wen, J.-R. (2024). *Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation*. arXiv. https://arxiv.org/abs/2311.09049
2. Hua, W., Xu, S., Ge, Y., & Zhang, Y. (2023). *How to Index Item IDs for Recommendation Foundation Models*. In *Proceedings of the Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region* (SIGIR-AP '23), 195–204. ACM. https://doi.org/10.1145/3624918.3625339

---
3. Zhang, Y., Feng, F., Zhang, J., Bao, K., Wang, Q., & He, X. (2024). *CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation*. arXiv. https://arxiv.org/abs/2310.19488
4. Li, X., Chen, C., Zhao, X., Zhang, Y., & Xing, C. (2023). *E4SRec: An Elegant Effective Efficient Extensible Solution of Large Language Models for Sequential Recommendation*. arXiv. https://arxiv.org/abs/2312.02443
5. Liao, J., Li, S., Yang, Z., Wu, J., Yuan, Y., Wang, X., & He, X. (2024). *LLaRA: Large Language-Recommendation Assistant*. arXiv. https://arxiv.org/abs/2312.02445

---

6. Tan, Q., Liu, N., Zhao, X., Yang, H., Zhou, J., & Hu, X. (2020). *Learning to Hash with Graph Neural Networks for Recommender Systems*. arXiv. https://arxiv.org/abs/2003.01917  

7. Savelka, J., Agarwal, A., An, M., Bogart, C., & Sakr, M. (2023). *Thrilled by Your Progress! Large Language Models (GPT-4) No Longer Struggle to Pass Assessments in Higher Education Programming Courses*. Proceedings of the 2023 ACM Conference on International Computing Education Research V.1. ACM. http://dx.doi.org/10.1145/3568813.3600142  

8. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). *Attention Is All You Need*. CoRR, abs/1706.03762. http://arxiv.org/abs/1706.03762


---

# Anexos

