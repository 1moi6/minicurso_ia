fundamentos_dct = [
  {
    'titulo':"Introdução", 'conteudo':'''<div align="justify">

O aprendizado de máquina, embora popularmente associado à construção empírica de modelos a partir de grandes volumes de dados, repousa sobre uma infraestrutura teórica robusta, cujos alicerces se encontram, sobretudo, na **matemática** e na **estatística**. Estas duas áreas, embora distintas em seus enfoques, atuam de forma complementar e indispensável para garantir que os modelos de machine learning não apenas funcionem computacionalmente, mas sejam **teoricamente justificáveis** e **estatisticamente confiáveis**.

Do lado da **matemática pura e aplicada**, o aprendizado de máquina é essencialmente um problema de **otimização de funções multivariadas**. A aprendizagem consiste em encontrar os parâmetros $\\theta$ que minimizam uma função de perda definida sobre um conjunto de dados:

$$
\min_{\\theta} \; \\frac{1}{m} \sum_{i=1}^m \mathcal{L}(f_{\\theta}(\mathbf{x}^{(i)}), y^{(i)}).
$$

Para resolver esse problema, recorre-se a ferramentas da **análise matemática**, como o cálculo diferencial e integral, e da **álgebra linear**, indispensável para manipulação de vetores, matrizes e transformações em espaços de alta dimensão. Além disso, a **teoria da otimização** fornece os fundamentos formais para garantir existência, unicidade, e caracterização dos mínimos — locais ou globais — da função de perda, enquanto a **análise numérica** disponibiliza algoritmos computacionalmente viáveis (como gradiente descendente, métodos de Newton, otimização estocástica e variantes adaptativas) que possibilitam encontrar boas soluções mesmo em situações onde não há formulação analítica fechada.

Entretanto, **a matemática por si só não é suficiente** para justificar por que um modelo treinado em um conjunto de dados consegue, de fato, generalizar para novos dados. É aqui que entra o papel fundamental da **estatística**, cuja principal contribuição é fornecer o aparato conceitual para compreender a **incerteza** e o **comportamento probabilístico** dos dados e das estimativas produzidas.

A estatística permite formalizar a noção de **generalização**: ainda que a otimização minimize o erro sobre o conjunto de treinamento, a verdadeira medida de sucesso de um modelo está em seu desempenho sobre dados **não vistos**, oriundos da mesma distribuição. Conceitos como **conjunto de teste**, **validação cruzada**, **viés** e **variância** são elementos centrais da teoria estatística que permitem quantificar essa capacidade de generalização. 

A clássica **decomposição do erro esperado** em:

$$
\\text{Erro total} = \\text{Viés}^2 + \\text{Variância} + \\text{Ruído irreducível}
$$

fornece uma lente analítica para entender as compensações entre modelos mais simples (alta tendência ao viés) e modelos mais complexos (alta variância), sendo a base para técnicas como regularização, seleção de modelos e escolha de hiperparâmetros.

Além disso, a estatística oferece **garantias teóricas**, como limites de generalização baseados em desigualdades de concentração (Hoeffding, McDiarmid) ou dimensões de complexidade (VC-dimension, Rademacher complexity), que justificam, sob determinadas condições, que **o desempenho em um conjunto de treino é um bom preditor do desempenho futuro** — fundamento este sem o qual o uso prático de modelos seria meramente especulativo.

Em síntese, a **matemática contribui com a estrutura e a resolução computacional do problema de aprendizado**, enquanto a **estatística garante a validade inferencial** do que é aprendido. Sem a primeira, não haveria modelo eficaz; sem a segunda, não haveria confiança de que o modelo funcionaria fora do treinamento.

Portanto, o aprendizado de máquina moderno deve ser compreendido não como um processo empírico e opaco, mas como um **procedimento matemático-estatístico rigoroso**, no qual **funções são otimizadas, incertezas são quantificadas, e previsões são sustentadas por princípios sólidos**. É esse entrelaçamento de teoria matemática com inferência estatística que confere ao aprendizado de máquina o seu caráter científico e o seu poder preditivo.

</div>'''
},
  {'titulo':"Conceitos Essenciais", 'conteudo':'''<div align="justify">

O **Aprendizado de Máquina (Machine Learning)** é uma subárea da inteligência artificial voltada para o desenvolvimento de algoritmos capazes de aprender a realizar tarefas a partir de dados. Em termos formais, segundo a definição clássica de Tom Mitchell, *“um programa de computador é dito aprender de uma experiência $E$ com relação a uma tarefa $T$ e uma medida de desempenho $P$, se seu desempenho em $T$, medido por $P$, melhora com a experiência $E$”*.

No **aprendizado supervisionado**, uma das abordagens centrais do Machine Learning, o modelo tem acesso a um **conjunto de dados rotulado**, ou seja, para cada exemplo de entrada $\mathbf{x} \in \mathbb{R}^n$, está associado um rótulo ou valor de saída $y$. O objetivo do algoritmo é aprender uma função $f: \mathbb{R}^n \\to \mathcal{Y}$ que generalize bem para novos dados, isto é, que seja capaz de prever corretamente $y$ a partir de uma nova $\mathbf{x}$ não observada durante o treinamento.

Esse processo de "aprender com dados" envolve três componentes fundamentais:

- **A tarefa $T$**: por exemplo, classificar imagens, prever preços ou diagnosticar doenças.
- **A medida de desempenho $P$**: acurácia, erro quadrático médio, perda logística, entre outras.
- **A experiência $E$**: representada por um conjunto de dados $D = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^m$, geralmente dividida em subconjuntos de *treinamento*, *validação* e *teste*.

O treinamento consiste em ajustar os **parâmetros** do modelo para minimizar uma função de perda em relação aos exemplos do conjunto de treinamento. A partir disso, espera-se que o modelo consiga **generalizar**, ou seja, apresentar bom desempenho em exemplos ainda não vistos, pertencentes ao mesmo processo gerador de dados (assumido *i.i.d.* — *independent and identically distributed*).

No contexto supervisionado, distinguem-se duas categorias fundamentais de problemas:

- **Classificação**: quando a variável alvo $y$ assume valores discretos, geralmente representando categorias. Exemplos incluem a identificação de e-mails como "spam" ou "não-spam", a detecção de tumores em imagens médicas ou a classificação de espécies de plantas.

- **Regressão**: quando a variável alvo $y$ é contínua. Exemplos clássicos são a previsão de valores de imóveis com base em localização, área e número de quartos, ou a estimação da temperatura em função de variáveis meteorológicas.

Essas tarefas exigem técnicas distintas de modelagem, mas compartilham a estrutura essencial do aprendizado supervisionado. Um aspecto central no desenvolvimento de bons modelos é o **compromisso entre viés e variância**: modelos com **baixa capacidade** podem não capturar a complexidade dos dados (*subajuste*), enquanto modelos com **alta capacidade** podem se ajustar demais ao conjunto de treinamento (*sobreajuste*), perdendo capacidade de generalização.

Outro conceito relevante é o de **conjunto de validação**, usado para ajustar hiperparâmetros e evitar sobreajuste ao conjunto de treinamento. Após essa etapa, um conjunto de teste independente é utilizado para estimar a performance final do modelo.

</div>'''},
{'titulo':"Treinamento", 'conteudo':'''<div align="justify">

O **processo de treinamento** em aprendizado supervisionado visa ajustar os **parâmetros internos** de um modelo de modo que ele seja capaz de aprender a partir dos dados. Dado um conjunto de treinamento $D = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^m$, o objetivo do aprendizado é encontrar uma função $f_{\\theta}$, parametrizada por $\\theta$, que minimize uma função de custo (ou perda), que quantifica o erro entre a predição do modelo $f_{\\theta}(\mathbf{x}^{(i)})$ e o rótulo verdadeiro $y^{(i)}$.

Matematicamente, queremos resolver o seguinte problema de minimização:

$$
\min_{\\theta} \; \\frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(f_{\\theta}(\mathbf{x}^{(i)}), y^{(i)}),
$$

onde $\mathcal{L}$ é uma função de perda apropriada. Exemplos comuns incluem:
- **Erro quadrático médio (MSE)** – em regressão:
  $$
  \mathcal{L}_{\\text{MSE}}(f(\mathbf{x}), y) = \\frac{1}{2}(f(\mathbf{x}) - y)^2
  $$

- **Entropia cruzada (Cross-Entropy)** – em classificação binária:
  $$
  \mathcal{L}_{\\text{CE}}(f(\mathbf{x}), y) = - \left[ y \log f(\mathbf{x}) + (1 - y) \log (1 - f(\mathbf{x})) \\right]
  $$
O **gradiente descendente** atualiza os parâmetros $\\theta$ iterativamente pela regra:

$$
\\theta \leftarrow \\theta - \\frac{\eta}{m} \sum_{i=1}^{m} \\nabla_{\\theta} \mathcal{L}(f_{\\theta}(\mathbf{x}^{(i)}), y^{(i)}),
$$

onde:

- $\eta > 0$ é a **taxa de aprendizado** (learning rate),
- $\\nabla_{\\theta} \mathcal{L}$ é o **gradiente da função de perda** em relação aos parâmetros $\\theta$.

Essa fórmula é completamente **genérica** e se aplica a qualquer modelo $f_{\\theta}$ e função de perda $\mathcal{L}$. O formato específico do gradiente depende da estrutura de $f_{\\theta}$ e da escolha de $\mathcal{L}$.


</div>

---

#### 2.3. Regressão Linear e Logística

<strong>2.3.1 Regressão Linear e Logística</strong> 

<div align="justify">

A **regressão linear** é um dos modelos mais simples e fundamentais do aprendizado supervisionado. Seu objetivo é modelar uma relação linear entre um vetor de entrada $\mathbf{x} \in \mathbb{R}^n$ e uma variável de saída contínua $y \in \mathbb{R}$, assumindo que:

$$
y \\approx \hat{y} = f_{\\theta}(\mathbf{x}) = \mathbf{w}^\\top \mathbf{x} + b,
$$

onde:

- $\mathbf{w} \in \mathbb{R}^n$ é o vetor de pesos (ou coeficientes) que determina a contribuição de cada variável de entrada;
- $b \in \mathbb{R}$ é o termo de interceptação (viés);
- $\hat{y}$ é a saída predita pelo modelo para a entrada $\mathbf{x}$.

O objetivo do treinamento é encontrar os parâmetros $(\mathbf{w}, b)$ que minimizem o **erro quadrático médio** (MSE — *mean squared error*) entre os valores previstos $\hat{y}^{(i)}$ e os valores reais $y^{(i)}$ sobre um conjunto de treinamento com $m$ exemplos:

$$
\mathcal{L}(\mathbf{w}, b) = \\frac{1}{m} \sum_{i=1}^m \left( \mathbf{w}^\\top \mathbf{x}^{(i)} + b - y^{(i)} \\right)^2.
$$

Essa função de perda é convexa, o que permite encontrar uma solução analítica através do chamado **método dos mínimos quadrados**. Para isso, é comum reescrever o modelo em notação matricial. Seja:

- $\mathbf{X} \in \mathbb{R}^{m \\times n}$ a matriz de entrada, onde cada linha é um vetor $\mathbf{x}^{(i)}$;
- $\mathbf{y} \in \mathbb{R}^m$ o vetor de saídas reais;
- $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b\mathbf{1}$ o vetor de predições.

A solução que minimiza a função de perda, ignorando momentaneamente o viés $b$ (ou absorvendo-o em $\mathbf{w}$ com uma variável adicional constante), é dada por:

$$
\mathbf{w}^* = (\mathbf{X}^\\top \mathbf{X})^{-1} \mathbf{X}^\\top \mathbf{y},
$$

desde que $\mathbf{X}^\\top \mathbf{X}$ seja invertível. Esse resultado é conhecido como **equação normal** da regressão linear.

</div>'''},
{
  'titulo':"Capacidade do Modelo, Subajuste e Sobreajuste", 'conteudo':'''<div align="justify">

Ao treinar um modelo de aprendizado de máquina, o objetivo não é apenas ajustar-se bem aos dados de treinamento, mas sim **generalizar** para novos dados nunca vistos. Compreender os conceitos de **capacidade do modelo**, **subajuste** (*underfitting*) e **sobreajuste** (*overfitting*) é essencial para alcançar esse objetivo.


**Capacidade** refere-se à **complexidade funcional** que um modelo é capaz de representar. Um modelo com baixa capacidade pode representar apenas funções simples (por exemplo, funções lineares), enquanto um modelo com alta capacidade pode representar funções complexas e altamente não lineares.

De modo geral:

- **Baixa capacidade**: o modelo não é capaz de capturar padrões importantes dos dados;
- **Alta capacidade**: o modelo é flexível o suficiente para se ajustar até mesmo ao ruído presente nos dados de treinamento.

É comum, no processo de modelagem supervisionada, dividir o conjunto total de dados em três subconjuntos: o **conjunto de treinamento** ($\mathcal{D}_{\\text{train}}$), utilizado para ajustar os parâmetros internos do modelo; o **conjunto de validação** ($\\mathcal{D}_{\\text{val}}$), usado para ajustar hiperparâmetros e monitorar o desempenho do modelo em dados não vistos durante o processo de otimização; e o **conjunto de teste** ($\mathcal{D}_{\\text{test}}$), reservado exclusivamente para estimar o desempenho final do modelo, refletindo sua capacidade de generalização.

A partir desses conjuntos, podemos definir medidas quantitativas para avaliar o aprendizado. O **erro de treinamento** é a perda computada sobre $\mathcal{D}_{\\text{train}}$ e indica quão bem o modelo se ajusta aos dados que foram usados para sua construção. Já o **erro de validação** (calculado sobre $\mathcal{D}_{\\text{val}}$) permite monitorar o ajuste a dados novos ao longo do processo de treino, ajudando a evitar sobreajuste. Por fim, o **erro de teste**, medido em $\mathcal{D}_{\\text{test}}$, representa a estimativa final e imparcial da performance do modelo em situações reais.

Comparar essas três métricas permite diagnosticar se o modelo sofre de **underfitting** (quando o erro de treino já é alto) ou **overfitting** (quando o erro de treino é baixo, mas o erro de validação ou teste é alto), orientando decisões sobre capacidade do modelo, regularização ou necessidade de mais dados.


Nesse contexto, **Underfitting** (subajuste) ocorre quando o modelo tem **capacidade insuficiente** para capturar a estrutura dos dados. Isso se traduz em **alto erro no treinamento e também no teste**.
Já o **Overfitting** (sobreajuste) acontece quando o modelo tem **capacidade excessiva** e aprende não apenas os padrões verdadeiros dos dados, mas também **o ruído e as flutuações específicas** do conjunto de treinamento. Isso resulta em **baixo erro no treinamento**, mas **alto erro no teste**.


Podemos ilustrar esse comportamento com a **curva de erro** em função da capacidade do modelo:

<p align="center">
<img src="https://raw.githubusercontent.com/1moi6/minicurso_ia/refs/heads/main/assets/images/erro_decomposicao_teorico.png" width="400"/>
</p>

O ponto ótimo de capacidade está geralmente associado a um equilíbrio entre **viés** (erro sistemático) e **variância** (sensibilidade a pequenas variações nos dados).


Matematicamente, esse fenômeno pode ser estudado a partir da decomposição do erro esperado. 
No contexto da aprendizagem supervisionada, a capacidade de um modelo generalizar bem para dados nunca vistos depende de como ele equilibra três componentes fundamentais do erro de previsão: o **viés**, a **variância** e o **ruído irredutível**. Essa análise é formalizada pela decomposição do **erro quadrático médio esperado** (Mean Squared Error – MSE), a qual permite interpretar separadamente os fatores que contribuem para o desempenho de um preditor em um ponto fixo de entrada $\mathbf{x}$. A fórmula da decomposição é dada por:

$$
\mathbb{E}_{\mathcal{D}, y} \left[ (y - \hat{f}(\mathbf{x}; \mathcal{D}))^2 \\right] = \left( \\text{Bias}[\hat{f}(\mathbf{x})] \\right)^2 + \\text{Var}[\hat{f}(\mathbf{x})] + \sigma^2.
$$

A expectativa do lado esquerdo é calculada sobre dois níveis de aleatoriedade: o conjunto de dados de treinamento $\mathcal{D}$ e o ruído na variável resposta $y$. O termo $\hat{f}(\mathbf{x}; \mathcal{D})$ representa a predição do modelo treinado sobre um conjunto $\mathcal{D}$, avaliado em um ponto fixo $\mathbf{x}$ — tipicamente, um ponto fora da amostra de treinamento.

O **viés** mede o desvio sistemático entre a média das predições do modelo e a verdadeira função $f(\mathbf{x})$. Formalmente, é definido por:

$$
\\text{Bias}[\hat{f}(\mathbf{x})] = \mathbb{E}_{\mathcal{D}}[\hat{f}(\mathbf{x}; \mathcal{D})] - f(\mathbf{x}).
$$

Um viés elevado surge quando o modelo é demasiadamente rígido ou simples, de modo que não consegue capturar a complexidade da função verdadeira que governa os dados. Por exemplo, tentar ajustar um padrão altamente não linear com um modelo linear resultará em uma predição sistematicamente equivocada, independentemente do conjunto de dados usado. Essa característica está associada ao fenômeno conhecido como **underfitting**, no qual o modelo falha em aprender até mesmo os principais padrões da estrutura dos dados.

Por sua vez, a **variância** reflete o grau de instabilidade do modelo ao longo de diferentes amostras de treinamento. Sua expressão matemática é:

$$
\\text{Var}[\hat{f}(\mathbf{x})] = \mathbb{E}_{\mathcal{D}} \left[ \left( \hat{f}(\mathbf{x}; \mathcal{D}) - \mathbb{E}_{\mathcal{D}}[\hat{f}(\mathbf{x}; \mathcal{D})] \\right)^2 \\right].
$$

Esse termo quantifica o quanto as predições variam para um ponto fixo $\mathbf{x}$ quando o modelo é treinado em diferentes amostras $\mathcal{D}$ extraídas da mesma distribuição de dados. Portanto, ele **não mede a sensibilidade do modelo a entradas diferentes**, mas sim **à escolha da amostra de treinamento**. Quando a variância é alta, o modelo tende a se ajustar ao ruído específico dos dados de treino, produzindo predições inconsistentes em novos dados — uma situação típica de **sobreajuste (overfitting)**. Esse tipo de modelo pode ter desempenho quase perfeito no treinamento, mas fracassar ao generalizar para dados reais.

O terceiro termo da decomposição é o **erro irredutível**, representado por $\sigma^2$, que corresponde à variância intrínseca dos dados em torno da verdadeira função $f(\mathbf{x})$. Esse erro resulta de fatores imprevisíveis, variabilidade natural ou medições imprecisas que afetam a variável resposta $y$. Independentemente do modelo utilizado, essa incerteza não pode ser eliminada — ela representa o limite teórico inferior para o erro de qualquer estimativa.

A decomposição do erro esperado permite, assim, compreender a **compensação entre viés e variância**. Modelos simples tendem a apresentar baixo risco de variância, mas alto viés, enquanto modelos complexos reduzem o viés, mas amplificam a variância. O desafio central do aprendizado de máquina é encontrar um ponto de equilíbrio entre essas forças opostas, minimizando o erro total. Esse equilíbrio é influenciado não apenas pela escolha do modelo, mas também pelo tamanho e pela diversidade do conjunto de dados, bem como pelas técnicas de regularização e validação utilizadas.


Portanto, escolher um modelo adequado envolve:

1. Ajustar a **capacidade do modelo** ao tamanho e à complexidade dos dados;
2. Utilizar técnicas como **validação cruzada**, **regularização** e **aumento de dados** para evitar o sobreajuste;
3. Monitorar a **curva de aprendizagem** para detectar sinais precoces de overfitting ou underfitting.


A prática de dividir um conjunto de dados em subconjuntos de **treinamento**, **validação** e **teste** fundamenta-se em princípios centrais da *estatística inferencial*. Em essência, essa estrutura tem por objetivo isolar os efeitos do *viés*, da *variância* e do *ruído aleatório* no processo de modelagem, permitindo estimativas não tendenciosas do erro de generalização e diagnósticos empíricos sobre o comportamento preditivo do modelo.

O **conjunto de treinamento** desempenha o papel de fornecer uma amostra sobre a qual o modelo ajusta seus parâmetros internos. Do ponto de vista estatístico, este processo equivale a estimar uma função preditiva com base em uma amostra observacional. No entanto, qualquer avaliação realizada nesse mesmo conjunto estará sujeita a *viés otimista*, uma vez que os parâmetros foram calibrados para minimizar o erro nessa amostra específica. Por isso, o erro de treinamento é geralmente *subestimado* e, sozinho, não permite aferir a capacidade de generalização.

O **conjunto de validação** funciona como uma amostra independente e não utilizada no ajuste direto dos parâmetros do modelo. Seu propósito é fornecer uma *estimativa imparcial do desempenho do modelo em dados fora da amostra*, embora ainda dentro do mesmo processo de construção. Isso permite ao analista avaliar a *instabilidade do modelo frente a variações nos dados*, que está relacionada ao componente de *variância*. Um modelo que apresenta erro de validação muito superior ao de treinamento sugere que está capturando ruídos específicos da amostra de treino — caracterizando *overfitting*. Por outro lado, se o erro permanece elevado tanto na validação quanto no treino, indica que o modelo possui *viés elevado*, incapaz de representar adequadamente os padrões subjacentes aos dados.

Já o **conjunto de teste** é reservado para uma avaliação estatisticamente rigorosa do modelo final. Ele simula a aplicação do modelo em uma nova amostra, retirada da mesma distribuição subjacente, mas completamente *alheia ao processo de treinamento e seleção de modelo*. Esse isolamento é crucial para que o erro de teste represente uma *estimativa não enviesada do erro esperado verdadeiro*, o qual está associado à performance futura do modelo em contexto de produção. Em termos inferenciais, o conjunto de teste opera como um “conjunto de validação externa”, fornecendo uma medida da capacidade preditiva generalizável do modelo.

Essa divisão também possibilita uma *avaliação empírica indireta dos componentes da decomposição do erro*: enquanto o erro de treino reflete a capacidade do modelo de memorizar, o erro de validação informa sobre sua habilidade de generalizar sob os mesmos pressupostos estatísticos, e o erro de teste avalia a robustez dessa generalização sem qualquer influência de ajustes internos. A comparação entre esses erros, portanto, oferece uma leitura prática da interação entre *viés*, *variância* e *ruído* — mesmo quando tais quantidades não são diretamente observáveis.

Finalmente, vale destacar que essas práticas se alinham aos princípios de *planejamento experimental* e *validação cruzada* amplamente utilizados em estatística. Ao garantir que as inferências sejam feitas sobre dados independentes, evita-se a contaminação por dependências induzidas pelo treinamento, assegurando que os modelos produzidos não apenas se ajustem aos dados passados, mas também se sustentem *como preditores confiáveis para o futuro*.

---
</div>'''
},
{
  'titulo':'Exemplo experimental', 
  'conteudo':'''<div align="justify">
  
  Podemos visualizar empiricamente o comportamento dos erros de generalização de um modelo de regressão supervisionada por meio de um experimento que simula diferentes níveis de complexidade indutiva. Para isso, consideramos como função alvo a ser aprendida a expressão não linear e suave:

$$
f(x) = 2x + \cos(4\pi x), \quad x \in [0,1].
$$

O objetivo é construir modelos aproximadores dessa função a partir de amostras com ruído e avaliar, para diferentes níveis de complexidade, os erros de viés, variância e ruído irreducível que compõem o erro quadrático médio esperado.

Para aproximar $ f(x) $, usamos uma família de modelos polinomiais da forma:

$$
f_\\alpha(x) = \sum_{k=0}^{d} a_k x^k,
$$

onde os coeficientes $ a_k $ são aprendidos a partir de dados sintéticos gerados com ruído aditivo gaussiano. A complexidade do modelo é controlada por meio de uma **regularização suave** imposta diretamente sobre os coeficientes, com o intuito de penalizar fortemente termos de alta ordem. Essa regularização é implementada por uma estrutura de decaimento exponencial, penalizando coeficientes segundo:

$$
\\text{Penalidade:} \quad \sum_{k=0}^{d} \left( \\frac{a_k}{\\alpha^k} \\right)^2,
$$

onde $ \\alpha \in (0,1] $ é o **parâmetro de regularização** que controla a complexidade do modelo, e $ M $ é um hiperparâmetro multiplicativo fixo. Valores menores de $ \\alpha $ impõem maior decaimento e restrição à magnitude dos coeficientes de ordem elevada, resultando em modelos mais suaves e de baixa capacidade. Por outro lado, valores de $ \\alpha $ mais próximos de 1 permitem que o modelo expresse maior variação e detalhes locais, o que pode levar ao sobreajuste.

A cada iteração do experimento, geramos um conjunto de dados de treinamento $ \{(x_i, y_i)\}_{i=1}^n $, com $ x_i \sim \mathcal{U}(0,1) $ e $ y_i = f(x_i) + \\varepsilon_i $, onde $ \\varepsilon_i \sim \mathcal{N}(0, \sigma^2) $. O modelo $ f_\\alpha $ é então ajustado minimizando a seguinte função de perda regularizada:

$$
\min_{a_0,\ldots,a_d} \left\{ \\frac{1}{n} \sum_{i=1}^{n} \left(y_i - \sum_{k=0}^{d} a_k x_i^k \\right)^2 + \lambda \sum_{k=0}^{d} \left( \\frac{a_k}{\\alpha^k} \\right)^2 \\right\},
$$

onde $ \lambda $ é um parâmetro que controla a força da penalização. Essa minimização é realizada para diferentes valores de $ \\alpha $, mantendo fixos o número de amostras, a variância do ruído, e o grau máximo do polinômio. Para cada $ \\alpha $, o experimento é repetido sobre diversos conjuntos de dados distintos, e as predições do modelo são avaliadas em um conjunto fixo de pontos $ \{x^{(j)}\}_{j=1}^{m} $ não vistos no treinamento. A partir dessas predições, calcula-se o viés quadrático como a média do quadrado da diferença entre a predição média e o valor verdadeiro de $ f(x) $, a variância como a variância empírica das predições ao longo dos datasets, e o ruído como a variância do erro aditivo conhecido.

A variação do parâmetro $ \\alpha $ permite então observar como a capacidade do modelo influencia o erro de generalização. Espera-se que, para valores muito pequenos de $ \\alpha $, o modelo tenha baixa variância, mas alto viés, devido à incapacidade de capturar adequadamente a oscilação da função alvo — caracterizando um regime de underfitting. À medida que $ \\alpha $ cresce, o viés tende a diminuir, mas a variância pode aumentar sensivelmente, indicando sobreajuste ao ruído dos dados. O valor ótimo de $ \\alpha $ é aquele que minimiza a soma total dos erros, equilibrando a capacidade de generalização com a fidelidade à função verdadeira.

Esse experimento ilustra de maneira clara o papel dos hiperparâmetros na regulação da complexidade de modelos de aprendizado e fornece uma ferramenta empírica útil para compreender a compensação entre viés e variância, que está no cerne da teoria estatística do aprendizado de máquina.


  </div>
  '''

},
{'titulo':"Métricas de Avaliação", 'conteudo':'''<div align="justify">

Um aspecto sutil, mas conceitualmente importante no estudo de Machine Learning, é a distinção entre a **função de perda (loss function)** e as **métricas de avaliação**. Embora ambas sirvam para quantificar o desempenho de modelos, elas têm **papéis e objetivos diferentes** dentro do processo de modelagem.


A **função de perda** é um **objeto matemático central** no treinamento do modelo. Ela mede, para cada par de entrada $(\mathbf{x}, y)$, o grau de erro entre a saída prevista pelo modelo $f_{\\theta}(\mathbf{x})$ e a saída verdadeira $y$.

A função de perda deve ser **diferenciável**, pois será usada para calcular gradientes durante o treinamento via **algoritmos de otimização**, como o **gradiente descendente**.


Já as **métricas de avaliação** são utilizadas **após o treinamento**, para julgar a **eficácia do modelo** em dados não vistos, muitas vezes com propósitos práticos ou de comunicação.

Diferentemente das funções de perda, as métricas **não precisam ser diferenciáveis** e muitas vezes **operam sobre conjuntos inteiros de previsões**, como no caso da acurácia, precisão ou F1-score.

Um modelo de classificação binária pode ser treinado usando **entropia cruzada** como função de perda (que opera sobre probabilidades), mas avaliado em termos de **acurácia**, que exige que as probabilidades sejam transformadas em decisões (por exemplo, classificando como "positivo" se $f(\mathbf{x}) > 0{,}5$).

<div align="center">

| Aspecto              | Função de Perda                        | Métricas de Avaliação                |
|----------------------|----------------------------------------|--------------------------------------|
| Quando é usada       | Durante o treinamento                  | Após o treinamento                   |
| Objetivo             | Otimizar os parâmetros do modelo       | Avaliar o desempenho preditivo       |
| Exige derivadas?     | Sim (para gradiente descendente)       | Não (pode ser baseada em contagens)  |
| Exemplos             | MSE, Cross-Entropy                     | Acurácia, Precisão, Recall, F1-score |

</div>
---

</div>

<div align="justify">

Em tarefas de **classificação supervisionada**, é fundamental avaliar a qualidade das predições do modelo de maneira quantitativa. Para isso, utiliza-se a **matriz de confusão**, a partir da qual derivam-se diversas métricas, cada uma enfatizando aspectos distintos da performance do classificador.

<div align="center">

|                       | Previsto Positivo | Previsto Negativo |
|-----------------------|-------------------|-------------------|
| **Real Positivo**     | TP (Verdadeiro Positivo)  | FN (Falso Negativo)   |
| **Real Negativo**     | FP (Falso Positivo)       | TN (Verdadeiro Negativo) |

</div>

- **TP (True Positive)**: o modelo previu positivo, e a classe verdadeira era positiva;
- **TN (True Negative)**: o modelo previu negativo, e a classe verdadeira era negativa;
- **FP (False Positive)**: o modelo previu positivo, mas a classe verdadeira era negativa (erro tipo I);
- **FN (False Negative)**: o modelo previu negativo, mas a classe verdadeira era positiva (erro tipo II).

A **acurácia** mede a proporção de acertos (positivos e negativos) em relação ao total de previsões:

$$
\\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}
$$

É uma métrica intuitiva e fácil de interpretar. No entanto, **pode ser enganosa em conjuntos de dados desbalanceados** — isto é, quando uma das classes é muito mais frequente que a outra.

**Exemplo:**  
Suponha um classificador com os seguintes valores:

- TP = 70  
- TN = 20  
- FP = 5  
- FN = 5

Então:

$$
\\text{Accuracy} = \\frac{70 + 20}{70 + 20 + 5 + 5} = \\frac{90}{100} = 0{,}90
$$

O modelo acertou 90% das previsões.

A **precisão** (ou valor preditivo positivo) mede a **confiabilidade das predições positivas**:

$$
\\text{Precision} = \\frac{TP}{TP + FP}
$$

Ela responde à pergunta: *Dentre as instâncias classificadas como positivas pelo modelo, quantas realmente o são?*

É crucial em aplicações onde **falsos positivos têm alto custo** — por exemplo, um diagnóstico médico que identifique uma doença grave em alguém saudável.

**Exemplo (continuação):**

$$
\\text{Precision} = \\frac{70}{70 + 5} = \\frac{70}{75} \\approx 0{,}933
$$


O **recall** mede a **capacidade do modelo de identificar os positivos reais**:

$$
\\text{Recall} = \\frac{TP}{TP + FN}
$$

Responde à pergunta: *Dentre os casos realmente positivos, quantos foram detectados?*

É vital quando **falsos negativos são perigosos**, como em sistemas de segurança ou exames médicos que precisam detectar todos os casos críticos.

**Exemplo (continuação):**

$$
\\text{Recall} = \\frac{70}{70 + 5} = \\frac{70}{75} \\approx 0{,}933
$$


O **F1-score** é a **média harmônica** entre precisão e recall. Ele equilibra as duas métricas, penalizando fortemente desequilíbrios entre elas:

$$
F1 = 2 \cdot \\frac{\text{Precision} \cdot \\text{Recall}}{\text{Precision} + \\text{Recall}}
$$

É particularmente útil em **cenários com classes desbalanceadas**, onde uma única métrica (como a acurácia) pode ocultar falhas importantes.

**Exemplo (continuação):**

Com precisão = recall = 0,933, temos:

$$
F1 = 2 \cdot \\frac{0{,}933 \cdot 0{,}933}{0{,}933 + 0{,}933} = 2 \cdot \\frac{0{,}87}{1{,}866} \\approx 0{,}933
$$


- A **acurácia** é uma boa métrica quando as classes estão balanceadas;
- A **precisão** é preferida quando os **falsos positivos** são mais prejudiciais;
- O **recall** é importante quando os **falsos negativos** são críticos;
- O **F1-score** é útil quando há **compensação entre precisão e recall**.

</div>'''},
{'titulo':"Regressões Linear e Logística", 'conteudo':'''<strong>2.3.1 Regressão Linear e Logística</strong> 

<div align="justify">

A **regressão linear** é um dos modelos mais simples e fundamentais do aprendizado supervisionado. Seu objetivo é modelar uma relação linear entre um vetor de entrada $\mathbf{x} \in \mathbb{R}^n$ e uma variável de saída contínua $y \in \mathbb{R}$, assumindo que:

$$
y \\approx \hat{y} = f_{\\theta}(\mathbf{x}) = \mathbf{w}^\\top \mathbf{x} + b,
$$

onde:

- $\mathbf{w} \in \mathbb{R}^n$ é o vetor de pesos (ou coeficientes) que determina a contribuição de cada variável de entrada;
- $b \in \mathbb{R}$ é o termo de interceptação (viés);
- $\hat{y}$ é a saída predita pelo modelo para a entrada $\mathbf{x}$.

O objetivo do treinamento é encontrar os parâmetros $(\mathbf{w}, b)$ que minimizem o **erro quadrático médio** (MSE — *mean squared error*) entre os valores previstos $\hat{y}^{(i)}$ e os valores reais $y^{(i)}$ sobre um conjunto de treinamento com $m$ exemplos:

$$
\mathcal{L}(\mathbf{w}, b) = \\frac{1}{m} \sum_{i=1}^m \left( \mathbf{w}^\\top \mathbf{x}^{(i)} + b - y^{(i)} \\right)^2.
$$

Essa função de perda é convexa, o que permite encontrar uma solução analítica através do chamado **método dos mínimos quadrados**. Para isso, é comum reescrever o modelo em notação matricial. Seja:

- $\mathbf{X} \in \mathbb{R}^{m \\times n}$ a matriz de entrada, onde cada linha é um vetor $\mathbf{x}^{(i)}$;
- $\mathbf{y} \in \mathbb{R}^m$ o vetor de saídas reais;
- $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b\mathbf{1}$ o vetor de predições.

A solução que minimiza a função de perda, ignorando momentaneamente o viés $b$ (ou absorvendo-o em $\mathbf{w}$ com uma variável adicional constante), é dada por:

$$
\mathbf{w}^* = (\mathbf{X}^\\top \mathbf{X})^{-1} \mathbf{X}^\\top \mathbf{y},
$$

desde que $\mathbf{X}^\\top \mathbf{X}$ seja invertível. Esse resultado é conhecido como **equação normal** da regressão linear.

</div>

---

<strong>2.3.2 Regressão Logística</strong>

<div align="justify">

A **regressão logística** é um modelo estatístico utilizado para problemas de **classificação binária**, isto é, quando a variável de saída $y$ assume apenas dois valores, geralmente codificados como $y \in \{0, 1\}$. Embora tenha a palavra "regressão" no nome, seu propósito é **classificar** observações, e não estimar valores contínuos.

A principal ideia é transformar a saída de uma regressão linear — que pode assumir qualquer valor real — em uma **probabilidade** no intervalo $(0, 1)$, utilizando a **função logística**, também conhecida como **função sigmoide**:

$$
\sigma(z) = \\frac{1}{1 + e^{-z}}.
$$

No modelo de regressão logística, a probabilidade de que $y = 1$ dado $\mathbf{x}$ é modelada como:

$$
P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\\top \mathbf{x} + b) = \\frac{1}{1 + e^{-(\mathbf{w}^\\top \mathbf{x} + b)}}.
$$

De forma equivalente:

$$
P(y = 0 \mid \mathbf{x}) = 1 - \sigma(\mathbf{w}^\\top \mathbf{x} + b).
$$

A saída do modelo $\hat{y} = \sigma(\mathbf{w}^\\top \mathbf{x} + b)$ pode ser interpretada como a **probabilidade estimada** de que a observação pertença à classe 1. A decisão final é feita com base em um limiar (threshold), geralmente 0.5:

$$
\hat{y} =
\\begin{cases}
1, & \\text{se } \sigma(\mathbf{w}^\\top \mathbf{x} + b) \geq 0.5 \\\ 
0, & \\text{caso contrário}
\end{cases}
$$

Para treinar o modelo, utiliza-se a **função de perda logarítmica** (também chamada de **log-loss** ou **entropia cruzada**) definida por:

$$
\mathcal{L}(\mathbf{w}, b) = - \\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \\right],
$$

onde $\hat{y}^{(i)} = \sigma(\mathbf{w}^\\top \mathbf{x}^{(i)} + b)$.

Essa função é convexa e pode ser minimizada por métodos iterativos como o **gradiente descendente**, sendo a base para diversos modelos mais sofisticados em aprendizado de máquina.

Como mencionado anteriormente, para usar o algoritmo de **gradiente descendente** é necessário obter as derivadas parciais de $\mathcal{L}$ em relação a $\mathbf{w}$ e $b$.
Nesse caso, a derivada da função de perda em relação a um peso $w_j$ é:

$$
\\frac{\partial \mathcal{L}}{\partial w_j} = \\frac{1}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \\right) x_j^{(i)}.
$$

De forma vetorial, o gradiente em relação a $\mathbf{w}$ é:

$$
\\nabla_{\mathbf{w}} \mathcal{L} = \\frac{1}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \\right) \mathbf{x}^{(i)}.
$$

Já a derivada da perda em relação ao viés $b$ é:

$$
\\frac{\partial \mathcal{L}}{\partial b} = \\frac{1}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \\right).
$$


Assim, considerando uma taxa de aprendizado $\eta > 0$, os parâmetros são atualizados a cada iteração por:

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \\nabla_{\mathbf{w}} \mathcal{L},
\quad
b \leftarrow b - \eta \\frac{\partial \mathcal{L}}{\partial b}.
$$


Nessa atualização:
- O termo $(\hat{y}^{(i)} - y^{(i)})$ representa o **erro de predição**;
- O vetor $\mathbf{x}^{(i)}$ serve como um fator de **sensibilidade** para cada entrada;
- O gradiente aponta na direção de maior crescimento da função de perda; o gradiente descendente move os parâmetros na direção oposta, reduzindo a perda.

Esse procedimento é repetido iterativamente até convergência, geralmente monitorando a perda ou o erro de validação para interromper o processo.

</div>'''}
]


resumo='''### Resumo

<div align="justify">
O presente material visa explorar a interseção fundamental entre <strong>Inteligência Artificial (IA)</strong>, <strong>Machine Learning (ML)</strong> e <strong>Matemática</strong>, com um foco particular nas <strong>Redes Neurais Artificiais (RNAs)</strong> aplicadas a problemas de <strong>classificação</strong>.

A compreensão aprofundada da <strong>fundamentação matemática</strong> é crucial para estudantes da área, permitindo-lhes não apenas <strong>utilizar</strong>, mas também <strong>inovar</strong> nessas tecnologias.

Ao longo do texto, <strong>exemplos e aplicações em biomatemática</strong> serão utilizados como fio condutor, ilustrando a <strong>relevância prática</strong> desses conceitos em domínios como <strong>epidemiologia</strong> e <strong>dinâmica populacional</strong>.
</div>

### Objetivos

<div align="justify">
<ul>
  <li>Apresentar os fundamentos teóricos da <strong>Inteligência Artificial</strong> e do <strong>Aprendizado de Máquina</strong>.</li>
  <li>Discutir a estrutura e funcionamento das <strong>Redes Neurais Artificiais</strong>, com ênfase no processo de <strong>classificação supervisionada</strong>.</li>
  <li>Explorar a <strong>matemática por trás</strong> do treinamento de redes neurais, incluindo funções de ativação, derivadas e retropropagação.</li>
  <li>Aplicar os conceitos em <strong>exemplos inspirados em problemas reais</strong> da biomatemática.</li>
</ul>
</div>

### Estrutura do Material

<div align="justify">
<ol>
  <li><strong>Fundamentos de IA e Machine Learning</strong><br>
  Introdução aos conceitos essenciais, tipos de aprendizado e motivação histórica.</li>

  <li><strong>Neurônios Artificiais e Arquiteturas de Rede</strong><br>
  Definição formal de um neurônio artificial, camadas, funções de ativação e arquiteturas comuns.</li>

  <li><strong>Treinamento e Otimização</strong><br>
  Abordagem detalhada sobre o algoritmo de retropropagação, função de custo, gradientes e convergência.</li>

  <li><strong>Aplicações em Biologia Matemática</strong><br>
  Exemplos práticos com foco em <strong>epidemiologia</strong>, <strong>dinâmica populacional</strong> e <strong>classificação de padrões biológicos</strong>.</li>

  <li><strong>Discussões e Perspectivas</strong><br>
  Reflexões sobre os limites, desafios éticos e perspectivas futuras no uso de RNAs em contextos científicos.</li>
</ol>
</div>

---

<div align="justify">
<em>Este material foi pensado especialmente para estudantes de Matemática, com forte embasamento técnico, e com interesse em expandir suas competências para o campo da Inteligência Artificial aplicada.</em>
</div>
'''

introducao='''
<div align="justify">

### 1. Introdução: A Convergência de IA, ML e Matemática

#### 1.1. O que é Inteligência Artificial (IA)?

A **Inteligência Artificial (IA)** é um campo interdisciplinar dedicado ao desenvolvimento de sistemas computacionais capazes de realizar tarefas que normalmente requerem inteligência humana. Entre essas tarefas, destacam-se o raciocínio lógico, o reconhecimento de padrões, a compreensão de linguagem natural, a tomada de decisões e a capacidade de aprender com a experiência.

Desde suas origens filosóficas na Antiguidade — como ilustrado nos mitos de Pigmalião e Pandora — até os primeiros algoritmos simbólicos no século XX, a IA passou por diversas fases. Inicialmente, concentrou-se em problemas de natureza lógica e formal, como jogos de tabuleiro e demonstrações matemáticas, que podiam ser resolvidos por meio de regras explícitas.

Contudo, o grande desafio revelou-se justamente nas tarefas que, embora triviais para os humanos, são difíceis de descrever formalmente — como reconhecer rostos, interpretar fala ou dirigir um carro. Esses problemas exigem uma representação de conhecimento contextual e tácito, que não pode ser facilmente codificada em regras.

Projetos como o **Cyc** buscaram codificar manualmente o conhecimento do mundo, mas fracassaram ao enfrentar a complexidade e ambiguidade inerentes ao raciocínio cotidiano. Um exemplo emblemático foi a incapacidade do sistema em lidar com a simples situação de uma pessoa usando um barbeador elétrico, gerando inferências logicamente corretas, mas semanticamente absurdas.

Esse impasse motivou uma mudança de paradigma: ao invés de programar explicitamente o conhecimento, buscou-se **permitir que os sistemas aprendessem a partir dos dados** — surgindo, assim, o campo do **Machine Learning**.

---

#### 1.2. O que é Machine Learning (ML)?

O **Machine Learning (ML)** é um subcampo da IA que busca desenvolver algoritmos capazes de identificar padrões e realizar previsões com base em dados. O aprendizado é realizado por meio da generalização a partir de exemplos, sem que se precise programar explicitamente todas as regras da tarefa.

O cerne do ML está na construção de uma **função de aproximação** $f: X \\rightarrow Y$ capaz de prever saídas $y \in Y$ a partir de entradas $x \in X$. O processo envolve encontrar uma hipótese $h \in \mathcal{H}$ dentro de um espaço de hipóteses $\mathcal{H}$ que minimize um erro esperado:

$$
\min_{h \in \mathcal{H}} \mathbb{E}_{(x, y) \sim \mathcal{D}}[\mathcal{L}(h(x), y)],
$$

onde $\mathcal{L}$ é uma função de perda e $\mathcal{D}$ é a distribuição dos dados.

Entretanto, a eficácia desses algoritmos depende criticamente da **representação dos dados**. Por exemplo, algoritmos simples como a regressão logística podem prever a necessidade de cesariana a partir de atributos médicos estruturados. Porém, se fornecermos diretamente uma imagem de ressonância magnética sem processamentos prévios, o algoritmo não extrairá informações úteis.

Isso evidencia a importância da **engenharia de atributos** — tradicionalmente feita de forma manual — e pavimenta o caminho para o **aprendizado de representações**, onde o próprio algoritmo aprende quais características são relevantes. Essa transição marca o avanço do ML em direção à **aprendizagem hierárquica**, base dos métodos modernos de **Deep Learning**.

---

#### 1.3. Breve Histórico das Redes Neurais Artificiais (RNAs)

A evolução das **Redes Neurais Artificiais (RNAs)** pode ser compreendida em três grandes ondas, cada uma com avanços significativos, mas também com limitações que motivaram transições para novas abordagens.

##### Primeira Onda — Cibernética (décadas de 1940 a 1960)

Essa fase inicial teve como base a tentativa de modelar o cérebro humano por meio de circuitos lógicos simples.

- **McCulloch e Pitts (1943)** propuseram o primeiro modelo de **neurônio artificial**: uma unidade binária com limiar, capaz de representar funções booleanas simples.
- A **regra de Hebb (1949)** introduziu a ideia de aprendizado baseado na coativação de neurônios — "neurônios que disparam juntos, conectam-se".
- O **Perceptron** (Rosenblatt, 1958) representou um marco ao permitir **aprendizado supervisionado** com ajuste de pesos baseado no erro.
- **ADALINE** (Widrow e Hoff, 1960) estendeu o perceptron para saídas contínuas, utilizando o algoritmo de **mínimos quadrados** e o precursor do **gradiente estocástico**.
  
Esses modelos, no entanto, eram limitados a problemas **linearmente separáveis**. A crítica devastadora de **Minsky e Papert (1969)** ao perceptron — por não resolver o problema XOR — levou a um longo período de descrédito nas RNAs.

##### Segunda Onda — Conexionismo (décadas de 1980 a 1990)

Com o avanço da computação e o interesse renovado pela cognição, surgem os modelos **multi-camadas**, capazes de representar relações não-lineares.

- O algoritmo de **retropropagação do erro (backpropagation)** (Rumelhart et al., 1986) tornou possível **ajustar pesos em redes com múltiplas camadas ocultas**, habilitando RNAs a resolver o problema XOR.
- **Hinton e colaboradores** introduziram o conceito de **representações distribuídas**, nas quais a ativação de múltiplas unidades codifica significados complexos.
- Surgem as primeiras arquiteturas para tarefas sequenciais, culminando na criação da **LSTM** (Hochreiter & Schmidhuber, 1997), que supera problemas como o **desvanecimento do gradiente**.
  
Apesar desses avanços, outras técnicas como **SVMs** e **modelos gráficos probabilísticos** passaram a dominar a cena, levando a novo declínio das RNAs nos anos 1990.

##### Terceira Onda — Deep Learning (a partir de 2006)

O renascimento das redes profundas foi impulsionado por avanços teóricos, aumento de dados e poder computacional.

- **Hinton (2006)** mostrou que redes profundas podiam ser **pré-treinadas camada a camada** com autoencoders ou modelos generativos (como Deep Belief Networks), facilitando o ajuste fino supervisionado posterior.
- O termo **Deep Learning** passou a designar o uso sistemático de múltiplas camadas para **aprendizado de representações hierárquicas**.
- Aplicações em larga escala se tornaram viáveis com o uso de **GPUs** e grandes datasets (ex: ImageNet).
- Modelos como **CNNs**, **RNNs**, **Transformers** e **autoencoders variacionais** demonstraram eficiência em tarefas como visão computacional, processamento de linguagem natural, bioinformática, entre outros.

Essa terceira onda consolidou o **Deep Learning** como abordagem dominante dentro do ML moderno, integrando conceitos computacionais, estatísticos e matemáticos em escala industrial.

---

#### 1.4. A Matemática por Trás da IA

A ascensão da IA e do ML está intimamente ligada a avanços em áreas centrais da matemática, como:

- **Álgebra Linear**, para representação vetorial de dados e parametrização de modelos;
- **Cálculo Diferencial**, especialmente no cálculo de gradientes via **retropropagação**;
- **Estatística e Probabilidade**, para modelagem de incertezas e aprendizado bayesiano;
- **Otimização Numérica**, no treinamento eficiente de modelos com milhões de parâmetros.

Assim, a matemática não apenas fundamenta os algoritmos, mas também oferece o **vocabulário formal** necessário para entender, avaliar e propor novas abordagens em IA.

---

</div>
'''

introducao2 = '''### 1. Introdução: a convergência entre IA, ML e Matemática

<div style="text-align: center;">
<a name="imagem"></a>
<img src="https://github.com/1moi6/minicurso_ia/blob/main/assets/images/teste.png?raw=true" width="400"/>
<p style="font-style: italic;">Figura: Arquitetura de uma Rede Neural</p>
</div>

#### 1.1. O que é Inteligência Artificial (IA)?



<div align="justify">

A <strong> Inteligência Artificial (IA) </strong> é uma área em constante evolução cujo propósito é dotar máquinas da capacidade de realizar tarefas tipicamente humanas, como raciocinar, aprender, planejar e criar. Sistemas de IA são projetados para interagir com o ambiente, interpretando dados — provenientes de sensores, bases de dados ou outras fontes — e respondendo de maneira adaptativa.

Um traço fundamental da IA é sua **autonomia**: a capacidade de modificar o próprio comportamento com base na análise de experiências anteriores. Isso a diferencia de meros sistemas programados, posicionando a IA como um campo voltado à construção de sistemas **capazes de aprender e evoluir**, de maneira análoga aos organismos biológicos.

Por sua natureza, a IA é um campo **interdisciplinar**, apoiado por áreas como ciência cognitiva, ciência da computação e, sobretudo, **matemática**.

Nos estágios iniciais, os pesquisadores tentaram codificar o conhecimento humano por meio de regras explícitas. Projetos como o **Cyc** fracassaram ao tentar formalizar o conhecimento do mundo real — por exemplo, ao não conseguir interpretar corretamente uma história simples sobre um homem utilizando um barbeador elétrico. Essas limitações revelaram que a inteligência artificial não poderia se basear apenas em regras fixas.

Esse desafio levou ao surgimento de abordagens baseadas em **aprendizado com dados**, marcando a transição do paradigma simbólico para o **Machine Learning (ML)**. Esse novo paradigma, baseado em ferramentas matemáticas como **probabilidade** e **otimização**, tornou-se central no desenvolvimento da IA moderna.

</div>

---

#### 1.2. O que é Machine Learning (ML)?
<div align="justify">

O <strong> Aprendizado de Máquinas </strong> (ou <strong> Machine Learning (ML) </strong>) é um subcampo da IA que capacita os sistemas a <strong>aprenderem diretamente a partir de dados</strong>, sem depender de instruções explícitas para cada situação. Utilizando algoritmos capazes de detectar padrões em grandes volumes de informação, o ML permite que computadores realizem <strong> previsões e decisões <strong> de forma cada vez mais precisa à medida que são expostos a novos dados.

Ao contrário da IA tradicional baseada em regras, o ML permite que as máquinas **generalizem experiências anteriores** para novos contextos. Assim, o ML se configura como a principal ferramenta para concretizar a *inteligência* nos sistemas de IA.

A eficácia de um modelo de ML depende fortemente de **como os dados são representados**. Por exemplo, uma **regressão logística** pode prever a necessidade de uma cesariana se os dados incluírem a presença de uma cicatriz uterina. No entanto, o mesmo modelo não funcionaria com imagens brutas de ressonância magnética (MRI), pois os valores dos pixels isoladamente não revelam essa informação.  

Esse desafio de representação é recorrente na ciência da computação, onde **a forma como os dados são estruturados impacta diretamente o desempenho dos algoritmos**.

Durante muito tempo, a prática comum era construir manualmente um conjunto de **características (features)** relevantes para o problema e alimentá-las a um algoritmo de ML simples. Mas tarefas mais complexas, como detecção de objetos em imagens, tornaram inviável a criação manual dessas representações, dadas as variações de luz, sombra, ângulo e oclusão.

Isso impulsionou o desenvolvimento do **aprendizado de representações** (*representation learning*), no qual o próprio algoritmo aprende automaticamente quais aspectos dos dados são relevantes. Esse avanço reduziu a dependência de conhecimento humano especializado, **aumentando a escalabilidade e autonomia dos sistemas de IA**.

</div>

---

#### 1.3. Breve Histórico das Redes Neurais Artificiais (RNAs)
<div align="justify">

As </strong>Redes Neurais Artificiais (RNAs)</strong> possuem uma trajetória marcada por altos e baixos, refletindo tanto os avanços teóricos quanto as limitações tecnológicas de cada época. Essa história pode ser dividida em três grandes ondas:

</div>

##### Primeira Onda — Cibernética (décadas de 1940–1960)

- Criação do **neurônio de McCulloch e Pitts** (1943) e da **regra de Hebb** (1949)
- Desenvolvimento dos modelos **Perceptron** (Rosenblatt, 1958) e **ADALINE** (Widrow & Hoff, 1960)
- Introdução do **gradiente descendente estocástico**, base de muitos algoritmos atuais

Esses modelos, embora pioneiros, eram limitados a funções lineares. A incapacidade de resolver problemas simples, como a função XOR, levou à crítica de **Minsky e Papert (1969)** e ao subsequente declínio da área.

##### Segunda Onda — Conexionismo (décadas de 1980–1990)

- Redescoberta e popularização do algoritmo de **retropropagação** (*backpropagation*) por **Rumelhart et al. (1986)**
- Introdução da **representação distribuída** (Hinton et al., 1986)
- Desenvolvimento das **LSTM** (Long Short-Term Memory) por **Hochreiter e Schmidhuber (1997)** para lidar com dependências temporais

<div align="justify">

Apesar desses avanços, o campo perdeu tração frente ao crescimento de técnicas alternativas, como **máquinas de vetores de suporte** e **modelos probabilísticos gráficos**.

</div>

##### Terceira Onda — Deep Learning (a partir de 2006)

- Avanço decisivo de **Geoffrey Hinton**, com as **deep belief networks** treinadas via pré-treinamento camada a camada
- Consolidação do termo **Deep Learning**, destacando a importância das redes **profundas**
- Expansão acelerada graças a:
  - Disponibilidade de **grandes volumes de dados**
  - Avanços em **hardware (GPUs)** e bibliotecas de software
  - Aumento da precisão de modelos aplicados em **visão computacional, linguagem natural e robótica**

---
<div align="justify">

A evolução das RNAs ilustra a interdependência entre <strong>teoria matemática e capacidade computacional</strong>. A matemática fornece os fundamentos para os modelos, mas é o avanço tecnológico que permite sua implementação em escala real. O renascimento das RNAs com o Deep Learning mostra que **inovações significativas exigem tanto compreensão teórica quanto infraestrutura para serem efetivamente aplicadas.**

[Ir para Redes Neurais](#imagem)
</div>

---
'''

fundamentos = '''
<a name="principais_conceitos"></a>
<div align="justify">

O aprendizado de máquina, embora popularmente associado à construção empírica de modelos a partir de grandes volumes de dados, repousa sobre uma infraestrutura teórica robusta, cujos alicerces se encontram, sobretudo, na **matemática** e na **estatística**. Estas duas áreas, embora distintas em seus enfoques, atuam de forma complementar e indispensável para garantir que os modelos de machine learning não apenas funcionem computacionalmente, mas sejam **teoricamente justificáveis** e **estatisticamente confiáveis**.

Do lado da **matemática pura e aplicada**, o aprendizado de máquina é essencialmente um problema de **otimização de funções multivariadas**. A aprendizagem consiste em encontrar os parâmetros $\\theta$ que minimizam uma função de perda definida sobre um conjunto de dados:

$$
\min_{\\theta} \; \\frac{1}{m} \sum_{i=1}^m \mathcal{L}(f_{\\theta}(\mathbf{x}^{(i)}), y^{(i)}).
$$

Para resolver esse problema, recorre-se a ferramentas da **análise matemática**, como o cálculo diferencial e integral, e da **álgebra linear**, indispensável para manipulação de vetores, matrizes e transformações em espaços de alta dimensão. Além disso, a **teoria da otimização** fornece os fundamentos formais para garantir existência, unicidade, e caracterização dos mínimos — locais ou globais — da função de perda, enquanto a **análise numérica** disponibiliza algoritmos computacionalmente viáveis (como gradiente descendente, métodos de Newton, otimização estocástica e variantes adaptativas) que possibilitam encontrar boas soluções mesmo em situações onde não há formulação analítica fechada.

Entretanto, **a matemática por si só não é suficiente** para justificar por que um modelo treinado em um conjunto de dados consegue, de fato, generalizar para novos dados. É aqui que entra o papel fundamental da **estatística**, cuja principal contribuição é fornecer o aparato conceitual para compreender a **incerteza** e o **comportamento probabilístico** dos dados e das estimativas produzidas.

A estatística permite formalizar a noção de **generalização**: ainda que a otimização minimize o erro sobre o conjunto de treinamento, a verdadeira medida de sucesso de um modelo está em seu desempenho sobre dados **não vistos**, oriundos da mesma distribuição. Conceitos como **conjunto de teste**, **validação cruzada**, **viés** e **variância** são elementos centrais da teoria estatística que permitem quantificar essa capacidade de generalização. 

A clássica **decomposição do erro esperado** em:

$$
\\text{Erro total} = \\text{Viés}^2 + \\text{Variância} + \\text{Ruído irreducível}
$$

fornece uma lente analítica para entender os compensações entre modelos mais simples (alta tendência ao viés) e modelos mais complexos (alta variância), sendo a base para técnicas como regularização, seleção de modelos e escolha de hiperparâmetros.

Além disso, a estatística oferece **garantias teóricas**, como limites de generalização baseados em desigualdades de concentração (Hoeffding, McDiarmid) ou dimensões de complexidade (VC-dimension, Rademacher complexity), que justificam, sob determinadas condições, que **o desempenho em um conjunto de treino é um bom preditor do desempenho futuro** — fundamento este sem o qual o uso prático de modelos seria meramente especulativo.

Em síntese, a **matemática contribui com a estrutura e a resolução computacional do problema de aprendizado**, enquanto a **estatística garante a validade inferencial** do que é aprendido. Sem a primeira, não haveria modelo eficaz; sem a segunda, não haveria confiança de que o modelo funcionaria fora do treinamento.

Portanto, o aprendizado de máquina moderno deve ser compreendido não como um processo empírico e opaco, mas como um **procedimento matemático-estatístico rigoroso**, no qual **funções são otimizadas, incertezas são quantificadas, e previsões são sustentadas por princípios sólidos**. É esse entrelaçamento de teoria matemática com inferência estatística que confere ao aprendizado de máquina o seu caráter científico e o seu poder preditivo.

</div>

#### 2.1. Conceitos Essenciais

<div align="justify">

O **Aprendizado de Máquina (Machine Learning)** é uma subárea da inteligência artificial voltada para o desenvolvimento de algoritmos capazes de aprender a realizar tarefas a partir de dados. Em termos formais, segundo a definição clássica de Tom Mitchell, *“um programa de computador é dito aprender de uma experiência $E$ com relação a uma tarefa $T$ e uma medida de desempenho $P$, se seu desempenho em $T$, medido por $P$, melhora com a experiência $E$”*.

No **aprendizado supervisionado**, uma das abordagens centrais do Machine Learning, o modelo tem acesso a um **conjunto de dados rotulado**, ou seja, para cada exemplo de entrada $\mathbf{x} \in \mathbb{R}^n$, está associado um rótulo ou valor de saída $y$. O objetivo do algoritmo é aprender uma função $f: \mathbb{R}^n \\to \mathcal{Y}$ que generalize bem para novos dados, isto é, que seja capaz de prever corretamente $y$ a partir de uma nova $\mathbf{x}$ não observada durante o treinamento.

Esse processo de "aprender com dados" envolve três componentes fundamentais:

- **A tarefa $T$**: por exemplo, classificar imagens, prever preços ou diagnosticar doenças.
- **A medida de desempenho $P$**: acurácia, erro quadrático médio, perda logística, entre outras.
- **A experiência $E$**: representada por um conjunto de dados $D = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^m$, geralmente dividida em subconjuntos de *treinamento*, *validação* e *teste*.

O treinamento consiste em ajustar os **parâmetros** do modelo para minimizar uma função de perda em relação aos exemplos do conjunto de treinamento. A partir disso, espera-se que o modelo consiga **generalizar**, ou seja, apresentar bom desempenho em exemplos ainda não vistos, pertencentes ao mesmo processo gerador de dados (assumido *i.i.d.* — *independent and identically distributed*).

No contexto supervisionado, distinguem-se duas categorias fundamentais de problemas:

- **Classificação**: quando a variável alvo $y$ assume valores discretos, geralmente representando categorias. Exemplos incluem a identificação de e-mails como "spam" ou "não-spam", a detecção de tumores em imagens médicas ou a classificação de espécies de plantas.

- **Regressão**: quando a variável alvo $y$ é contínua. Exemplos clássicos são a previsão de valores de imóveis com base em localização, área e número de quartos, ou a estimação da temperatura em função de variáveis meteorológicas.

Essas tarefas exigem técnicas distintas de modelagem, mas compartilham a estrutura essencial do aprendizado supervisionado. Um aspecto central no desenvolvimento de bons modelos é o **compromisso entre viés e variância**: modelos com **baixa capacidade** podem não capturar a complexidade dos dados (*subajuste*), enquanto modelos com **alta capacidade** podem se ajustar demais ao conjunto de treinamento (*sobreajuste*), perdendo capacidade de generalização.

Outro conceito relevante é o de **conjunto de validação**, usado para ajustar hiperparâmetros e evitar sobreajuste ao conjunto de treinamento. Após essa etapa, um conjunto de teste independente é utilizado para estimar a performance final do modelo.

</div>

<a name="treinamento"></a>

#### 2.2. O Processo de Treinamento

<div align="justify">

O **processo de treinamento** em aprendizado supervisionado visa ajustar os **parâmetros internos** de um modelo de modo que ele seja capaz de aprender a partir dos dados. Dado um conjunto de treinamento $D = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^m$, o objetivo do aprendizado é encontrar uma função $f_{\\theta}$, parametrizada por $\\theta$, que minimize uma função de custo (ou perda), que quantifica o erro entre a predição do modelo $f_{\\theta}(\mathbf{x}^{(i)})$ e o rótulo verdadeiro $y^{(i)}$.

Matematicamente, queremos resolver o seguinte problema de minimização:

$$
\min_{\\theta} \; \\frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(f_{\\theta}(\mathbf{x}^{(i)}), y^{(i)}),
$$

onde $\mathcal{L}$ é uma função de perda apropriada. Exemplos comuns incluem:
- **Erro quadrático médio (MSE)** – em regressão:
  $$
  \mathcal{L}_{\\text{MSE}}(f(\mathbf{x}), y) = \\frac{1}{2}(f(\mathbf{x}) - y)^2
  $$

- **Entropia cruzada (Cross-Entropy)** – em classificação binária:
  $$
  \mathcal{L}_{\\text{CE}}(f(\mathbf{x}), y) = - \left[ y \log f(\mathbf{x}) + (1 - y) \log (1 - f(\mathbf{x})) \\right]
  $$
O **gradiente descendente** atualiza os parâmetros $\\theta$ iterativamente pela regra:

$$
\\theta \leftarrow \\theta - \\frac{\eta}{m} \sum_{i=1}^{m} \\nabla_{\\theta} \mathcal{L}(f_{\\theta}(\mathbf{x}^{(i)}), y^{(i)}),
$$

onde:

- $\eta > 0$ é a **taxa de aprendizado** (learning rate),
- $\\nabla_{\\theta} \mathcal{L}$ é o **gradiente da função de perda** em relação aos parâmetros $\\theta$.

Essa fórmula é completamente **genérica** e se aplica a qualquer modelo $f_{\\theta}$ e função de perda $\mathcal{L}$. O formato específico do gradiente depende da estrutura de $f_{\\theta}$ e da escolha de $\mathcal{L}$.


</div>

---

#### 2.3. Regressão Linear e Logística

<strong>2.3.1 Regressão Linear e Logística</strong> 

<div align="justify">

A **regressão linear** é um dos modelos mais simples e fundamentais do aprendizado supervisionado. Seu objetivo é modelar uma relação linear entre um vetor de entrada $\mathbf{x} \in \mathbb{R}^n$ e uma variável de saída contínua $y \in \mathbb{R}$, assumindo que:

$$
y \\approx \hat{y} = f_{\\theta}(\mathbf{x}) = \mathbf{w}^\\top \mathbf{x} + b,
$$

onde:

- $\mathbf{w} \in \mathbb{R}^n$ é o vetor de pesos (ou coeficientes) que determina a contribuição de cada variável de entrada;
- $b \in \mathbb{R}$ é o termo de interceptação (viés);
- $\hat{y}$ é a saída predita pelo modelo para a entrada $\mathbf{x}$.

O objetivo do treinamento é encontrar os parâmetros $(\mathbf{w}, b)$ que minimizem o **erro quadrático médio** (MSE — *mean squared error*) entre os valores previstos $\hat{y}^{(i)}$ e os valores reais $y^{(i)}$ sobre um conjunto de treinamento com $m$ exemplos:

$$
\mathcal{L}(\mathbf{w}, b) = \\frac{1}{m} \sum_{i=1}^m \left( \mathbf{w}^\\top \mathbf{x}^{(i)} + b - y^{(i)} \\right)^2.
$$

Essa função de perda é convexa, o que permite encontrar uma solução analítica através do chamado **método dos mínimos quadrados**. Para isso, é comum reescrever o modelo em notação matricial. Seja:

- $\mathbf{X} \in \mathbb{R}^{m \\times n}$ a matriz de entrada, onde cada linha é um vetor $\mathbf{x}^{(i)}$;
- $\mathbf{y} \in \mathbb{R}^m$ o vetor de saídas reais;
- $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b\mathbf{1}$ o vetor de predições.

A solução que minimiza a função de perda, ignorando momentaneamente o viés $b$ (ou absorvendo-o em $\mathbf{w}$ com uma variável adicional constante), é dada por:

$$
\mathbf{w}^* = (\mathbf{X}^\\top \mathbf{X})^{-1} \mathbf{X}^\\top \mathbf{y},
$$

desde que $\mathbf{X}^\\top \mathbf{X}$ seja invertível. Esse resultado é conhecido como **equação normal** da regressão linear.

</div>

---

<strong>2.3.2 Regressão Logística</strong>

<div align="justify">

A **regressão logística** é um modelo estatístico utilizado para problemas de **classificação binária**, isto é, quando a variável de saída $y$ assume apenas dois valores, geralmente codificados como $y \in \{0, 1\}$. Embora tenha a palavra "regressão" no nome, seu propósito é **classificar** observações, e não estimar valores contínuos.

A principal ideia é transformar a saída de uma regressão linear — que pode assumir qualquer valor real — em uma **probabilidade** no intervalo $(0, 1)$, utilizando a **função logística**, também conhecida como **função sigmoide**:

$$
\sigma(z) = \\frac{1}{1 + e^{-z}}.
$$

No modelo de regressão logística, a probabilidade de que $y = 1$ dado $\mathbf{x}$ é modelada como:

$$
P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\\top \mathbf{x} + b) = \\frac{1}{1 + e^{-(\mathbf{w}^\\top \mathbf{x} + b)}}.
$$

De forma equivalente:

$$
P(y = 0 \mid \mathbf{x}) = 1 - \sigma(\mathbf{w}^\\top \mathbf{x} + b).
$$

A saída do modelo $\hat{y} = \sigma(\mathbf{w}^\\top \mathbf{x} + b)$ pode ser interpretada como a **probabilidade estimada** de que a observação pertença à classe 1. A decisão final é feita com base em um limiar (threshold), geralmente 0.5:

$$
\hat{y} =
\\begin{cases}
1, & \\text{se } \sigma(\mathbf{w}^\\top \mathbf{x} + b) \geq 0.5 \\\ 
0, & \\text{caso contrário}
\end{cases}
$$

Para treinar o modelo, utiliza-se a **função de perda logarítmica** (também chamada de **log-loss** ou **entropia cruzada**) definida por:

$$
\mathcal{L}(\mathbf{w}, b) = - \\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \\right],
$$

onde $\hat{y}^{(i)} = \sigma(\mathbf{w}^\\top \mathbf{x}^{(i)} + b)$.

Essa função é convexa e pode ser minimizada por métodos iterativos como o **gradiente descendente**, sendo a base para diversos modelos mais sofisticados em aprendizado de máquina.

Como mencionado anteriormente, para usar o algoritmo de **gradiente descendente** é necessário obter as derivadas parciais de $\mathcal{L}$ em relação a $\mathbf{w}$ e $b$.
Nesse caso, a derivada da função de perda em relação a um peso $w_j$ é:

$$
\\frac{\partial \mathcal{L}}{\partial w_j} = \\frac{1}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \\right) x_j^{(i)}.
$$

De forma vetorial, o gradiente em relação a $\mathbf{w}$ é:

$$
\\nabla_{\mathbf{w}} \mathcal{L} = \\frac{1}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \\right) \mathbf{x}^{(i)}.
$$

Já a derivada da perda em relação ao viés $b$ é:

$$
\\frac{\partial \mathcal{L}}{\partial b} = \\frac{1}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \\right).
$$


Assim, considerando uma taxa de aprendizado $\eta > 0$, os parâmetros são atualizados a cada iteração por:

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \\nabla_{\mathbf{w}} \mathcal{L},
\quad
b \leftarrow b - \eta \\frac{\partial \mathcal{L}}{\partial b}.
$$


Nessa atualização:
- O termo $(\hat{y}^{(i)} - y^{(i)})$ representa o **erro de predição**;
- O vetor $\mathbf{x}^{(i)}$ serve como um fator de **sensibilidade** para cada entrada;
- O gradiente aponta na direção de maior crescimento da função de perda; o gradiente descendente move os parâmetros na direção oposta, reduzindo a perda.

Esse procedimento é repetido iterativamente até convergência, geralmente monitorando a perda ou o erro de validação para interromper o processo.

</div>
 
---

#### 2.5. Capacidade do Modelo, Overfitting e Underfitting

<div align="justify">

Ao treinar um modelo de aprendizado de máquina, o objetivo não é apenas ajustar-se bem aos dados de treinamento, mas sim **generalizar** para novos dados nunca vistos. Compreender os conceitos de **capacidade do modelo**, **subajuste** (*underfitting*) e **sobreajuste** (*overfitting*) é essencial para alcançar esse objetivo.


**Capacidade** refere-se à **complexidade funcional** que um modelo é capaz de representar. Um modelo com baixa capacidade pode representar apenas funções simples (por exemplo, funções lineares), enquanto um modelo com alta capacidade pode representar funções complexas e altamente não lineares.

De modo geral:

- **Baixa capacidade**: o modelo não é capaz de capturar padrões importantes dos dados;
- **Alta capacidade**: o modelo é flexível o suficiente para se ajustar até mesmo ao ruído presente nos dados de treinamento.

É comum, no processo de modelagem supervisionada, dividir o conjunto total de dados em três subconjuntos: o **conjunto de treinamento** ($\mathcal{D}_{\\text{train}}$), utilizado para ajustar os parâmetros internos do modelo; o **conjunto de validação** ($\\mathcal{D}_{\\text{val}}$), usado para ajustar hiperparâmetros e monitorar o desempenho do modelo em dados não vistos durante o processo de otimização; e o **conjunto de teste** ($\mathcal{D}_{\\text{test}}$), reservado exclusivamente para estimar o desempenho final do modelo, refletindo sua capacidade de generalização.

A partir desses conjuntos, podemos definir medidas quantitativas para avaliar o aprendizado. O **erro de treinamento** é a perda computada sobre $\mathcal{D}_{\\text{train}}$ e indica quão bem o modelo se ajusta aos dados que foram usados para sua construção. Já o **erro de validação** (calculado sobre $\mathcal{D}_{\\text{val}}$) permite monitorar o ajuste a dados novos ao longo do processo de treino, ajudando a evitar sobreajuste. Por fim, o **erro de teste**, medido em $\mathcal{D}_{\\text{test}}$, representa a estimativa final e imparcial da performance do modelo em situações reais.

Comparar essas três métricas permite diagnosticar se o modelo sofre de **underfitting** (quando o erro de treino já é alto) ou **overfitting** (quando o erro de treino é baixo, mas o erro de validação ou teste é alto), orientando decisões sobre capacidade do modelo, regularização ou necessidade de mais dados.


Nesse contexto, **Underfitting** (subajuste) ocorre quando o modelo tem **capacidade insuficiente** para capturar a estrutura dos dados. Isso se traduz em **alto erro no treinamento e também no teste**.
Já o **Overfitting** (sobreajuste) acontece quando o modelo tem **capacidade excessiva** e aprende não apenas os padrões verdadeiros dos dados, mas também **o ruído e as flutuações específicas** do conjunto de treinamento. Isso resulta em **baixo erro no treinamento**, mas **alto erro no teste**.


Podemos ilustrar esse comportamento com a **curva de erro** em função da capacidade do modelo:

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Overfitting_svg.svg/800px-Overfitting_svg.svg.png" width="400"/>
</p>

O ponto ótimo de capacidade está geralmente associado a um equilíbrio entre **viés** (erro sistemático) e **variância** (sensibilidade a pequenas variações nos dados).


Matematicamente, esse fenômeno pode ser estudado a partir da decomposição do erro esperado:

$$
\mathbb{E}_{\mathcal{D}}[(f_{\\theta}(\mathbf{x}) - y)^2] = \\text{viés}^2 + \\text{variância} + \\text{ruído irredutível}.
$$

- Um modelo com **alto viés** tende a subestimar a complexidade da função verdadeira;
- Um modelo com **alta variância** se adapta demais às amostras do treinamento;
- O **ruído irredutível** representa a variabilidade intrínseca dos dados, que nenhum modelo pode eliminar.


Portanto, escolher um modelo adequado envolve:

1. Ajustar a **capacidade do modelo** ao tamanho e à complexidade dos dados;
2. Utilizar técnicas como **validação cruzada**, **regularização** e **aumento de dados** para evitar o sobreajuste;
3. Monitorar a **curva de aprendizagem** para detectar sinais precoces de overfitting ou underfitting.


A prática de dividir um conjunto de dados em subconjuntos de **treinamento**, **validação** e **teste** fundamenta-se em princípios centrais da *estatística inferencial*. Em essência, essa estrutura tem por objetivo isolar os efeitos do *viés*, da *variância* e do *ruído aleatório* no processo de modelagem, permitindo estimativas não tendenciosas do erro de generalização e diagnósticos empíricos sobre o comportamento preditivo do modelo.

O **conjunto de treinamento** desempenha o papel de fornecer uma amostra sobre a qual o modelo ajusta seus parâmetros internos. Do ponto de vista estatístico, este processo equivale a estimar uma função preditiva com base em uma amostra observacional. No entanto, qualquer avaliação realizada nesse mesmo conjunto estará sujeita a *viés otimista*, uma vez que os parâmetros foram calibrados para minimizar o erro nessa amostra específica. Por isso, o erro de treinamento é geralmente *subestimado* e, sozinho, não permite aferir a capacidade de generalização.

O **conjunto de validação** funciona como uma amostra independente e não utilizada no ajuste direto dos parâmetros do modelo. Seu propósito é fornecer uma *estimativa imparcial do desempenho do modelo em dados fora da amostra*, embora ainda dentro do mesmo processo de construção. Isso permite ao analista avaliar a *instabilidade do modelo frente a variações nos dados*, que está relacionada ao componente de *variância*. Um modelo que apresenta erro de validação muito superior ao de treinamento sugere que está capturando ruídos específicos da amostra de treino — caracterizando *overfitting*. Por outro lado, se o erro permanece elevado tanto na validação quanto no treino, indica que o modelo possui *viés elevado*, incapaz de representar adequadamente os padrões subjacentes aos dados.

Já o **conjunto de teste** é reservado para uma avaliação estatisticamente rigorosa do modelo final. Ele simula a aplicação do modelo em uma nova amostra, retirada da mesma distribuição subjacente, mas completamente *alheia ao processo de treinamento e seleção de modelo*. Esse isolamento é crucial para que o erro de teste represente uma *estimativa não enviesada do erro esperado verdadeiro*, o qual está associado à performance futura do modelo em contexto de produção. Em termos inferenciais, o conjunto de teste opera como um “conjunto de validação externa”, fornecendo uma medida da capacidade preditiva generalizável do modelo.

Essa divisão também possibilita uma *avaliação empírica indireta dos componentes da decomposição do erro*: enquanto o erro de treino reflete a capacidade do modelo de memorizar, o erro de validação informa sobre sua habilidade de generalizar sob os mesmos pressupostos estatísticos, e o erro de teste avalia a robustez dessa generalização sem qualquer influência de ajustes internos. A comparação entre esses erros, portanto, oferece uma leitura prática da interação entre *viés*, *variância* e *ruído* — mesmo quando tais quantidades não são diretamente observáveis.

Finalmente, vale destacar que essas práticas se alinham aos princípios de *planejamento experimental* e *validação cruzada* amplamente utilizados em estatística. Ao garantir que as inferências sejam feitas sobre dados independentes, evita-se a contaminação por dependências induzidas pelo treinamento, assegurando que os modelos produzidos não apenas se ajustem aos dados passados, mas também se sustentem *como preditores confiáveis para o futuro*.

---
</div>
#### 2.6. Métricas de Avaliação

<div align="justify">

Um aspecto sutil, mas conceitualmente importante no estudo de Machine Learning, é a distinção entre a **função de perda (loss function)** e as **métricas de avaliação**. Embora ambas sirvam para quantificar o desempenho de modelos, elas têm **papéis e objetivos diferentes** dentro do processo de modelagem.


A **função de perda** é um **objeto matemático central** no treinamento do modelo. Ela mede, para cada par de entrada $(\mathbf{x}, y)$, o grau de erro entre a saída prevista pelo modelo $f_{\\theta}(\mathbf{x})$ e a saída verdadeira $y$.

A função de perda deve ser **diferenciável**, pois será usada para calcular gradientes durante o treinamento via **algoritmos de otimização**, como o **gradiente descendente**.


Já as **métricas de avaliação** são utilizadas **após o treinamento**, para julgar a **eficácia do modelo** em dados não vistos, muitas vezes com propósitos práticos ou de comunicação.

Diferentemente das funções de perda, as métricas **não precisam ser diferenciáveis** e muitas vezes **operam sobre conjuntos inteiros de previsões**, como no caso da acurácia, precisão ou F1-score.

Um modelo de classificação binária pode ser treinado usando **entropia cruzada** como função de perda (que opera sobre probabilidades), mas avaliado em termos de **acurácia**, que exige que as probabilidades sejam transformadas em decisões (por exemplo, classificando como "positivo" se $f(\mathbf{x}) > 0{,}5$).

<div align="center">

| Aspecto              | Função de Perda                        | Métricas de Avaliação                |
|----------------------|----------------------------------------|--------------------------------------|
| Quando é usada       | Durante o treinamento                  | Após o treinamento                   |
| Objetivo             | Otimizar os parâmetros do modelo       | Avaliar o desempenho preditivo       |
| Exige derivadas?     | Sim (para gradiente descendente)       | Não (pode ser baseada em contagens)  |
| Exemplos             | MSE, Cross-Entropy                     | Acurácia, Precisão, Recall, F1-score |

</div>
---

</div>

<div align="justify">

Em tarefas de **classificação supervisionada**, é fundamental avaliar a qualidade das predições do modelo de maneira quantitativa. Para isso, utiliza-se a **matriz de confusão**, a partir da qual derivam-se diversas métricas, cada uma enfatizando aspectos distintos da performance do classificador.

<div align="center">

|                       | Previsto Positivo | Previsto Negativo |
|-----------------------|-------------------|-------------------|
| **Real Positivo**     | TP (Verdadeiro Positivo)  | FN (Falso Negativo)   |
| **Real Negativo**     | FP (Falso Positivo)       | TN (Verdadeiro Negativo) |

</div>

- **TP (True Positive)**: o modelo previu positivo, e a classe verdadeira era positiva;
- **TN (True Negative)**: o modelo previu negativo, e a classe verdadeira era negativa;
- **FP (False Positive)**: o modelo previu positivo, mas a classe verdadeira era negativa (erro tipo I);
- **FN (False Negative)**: o modelo previu negativo, mas a classe verdadeira era positiva (erro tipo II).

A **acurácia** mede a proporção de acertos (positivos e negativos) em relação ao total de previsões:

$$
\\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}
$$

É uma métrica intuitiva e fácil de interpretar. No entanto, **pode ser enganosa em conjuntos de dados desbalanceados** — isto é, quando uma das classes é muito mais frequente que a outra.

**Exemplo:**  
Suponha um classificador com os seguintes valores:

- TP = 70  
- TN = 20  
- FP = 5  
- FN = 5

Então:

$$
\\text{Accuracy} = \\frac{70 + 20}{70 + 20 + 5 + 5} = \\frac{90}{100} = 0{,}90
$$

O modelo acertou 90% das previsões.

A **precisão** (ou valor preditivo positivo) mede a **confiabilidade das predições positivas**:

$$
\\text{Precision} = \\frac{TP}{TP + FP}
$$

Ela responde à pergunta: *Dentre as instâncias classificadas como positivas pelo modelo, quantas realmente o são?*

É crucial em aplicações onde **falsos positivos têm alto custo** — por exemplo, um diagnóstico médico que identifique uma doença grave em alguém saudável.

**Exemplo (continuação):**

$$
\\text{Precision} = \\frac{70}{70 + 5} = \\frac{70}{75} \\approx 0{,}933
$$


O **recall** mede a **capacidade do modelo de identificar os positivos reais**:

$$
\\text{Recall} = \\frac{TP}{TP + FN}
$$

Responde à pergunta: *Dentre os casos realmente positivos, quantos foram detectados?*

É vital quando **falsos negativos são perigosos**, como em sistemas de segurança ou exames médicos que precisam detectar todos os casos críticos.

**Exemplo (continuação):**

$$
\\text{Recall} = \\frac{70}{70 + 5} = \\frac{70}{75} \\approx 0{,}933
$$


O **F1-score** é a **média harmônica** entre precisão e recall. Ele equilibra as duas métricas, penalizando fortemente desequilíbrios entre elas:

$$
F1 = 2 \cdot \\frac{\text{Precision} \cdot \\text{Recall}}{\text{Precision} + \\text{Recall}}
$$

É particularmente útil em **cenários com classes desbalanceadas**, onde uma única métrica (como a acurácia) pode ocultar falhas importantes.

**Exemplo (continuação):**

Com precisão = recall = 0,933, temos:

$$
F1 = 2 \cdot \\frac{0{,}933 \cdot 0{,}933}{0{,}933 + 0{,}933} = 2 \cdot \\frac{0{,}87}{1{,}866} \\approx 0{,}933
$$


- A **acurácia** é uma boa métrica quando as classes estão balanceadas;
- A **precisão** é preferida quando os **falsos positivos** são mais prejudiciais;
- O **recall** é importante quando os **falsos negativos** são críticos;
- O **F1-score** é útil quando há **compensação entre precisão e recall**.

</div>

</div>
'''
