fundamentos_dct = [
    {
        'titulo': "Introdução", 'conteudo': '''<div align="justify">

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
    {'titulo': "Conceitos Essenciais", 'conteudo': '''<div align="justify">

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
    {'titulo': "Treinamento", 'conteudo': '''<div align="justify">

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
        'titulo': "Capacidade do Modelo, Subajuste e Sobreajuste", 'conteudo': '''<div align="justify">

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
        'titulo': 'Exemplo experimental',
        'conteudo': '''<div align="justify"> Podemos visualizar empiricamente o comportamento dos erros de generalização de um modelo de regressão supervisionada por meio de um experimento que simula diferentes níveis de complexidade indutiva. Para isso, consideramos como função alvo a ser aprendida a expressão não linear e suave:

$$
f(x) = 2x + \cos(4\pi x), \quad x \in [0,1].
$$

O objetivo é construir modelos aproximadores dessa função a partir de amostras com ruído e avaliar, para diferentes níveis de complexidade, os erros de viés, variância e ruído irreducível que compõem o erro quadrático médio esperado.

Para a aproximação de $f(x)$, empregamos uma família de modelos polinomiais da forma:

$$
f_\\alpha(x) = \sum_{k=0}^{d} a_k x^k,
$$

onde os coeficientes $a_k$ são aprendidos a partir de dados sintéticos ruidosos. A complexidade do modelo é controlada por uma **restrição dura (hard constraint)** imposta sobre os coeficientes do polinômio, limitando sua magnitude de maneira explícita. Essa restrição assume a forma:

$$
|a_k| \leq M \cdot \\alpha^k, \quad \\text{para } k = 0, 1, \ldots, d,
$$

em que $M$ é um hiperparâmetro fixo que define a escala global dos coeficientes, e $\\alpha \in (0,1]$ atua como um parâmetro de controle da complexidade do modelo. Coeficientes de ordem mais alta são assim restringidos mais severamente quando $\\alpha$ é pequeno, forçando o modelo a ser mais suave e menos propenso ao sobreajuste. Por outro lado, à medida que $\\alpha$ se aproxima de 1, os coeficientes de ordens superiores podem assumir maiores magnitudes, permitindo maior expressividade e complexidade no ajuste.

A cada iteração do experimento, geramos um conjunto de dados de treinamento $\{(x_i, y_i)\}_{i=1}^n$, com $x_i \sim \mathcal{U}(0,1)$ e $y_i = f(x_i) + \\varepsilon_i$, onde $\\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ representa o ruído aditivo. O modelo $f_\\alpha(x)$ é então ajustado por meio da minimização da função de perda quadrática empírica:

$$
\min_{a_0, \ldots, a_d} \left\{ \\frac{1}{n} \sum_{i=1}^{n} \left(y_i - \sum_{k=0}^{d} a_k x_i^k \\right)^2 \\right\},
$$

sujeita às restrições:

$$
|a_k| \leq M \cdot \\alpha^k, \quad \\forall k.
$$

Essa formulação caracteriza uma regularização implícita via restrições no espaço de hipóteses, em contraste com métodos de regularização penalizada. Não há penalidade explícita adicionada à função de perda — o controle da complexidade ocorre diretamente por meio do domínio admissível dos coeficientes.

Para cada valor de $\\alpha$, o experimento é repetido sobre diversos conjuntos de dados independentes, e o modelo ajustado é avaliado sobre um conjunto fixo de pontos de teste $\{x^{(j)}\}_{j=1}^{m}$ não utilizados no treinamento. A partir das predições dos diferentes modelos em cada ponto, calculam-se o viés quadrático, como o quadrado da diferença entre a média das predições e o valor verdadeiro de $f(x)$, a variância empírica das predições, e o erro irredutível, correspondente à variância do ruído aditivo previamente definido.

À medida que variamos o parâmetro $\\alpha$, é possível observar como a capacidade expressiva do modelo influencia a decomposição do erro. Para $\\alpha$ muito pequeno, as restrições são rígidas mesmo nos coeficientes de baixa ordem, resultando em modelos com baixo grau de flexibilidade e, portanto, alto viés. Já para $\\alpha$ próximo de 1, os modelos são menos restritos e podem se ajustar demais às particularidades dos dados ruidosos, levando a alta variância. O ponto ideal ocorre quando há um equilíbrio entre essas duas fontes de erro, minimizando o erro total de generalização.

<p align="center">
<img src="https://raw.githubusercontent.com/1moi6/minicurso_ia/refs/heads/main/assets/images/decomposicao_erro_experimental.png" width="500"/>
</p>

Este experimento fornece uma interpretação geométrica e estatística clara do papel dos hiperparâmetros e das **restrições explícitas** na regulação da complexidade de modelos polinomiais. Ele ilustra como o viés e a variância podem ser modulados ao se restringir diretamente o espaço de busca dos parâmetros, oferecendo assim uma perspectiva sobre as estratégias de regularização em aprendizado supervisionado.
        </div> '''
#         '''<div align="justify">
  
#   Podemos visualizar empiricamente o comportamento dos erros de generalização de um modelo de regressão supervisionada por meio de um experimento que simula diferentes níveis de complexidade indutiva. Para isso, consideramos como função alvo a ser aprendida a expressão não linear e suave:

# $$
# f(x) = 2x + \cos(4\pi x), \quad x \in [0,1].
# $$

# O objetivo é construir modelos aproximadores dessa função a partir de amostras com ruído e avaliar, para diferentes níveis de complexidade, os erros de viés, variância e ruído irreducível que compõem o erro quadrático médio esperado.

# Para aproximar $ f(x) $, usamos uma família de modelos polinomiais da forma:

# $$
# f_\\alpha(x) = \sum_{k=0}^{d} a_k x^k,
# $$

# onde os coeficientes $ a_k $ são aprendidos a partir de dados sintéticos gerados com ruído aditivo gaussiano. A complexidade do modelo é controlada por meio de uma **regularização suave** imposta diretamente sobre os coeficientes, com o intuito de penalizar fortemente termos de alta ordem. Essa regularização é implementada por uma estrutura de decaimento exponencial, penalizando coeficientes segundo:

# $$
# \\text{Penalidade:} \quad \sum_{k=0}^{d} \left( \\frac{a_k}{\\alpha^k} \\right)^2,
# $$

# onde $ \\alpha \in (0,1] $ é o **parâmetro de regularização** que controla a complexidade do modelo, e $ M $ é um hiperparâmetro multiplicativo fixo. Valores menores de $ \\alpha $ impõem maior decaimento e restrição à magnitude dos coeficientes de ordem elevada, resultando em modelos mais suaves e de baixa capacidade. Por outro lado, valores de $ \\alpha $ mais próximos de 1 permitem que o modelo expresse maior variação e detalhes locais, o que pode levar ao sobreajuste.

# A cada iteração do experimento, geramos um conjunto de dados de treinamento $ \{(x_i, y_i)\}_{i=1}^n $, com $ x_i \sim \mathcal{U}(0,1) $ e $ y_i = f(x_i) + \\varepsilon_i $, onde $ \\varepsilon_i \sim \mathcal{N}(0, \sigma^2) $. O modelo $ f_\\alpha $ é então ajustado minimizando a seguinte função de perda regularizada:

# $$
# \min_{a_0,\ldots,a_d} \left\{ \\frac{1}{n} \sum_{i=1}^{n} \left(y_i - \sum_{k=0}^{d} a_k x_i^k \\right)^2 + \lambda \sum_{k=0}^{d} \left( \\frac{a_k}{\\alpha^k} \\right)^2 \\right\},
# $$

# onde $ \lambda $ é um parâmetro que controla a força da penalização. Essa minimização é realizada para diferentes valores de $ \\alpha $, mantendo fixos o número de amostras, a variância do ruído, e o grau máximo do polinômio. Para cada $ \\alpha $, o experimento é repetido sobre diversos conjuntos de dados distintos, e as predições do modelo são avaliadas em um conjunto fixo de pontos $ \{x^{(j)}\}_{j=1}^{m} $ não vistos no treinamento. A partir dessas predições, calcula-se o viés quadrático como a média do quadrado da diferença entre a predição média e o valor verdadeiro de $ f(x) $, a variância como a variância empírica das predições ao longo dos datasets, e o ruído como a variância do erro aditivo conhecido.

# A variação do parâmetro $ \\alpha $ permite então observar como a capacidade do modelo influencia o erro de generalização. Espera-se que, para valores muito pequenos de $ \\alpha $, o modelo tenha baixa variância, mas alto viés, devido à incapacidade de capturar adequadamente a oscilação da função alvo — caracterizando um regime de underfitting. À medida que $ \\alpha $ cresce, o viés tende a diminuir, mas a variância pode aumentar sensivelmente, indicando sobreajuste ao ruído dos dados. O valor ótimo de $ \\alpha $ é aquele que minimiza a soma total dos erros, equilibrando a capacidade de generalização com a fidelidade à função verdadeira.

# <p align="center">
# <img src="https://raw.githubusercontent.com/1moi6/minicurso_ia/refs/heads/main/assets/images/decomposicao_erro_experimental.png" width="500"/>
# </p>


# Esse experimento ilustra de maneira clara o papel dos hiperparâmetros na regulação da complexidade de modelos de aprendizado e fornece uma ferramenta empírica útil para compreender a compensação entre viés e variância, que está no cerne da teoria estatística do aprendizado de máquina.


#   </div>
#   '''

    },
    {'titulo': "Métricas de Avaliação", 'conteudo': '''<div align="justify">

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
    {'titulo': "Regressões Linear e Logística", 'conteudo': '''<strong>2.3.1 Regressão Linear e Logística</strong> 

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


resumo = '''### Resumo

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

introducao = '''
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


redes_neurais = [
    {

        'titulo': 'Apresentação',
        'conteudo': '''
    
    <div align="justify">
    
 As **Redes Neurais Artificiais (RNAs)** representam um dos pilares centrais do aprendizado profundo e consistem em modelos matemáticos inspirados, de forma abstrata, na organização e funcionamento dos neurônios biológicos. Tais redes são compostas por unidades computacionais chamadas **neurônios artificiais**, dispostas em **camadas** e interconectadas por **pesos sinápticos** ajustáveis. A capacidade das redes neurais em aprender padrões complexos reside na composição sistemática de operações lineares e não lineares, formando um arcabouço computacional altamente expressivo.

Neste capítulo, exploraremos a arquitetura fundamental das RNAs com ênfase em seus componentes essenciais:

- A estrutura matemática do **neurônio artificial**, responsável por transformar vetores de entrada em ativações, utilizando combinações lineares seguidas por funções não-lineares denominadas **funções de ativação**;
- A organização dos neurônios em **camadas densas** (ou fully connected), em que cada unidade se conecta a todas as unidades da camada subsequente;
- O conceito de **profundidade** de uma rede, que está diretamente associado ao número de camadas ocultas e influencia sua capacidade de representação e generalização;
- O processo de **propagação direta (feedforward)**, responsável pelo fluxo de informação da entrada até a saída da rede, sem recursividade;
- A **fase de treinamento supervisionado**, na qual os parâmetros da rede (pesos e vieses) são atualizados iterativamente com base na minimização de uma função de custo, frequentemente por meio de algoritmos de otimização baseados em gradiente, como o método do gradiente descendente estocástico (SGD);
- O papel das **funções de ativação**, que quebram a linearidade das operações matriciais e permitem que a rede aprenda representações não triviais dos dados.

Ao longo do capítulo, abordaremos ainda aspectos formais do funcionamento das redes neurais, incluindo sua representação matricial, análise vetorial da propagação de sinais, e os fundamentos diferenciais envolvidos na retropropagação do erro. A apresentação será orientada por exemplos simples, acompanhada por visualizações e expressões matemáticas rigorosas, com vistas a desenvolver uma compreensão conceitual sólida e analítica do comportamento dessas redes.

Este capítulo não se propõe apenas a apresentar as RNAs como uma ferramenta computacional, mas a examinar seus fundamentos com o rigor matemático apropriado ao público da matemática aplicada, evidenciando como álgebra linear, cálculo diferencial e estatística convergem para formar a base teórica do aprendizado profundo.
    </div>

    '''
    },

    {
        'titulo': 'Neurônios Artificiais',
        'conteudo': '''<div align="justify">

Os **neurônios artificiais** constituem os blocos fundamentais das redes neurais profundas. Inspirados de forma abstrata nos neurônios biológicos, cada unidade computacional realiza uma transformação dos dados de entrada por meio de uma **operação afim**, seguida da aplicação de uma **função de ativação não-linear**. Essa composição simples é suficiente para tornar a rede capaz de aprender aproximações de funções complexas, desde que parametrizada e treinada adequadamente.

Considere uma entrada vetorial $\mathbf{x} \in \mathbb{R}^n$ e um conjunto de parâmetros (pesos sinápticos) $\mathbf{w} \in \mathbb{R}^n$ e um viés escalar $b \in \mathbb{R}$. Um **neurônio artificial** é uma operação matemática definidas pelas duas etapas seguintes:

1. Cálculo da pré-ativação (combinação linear):

$$
z = \mathbf{w}^\\top \mathbf{x} + b
$$

2. Aplicação da função de ativação $f$:

$$
a = f(z)
$$

em que $a$ representa a saída do neurônio. O papel da função $f$ é introduzir **não-linearidade**, permitindo que a rede aprenda representações que não poderiam ser capturadas por modelos lineares.


A **função de ativação** $f$ é aplicada ponto a ponto e desempenha um papel central na capacidade da rede de modelar funções não lineares. Sem essas funções, a composição de várias camadas seria ainda uma função linear do dado de entrada, limitando drasticamente a expressividade do modelo.

A  função de ativação **sigmoide** mapeia números reais para o intervalo aberto $(0, 1)$, sendo útil para representar probabilidades, por meio da transformação:

$$
f(z) = \\frac{1}{1 + e^{-z}}
$$

Embora tenha sido amplamente utilizada, especialmente em saídas de classificação binária, apresenta **regiões de saturação** onde o gradiente se torna muito pequeno, dificultando o aprendizado em redes profundas.


Já a função de ativação **ReLU** é definida por:

$$
f(z) = \max(0, z)
$$

Trata-se de uma função **linear por partes**, amplamente adotada devido à sua eficiência computacional e à sua capacidade de manter gradientes significativos em grande parte do domínio. No entanto, para valores negativos de $z$, a saída é nula, o que pode levar a **neurônios inativos** durante o treinamento.

A função de ativação **Leaky ReLU** propõe uma pequena inclinação no ramo negativo para evitar a inatividade total:

$$
f(z) = \left\{
  \\begin{array}{ll}
    z & \\text{se } z > 0 \\\\
    \\alpha z & \\text{se } z \leq 0
  \end{array}
\\right.
$$

<div align="center">
<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="550" height="331"><defs><clipPath id="etuuloHknQqk"><path fill="none" stroke="none" d=" M 0 0 L 1070 0 L 1070 662 L 0 662 L 0 0 Z"/></clipPath></defs><g transform="scale(0.5,0.5)" clip-path="url(#etuuloHknQqk)"><g><rect fill="rgb(255,255,255)" stroke="none" x="0" y="0" width="1070" height="662" fill-opacity="1"/><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 536.5 2.5 L 536.5 662.5" stroke-opacity="1" stroke-miterlimit="10"/><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 536.5 1.5 L 532.5 5.5" stroke-opacity="1" stroke-miterlimit="10"/><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 536.5 1.5 L 540.5 5.5" stroke-opacity="1" stroke-miterlimit="10"/><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 0.5 594.5 L 1068.5 594.5" stroke-opacity="1" stroke-miterlimit="10"/><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 1069.5 594.5 L 1065.5 590.5" stroke-opacity="1" stroke-miterlimit="10"/><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 1069.5 594.5 L 1065.5 598.5" stroke-opacity="1" stroke-miterlimit="10"/><text fill="rgb(37,37,37)" stroke="none" font-family="geogebra-sans-serif, sans-serif" font-size="12px" font-style="normal" font-weight="normal" text-decoration="normal" x="77" y="610" text-anchor="start" dominant-baseline="alphabetic" fill-opacity="1">–10</text><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 85.5 594.5 L 85.5 597.5" stroke-opacity="1" stroke-miterlimit="10"/><text fill="rgb(37,37,37)" stroke="none" font-family="geogebra-sans-serif, sans-serif" font-size="12px" font-style="normal" font-weight="normal" text-decoration="normal" x="170" y="610" text-anchor="start" dominant-baseline="alphabetic" fill-opacity="1">–8</text><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 175.5 594.5 L 175.5 597.5" stroke-opacity="1" stroke-miterlimit="10"/><text fill="rgb(37,37,37)" stroke="none" font-family="geogebra-sans-serif, sans-serif" font-size="12px" font-style="normal" font-weight="normal" text-decoration="normal" x="260" y="610" text-anchor="start" dominant-baseline="alphabetic" fill-opacity="1">–6</text><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 265.5 594.5 L 265.5 597.5" stroke-opacity="1" stroke-miterlimit="10"/><text fill="rgb(37,37,37)" stroke="none" font-family="geogebra-sans-serif, sans-serif" font-size="12px" font-style="normal" font-weight="normal" text-decoration="normal" x="351" y="610" text-anchor="start" dominant-baseline="alphabetic" fill-opacity="1">–4</text><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 356.5 594.5 L 356.5 597.5" stroke-opacity="1" stroke-miterlimit="10"/><text fill="rgb(37,37,37)" stroke="none" font-family="geogebra-sans-serif, sans-serif" font-size="12px" font-style="normal" font-weight="normal" text-decoration="normal" x="441" y="610" text-anchor="start" dominant-baseline="alphabetic" fill-opacity="1">–2</text><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 446.5 594.5 L 446.5 597.5" stroke-opacity="1" stroke-miterlimit="10"/><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 536.5 594.5 L 536.5 597.5" stroke-opacity="1" stroke-miterlimit="10"/><text fill="rgb(37,37,37)" stroke="none" font-family="geogebra-sans-serif, sans-serif" font-size="12px" font-style="normal" font-weight="normal" text-decoration="normal" x="624" y="610" text-anchor="start" dominant-baseline="alphabetic" fill-opacity="1">2</text><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 626.5 594.5 L 626.5 597.5" stroke-opacity="1" stroke-miterlimit="10"/><text fill="rgb(37,37,37)" stroke="none" font-family="geogebra-sans-serif, sans-serif" font-size="12px" font-style="normal" font-weight="normal" text-decoration="normal" x="715" y="610" text-anchor="start" dominant-baseline="alphabetic" fill-opacity="1">4</text><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 717.5 594.5 L 717.5 597.5" stroke-opacity="1" stroke-miterlimit="10"/><text fill="rgb(37,37,37)" stroke="none" font-family="geogebra-sans-serif, sans-serif" font-size="12px" font-style="normal" font-weight="normal" text-decoration="normal" x="805" y="610" text-anchor="start" dominant-baseline="alphabetic" fill-opacity="1">6</text><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 807.5 594.5 L 807.5 597.5" stroke-opacity="1" stroke-miterlimit="10"/><text fill="rgb(37,37,37)" stroke="none" font-family="geogebra-sans-serif, sans-serif" font-size="12px" font-style="normal" font-weight="normal" text-decoration="normal" x="895" y="610" text-anchor="start" dominant-baseline="alphabetic" fill-opacity="1">8</text><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 897.5 594.5 L 897.5 597.5" stroke-opacity="1" stroke-miterlimit="10"/><text fill="rgb(37,37,37)" stroke="none" font-family="geogebra-sans-serif, sans-serif" font-size="12px" font-style="normal" font-weight="normal" text-decoration="normal" x="982" y="610" text-anchor="start" dominant-baseline="alphabetic" fill-opacity="1">10</text><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 987.5 594.5 L 987.5 597.5" stroke-opacity="1" stroke-miterlimit="10"/><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 533.5 504.5 L 536.5 504.5" stroke-opacity="1" stroke-miterlimit="10"/><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 533.5 414.5 L 536.5 414.5" stroke-opacity="1" stroke-miterlimit="10"/><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 533.5 324.5 L 536.5 324.5" stroke-opacity="1" stroke-miterlimit="10"/><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 533.5 233.5 L 536.5 233.5" stroke-opacity="1" stroke-miterlimit="10"/><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 533.5 143.5 L 536.5 143.5" stroke-opacity="1" stroke-miterlimit="10"/><path fill="none" stroke="rgb(37,37,37)" paint-order="fill stroke markers" d=" M 533.5 53.5 L 536.5 53.5" stroke-opacity="1" stroke-miterlimit="10"/><text fill="rgb(37,37,37)" stroke="none" font-family="geogebra-sans-serif, sans-serif" font-size="12px" font-style="normal" font-weight="normal" text-decoration="normal" x="512" y="509" text-anchor="start" dominant-baseline="alphabetic" fill-opacity="1">0.2</text><text fill="rgb(37,37,37)" stroke="none" font-family="geogebra-sans-serif, sans-serif" font-size="12px" font-style="normal" font-weight="normal" text-decoration="normal" x="512" y="419" text-anchor="start" dominant-baseline="alphabetic" fill-opacity="1">0.4</text><text fill="rgb(37,37,37)" stroke="none" font-family="geogebra-sans-serif, sans-serif" font-size="12px" font-style="normal" font-weight="normal" text-decoration="normal" x="512" y="329" text-anchor="start" dominant-baseline="alphabetic" fill-opacity="1">0.6</text><text fill="rgb(37,37,37)" stroke="none" font-family="geogebra-sans-serif, sans-serif" font-size="12px" font-style="normal" font-weight="normal" text-decoration="normal" x="512" y="238" text-anchor="start" dominant-baseline="alphabetic" fill-opacity="1">0.8</text><text fill="rgb(37,37,37)" stroke="none" font-family="geogebra-sans-serif, sans-serif" font-size="12px" font-style="normal" font-weight="normal" text-decoration="normal" x="522" y="148" text-anchor="start" dominant-baseline="alphabetic" fill-opacity="1">1</text><text fill="rgb(37,37,37)" stroke="none" font-family="geogebra-sans-serif, sans-serif" font-size="12px" font-style="normal" font-weight="normal" text-decoration="normal" x="512" y="58" text-anchor="start" dominant-baseline="alphabetic" fill-opacity="1">1.2</text><text fill="rgb(37,37,37)" stroke="none" font-family="geogebra-sans-serif, sans-serif" font-size="12px" font-style="normal" font-weight="normal" text-decoration="normal" x="522" y="610" text-anchor="start" dominant-baseline="alphabetic" fill-opacity="1">0</text><path fill="none" stroke="rgb(255,0,51)" paint-order="fill stroke markers" d=" M 0 594.7239405482801 L 4.179687500000114 594.7236421442072 L 8.359375 594.7233147688298 L 12.5390625 594.7229556094447 L 16.718750000000114 594.7225615802853 L 20.8984375 594.7221292960131 L 25.078125000000057 594.7216550426382 L 29.257812500000057 594.7211347456155 L 33.4375 594.7205639348459 L 37.61718750000006 594.7199377062807 L 41.79687500000006 594.7192506797984 L 45.9765625 594.7184969529965 L 50.15625000000006 594.7176700504963 L 54.3359375 594.7167628683311 L 58.515625 594.715767612937 L 62.69531250000006 594.7146757342254 L 66.875 594.7134778521615 L 71.0546875 594.7121636762221 L 75.23437500000006 594.7107219170388 L 79.4140625 594.7091401894731 L 83.59375000000006 594.7074049062909 L 87.77343750000006 594.7055011615256 L 91.953125 594.7034126025346 L 96.13281250000006 594.7011212896494 L 100.31250000000006 594.6986075422232 L 104.4921875 594.6958497697561 L 108.67187500000006 594.6928242866582 L 112.85156250000006 594.6895051090642 L 117.03125 594.6858637319688 L 121.21093750000006 594.6818688847782 L 125.39062500000006 594.6774862631964 L 129.5703125 594.6726782351594 L 133.75000000000006 594.6674035183161 L 137.92968750000006 594.6616168263121 L 142.109375 594.6552684808706 L 146.28906250000006 594.6483039863803 L 150.46875000000006 594.64066356338 L 154.6484375 594.6322816369963 L 158.82812500000006 594.6230862760044 L 163.00781250000006 594.6129985777819 L 167.1875 594.601931993972 L 171.36718750000006 594.5897915911855 L 175.54687500000006 594.5764732405405 L 179.7265625 594.5618627292523 L 183.90625000000006 594.5458347868587 L 188.08593750000006 594.52825201797 L 192.265625 594.5089637326878 L 196.44531250000006 594.4878046650177 L 200.62500000000006 594.4645935687145 L 204.8046875 594.439131679037 L 208.98437500000006 594.4112010278509 L 213.16406250000006 594.3805625983817 L 217.34375 594.3469543047105 L 221.52343750000006 594.3100887797817 L 225.70312500000006 594.2696509542803 L 229.8828125 594.2252954072137 L 234.06250000000006 594.1766434673982 L 238.24218750000006 594.1232800433121 L 242.421875 594.064750156912 L 246.6015625 594.0005551550444 L 250.78125000000006 593.930148569989 L 254.9609375 593.8529315984874 L 259.140625 593.7682481663071 L 263.32031250000006 593.6753795430149 L 267.50000000000006 593.5735384691793 L 271.6796875 593.4618627557224 L 275.85937500000006 593.3394083126311 L 280.03906250000006 593.2051415617499 L 284.21875 593.0579311859811 L 288.39843750000006 592.8965391649652 L 292.578125 592.7196110453006 L 296.7578125 592.5256653917021 L 300.9375 592.3130823643023 L 305.1171875 592.0800913667535 L 309.296875 591.8247577100739 L 313.4765625 591.5449682385386 L 317.65625 591.2384158666334 L 321.8359375 590.902582980494 L 326.015625 590.5347236637556 L 330.1953125 590.1318447168097 L 334.375 589.6906854506345 L 338.5546875 589.2076962523023 L 342.734375 588.6790159396661 L 346.9140625 588.1004479484545 L 351.09375 587.4674354269617 L 355.2734375 586.7750353527858 L 359.453125 586.0178918337542 L 363.6328125 585.1902088125358 L 367.81250000000006 584.2857224627563 L 371.9921875 583.2976736450308 L 376.171875 582.2187808855084 L 380.35156250000006 581.0412144484602 L 384.53125 579.756572199121 L 388.7109375 578.3558580940221 L 392.89062500000006 576.8294642935153 L 397.0703125 575.1671580643822 L 401.25 573.3580748276167 L 405.42968750000006 571.3907189045428 L 409.609375 569.2529737184709 L 413.7890625 566.9321234120572 L 417.96875000000006 564.4148880326612 L 422.1484375 561.6874746065259 L 426.328125 558.7356465512731 L 430.50781250000006 555.5448139450615 L 434.6875 552.1001471560875 L 438.8671875 548.3867162107488 L 443.04687500000006 544.3896580128403 L 447.2265625 540.0943730883412 L 451.40625 535.4867528901034 L 455.58593750000006 530.5534378271155 L 459.765625 525.2821050644852 L 463.9453125 519.6617837654351 L 468.12500000000006 513.6831938254031 L 472.3046875 507.33910231355145 L 476.48437500000006 500.62468984933827 L 480.6640625 493.5379170936011 L 484.84375 486.0798795501754 L 489.02343750000006 478.2551371116031 L 493.203125 470.07200342119097 L 497.3828125 461.54277935588493 L 501.56250000000006 452.6839149470518 L 505.7421875 443.5160850089803 L 509.921875 434.0641657456257 L 514.1015625 424.35710268679543 L 518.28125 414.42766440192713 L 520.37109375 409.3907535144846 L 522.4609375 404.31208138213685 L 524.55078125 399.1966404228107 L 526.640625 394.0495749924247 L 528.73046875 388.8761614572712 L 530.8203125 383.68178710816926 L 532.91015625 378.47192808998966 L 535 373.2521265350199 L 537.08984375 368.0279671008112 L 539.1796875 362.8050531223405 L 541.26953125 357.5889825943052 L 543.359375 352.38532420201466 L 545.44921875 347.19959361858855 L 547.5390625 342.0372302820249 L 549.62890625 336.90357485826695 L 551.71875 331.8038475858585 L 553.80859375 326.7431276843619 L 555.8984375 321.7263339927426 L 560.078125 311.84329229650825 L 564.2578125 302.19022070465974 L 568.4375 292.79906550062367 L 572.6171875 283.69788521799387 L 576.796875 274.9106299725361 L 580.9765625 266.4570341839247 L 585.15625 258.3526161442225 L 589.3359375 250.60877400808153 L 593.515625 243.23296493703936 L 597.6953125 236.22895236824417 L 601.875 229.59710564564114 L 606.0546875 223.33473642756536 L 610.234375 217.43645720102944 L 614.4140625 211.89454869984388 L 618.59375 206.69932485016335 L 622.7734375 201.83948587734847 L 626.953125 197.30245225245073 L 631.1328125 193.07467411714612 L 635.3125 189.14191261745282 L 639.4921875 185.48949114498214 L 643.671875 182.1025158025269 L 647.8515625 178.96606547287178 L 652.03125 176.06535268636065 L 656.2109375 173.3858570755117 L 660.390625 170.91343360189148 L 664.5703125 168.63439797248253 L 668.75 166.53559176104636 L 672.9296875 164.60442974385325 L 677.109375 162.82893187508915 L 681.2890625 161.1977421881312 L 685.46875 159.70013673383397 L 689.6484375 158.32602247143916 L 693.828125 157.06592882378266 L 698.0078125 155.9109934052267 L 702.1875 154.85294323473607 L 706.3671875 153.88407256221967 L 710.546875 152.99721826644895 L 714.7265625 152.18573362902765 L 718.90625 151.44346115151995 L 723.0859375 150.76470496173425 L 727.265625 150.1442032495915 L 731.4453125 149.57710108195596 L 735.625 149.05892386804982 L 739.8046875 148.5855516813039 L 743.984375 148.1531945883981 L 748.1640625 147.7583690905309 L 752.34375 147.39787574441425 L 756.5234375 147.06877799999864 L 760.703125 146.7683822674719 L 764.8828125 146.4942192067324 L 769.0625 146.2440262175004 L 773.2421875 146.01573109679902 L 777.421875 145.80743682208106 L 781.6015625 145.61740741228118 L 785.78125 145.44405481507277 L 789.9609375 145.28592676622213 L 794.140625 145.14169556583704 L 798.3203125 145.0101477162238 L 802.5 144.89017436677437 L 806.6796875 144.78076251261325 L 810.859375 144.68098689548754 L 815.0390625 144.5900025574556 L 819.21875 144.50703800022086 L 823.3984375 144.43138890538307 L 827.578125 144.36241237337146 L 831.7578125 144.29952164134028 L 835.9375 144.24218124279543 L 840.1171875 144.18990257416033 L 844.296875 144.14223983585413 L 848.4765625 144.09878631772443 L 852.65625 144.0591710008527 L 856.8359375 144.0230554498052 L 861.015625 143.99013097135406 L 865.1953125 143.96011601752434 L 869.375 143.9327538125416 L 873.5546875 143.90781018486206 L 877.734375 143.88507158696456 L 881.9140625 143.86434328697868 L 886.09375 143.84544771751524 L 890.2734375 143.82822296826424 L 894.453125 143.81252141003597 L 898.6328125 143.79820843894117 L 902.8125 143.78516133035424 L 906.9921875 143.77326819317238 L 911.171875 143.76242701568367 L 915.3515625 143.75254479509596 L 919.53125 143.7435367434535 L 923.7109375 143.73532556329042 L 927.890625 143.72784078694042 L 932.0703125 143.7210181739444 L 936.25 143.71479916147757 L 940.4296875 143.70913036315562 L 944.609375 143.70396311198238 L 948.7890624999999 143.69925304356804 L 952.96875 143.69495971608444 L 957.1484375 143.6910462637307 L 961.3281249999999 143.687479080765 L 965.5078125 143.68422753341366 L 969.6875 143.68126369720557 L 973.8671875 143.6785621174938 L 978.046875 143.676099591121 L 982.2265625 143.67385496736767 L 986.40625 143.67180896648136 L 990.5859375 143.66994401423858 L 994.765625 143.668244091124 L 998.9453125 143.66669459483762 L 1003.125 143.6652822149539 L 1007.3046875 143.66399481865983 L 1011.484375 143.6628213465939 L 1015.6640625 143.66175171789365 L 1019.84375 143.6607767436389 L 1024.0234375 143.6598880479487 L 1028.203125 143.65907799605532 L 1032.3828125 143.6583396287395 L 1036.5625 143.65766660256457 L 1040.7421875 143.6570531353957 L 1044.921875 143.6564939567392 L 1049.1015625 143.6559842624739 L 1053.28125 143.65551967358704 L 1057.4609375 143.6550961985604 L 1061.640625 143.65471019908426 L 1065.8203125 143.65435835880442 L 1070 143.65403765483387" stroke-opacity="0.8" stroke-linecap="round" stroke-linejoin="round" stroke-miterlimit="10" stroke-width="6.5"/><text fill="rgb(255,0,51)" stroke="none" font-family="geogebra-sans-serif, sans-serif" font-size="16px" font-style="normal" font-weight="normal" text-decoration="normal" x="8" y="594" text-anchor="start" dominant-baseline="alphabetic" fill-opacity="1">f</text></g></g></svg>
</div>

comumente, $\\alpha = 0{,}01$.

<p align="center">
<img src="https://raw.githubusercontent.com/1moi6/minicurso_ia/refs/heads/main/assets/images/ativacao.png" width="500"/>
</p>

Finalmente, a função **softmax** é usada na **camada de saída** em tarefas de **classificação multiclasse**, convertendo um vetor de valores reais $(z_1, \dots, z_K)$ em uma **distribuição de probabilidade** sobre $K$ classes:

$$
\\text{softmax}(z_i) = \\frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$

Essa função assegura que cada saída esteja no intervalo $(0,1)$ e que a soma total das saídas seja igual a 1. Em termos computacionais, a softmax permite que o modelo represente **incerteza** sobre múltiplas classes, ao mesmo tempo em que mantém uma estrutura compatível com o treinamento por **máxima verossimilhança**.

As funções de ativação determinam o comportamento dinâmico das redes neurais e influenciam diretamente a **propagação do gradiente** durante o treinamento. A escolha da função ideal depende do contexto da tarefa e da profundidade da rede. Em termos práticos:

- **ReLU e suas variantes** são preferidas nas **camadas ocultas** devido à sua simplicidade e eficácia.
- **Sigmoide e softmax** são mais apropriadas para **camadas de saída**, especialmente em problemas supervisionados com interpretação probabilística.

Combinadas com transformações lineares, essas funções de ativação compõem o alicerce do aprendizado profundo, fornecendo os mecanismos pelos quais redes neurais conseguem modelar fenômenos altamente não-lineares.

</div>

'''
    },
    {
        'titulo': 'Redes de Camadas Densas',
        'conteudo': '''<div align="justify">

As **Redes Neurais Multicamadas** (em inglês, *Dense Feedforward Neural Networks*), também conhecidas como **Perceptrons Multicamadas** (*Multilayer Perceptrons*, ou MLPs), constituem uma das arquiteturas mais fundamentais do aprendizado profundo. Essas redes realizam transformações sucessivas sobre os dados por meio de camadas de neurônios interconectados, nas quais a informação flui de uma camada entrada pelas camadas intermediárias até a camada de saída, sem ciclos ou realimentações.

A estrutura básica de uma MLP é composta por três blocos principais:

1. Uma **camada de entrada** (*input layer*), que recebe o vetor de características dos dados;
2. Uma ou mais **camadas ocultas** (*hidden layers*), responsáveis pela construção de representações internas por meio de transformações parametrizadas;
3. Uma **camada de saída** (*output layer*), que fornece a predição final da rede.

<div align="center">
<svg xmlns="http://www.w3.org/2000/svg" style="cursor: move;" width="700" height="474"><g transform="translate(-770,-474) scale(1.0562276533121615)"><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(128, 128, 255); fill: none;" marker-end="" d="M768.6666666666666,611C863.1666666666666,611 863.1666666666666,489.5 957.6666666666666,489.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(255, 143, 143); fill: none;" marker-end="" d="M768.6666666666666,611C863.1666666666666,611 863.1666666666666,570.5 957.6666666666666,570.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(203, 203, 255); fill: none;" marker-end="" d="M768.6666666666666,611C863.1666666666666,611 863.1666666666666,651.5 957.6666666666666,651.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(152, 152, 255); fill: none;" marker-end="" d="M768.6666666666666,692C863.1666666666666,692 863.1666666666666,489.5 957.6666666666666,489.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(130, 130, 255); fill: none;" marker-end="" d="M768.6666666666666,692C863.1666666666666,692 863.1666666666666,570.5 957.6666666666666,570.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(149, 149, 255); fill: none;" marker-end="" d="M768.6666666666666,692C863.1666666666666,692 863.1666666666666,651.5 957.6666666666666,651.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(255, 192, 192); fill: none;" marker-end="" d="M957.6666666666666,489.5C1052.1666666666667,489.5 1052.1666666666667,573.5 1146.6666666666667,573.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(62, 62, 255); fill: none;" marker-end="" d="M957.6666666666666,489.5C1052.1666666666667,489.5 1052.1666666666667,651.5 1146.6666666666667,651.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(255, 54, 54); fill: none;" marker-end="" d="M957.6666666666666,570.5C1052.1666666666667,570.5 1052.1666666666667,573.5 1146.6666666666667,573.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(159, 159, 255); fill: none;" marker-end="" d="M957.6666666666666,570.5C1052.1666666666667,570.5 1052.1666666666667,651.5 1146.6666666666667,651.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(71, 71, 255); fill: none;" marker-end="" d="M957.6666666666666,651.5C1052.1666666666667,651.5 1052.1666666666667,573.5 1146.6666666666667,573.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(117, 117, 255); fill: none;" marker-end="" d="M957.6666666666666,651.5C1052.1666666666667,651.5 1052.1666666666667,651.5 1146.6666666666667,651.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(255, 157, 157); fill: none;" marker-end="" d="M1146.6666666666667,573.5C1241.1666666666667,573.5 1241.1666666666667,651.5 1335.6666666666667,651.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(116, 116, 255); fill: none;" marker-end="" d="M1146.6666666666667,651.5C1241.1666666666667,651.5 1241.1666666666667,651.5 1335.6666666666667,651.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(46, 46, 255); fill: none;" marker-end="" d="M768.6666666666666,611C863.1666666666666,611 863.1666666666666,732.5 957.6666666666666,732.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(9, 9, 255); fill: none;" marker-end="" d="M768.6666666666666,611C863.1666666666666,611 863.1666666666666,813.5 957.6666666666666,813.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(255, 72, 72); fill: none;" marker-end="" d="M768.6666666666666,692C863.1666666666666,692 863.1666666666666,732.5 957.6666666666666,732.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(255, 13, 13); fill: none;" marker-end="" d="M768.6666666666666,692C863.1666666666666,692 863.1666666666666,813.5 957.6666666666666,813.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(180, 180, 255); fill: none;" marker-end="" d="M957.6666666666666,732.5C1052.1666666666667,732.5 1052.1666666666667,573.5 1146.6666666666667,573.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(255, 26, 26); fill: none;" marker-end="" d="M957.6666666666666,732.5C1052.1666666666667,732.5 1052.1666666666667,651.5 1146.6666666666667,651.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(99, 99, 255); fill: none;" marker-end="" d="M957.6666666666666,813.5C1052.1666666666667,813.5 1052.1666666666667,573.5 1146.6666666666667,573.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(137, 137, 255); fill: none;" marker-end="" d="M957.6666666666666,813.5C1052.1666666666667,813.5 1052.1666666666667,651.5 1146.6666666666667,651.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(255, 102, 102); fill: none;" marker-end="" d="M957.6666666666666,489.5C1052.1666666666667,489.5 1052.1666666666667,729.5 1146.6666666666667,729.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(67, 67, 255); fill: none;" marker-end="" d="M957.6666666666666,570.5C1052.1666666666667,570.5 1052.1666666666667,729.5 1146.6666666666667,729.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(255, 80, 80); fill: none;" marker-end="" d="M957.6666666666666,651.5C1052.1666666666667,651.5 1052.1666666666667,729.5 1146.6666666666667,729.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(30, 30, 255); fill: none;" marker-end="" d="M957.6666666666666,732.5C1052.1666666666667,732.5 1052.1666666666667,729.5 1146.6666666666667,729.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(236, 236, 255); fill: none;" marker-end="" d="M957.6666666666666,813.5C1052.1666666666667,813.5 1052.1666666666667,729.5 1146.6666666666667,729.5"></path><path class="link" style="stroke-width: 2px; stroke-opacity: 1; stroke: rgb(255, 126, 126); fill: none;" marker-end="" d="M1146.6666666666667,729.5C1241.1666666666667,729.5 1241.1666666666667,651.5 1335.6666666666667,651.5"></path><circle r="14.5" class="node" id="0_0" style="fill: rgb(255, 255, 255); stroke: rgb(51, 51, 51);" cx="768.6666666666666" cy="611"></circle><circle r="14.5" class="node" id="0_1" style="fill: rgb(255, 255, 255); stroke: rgb(51, 51, 51);" cx="768.6666666666666" cy="692"></circle><circle r="14.5" class="node" id="1_0" style="fill: rgb(255, 255, 255); stroke: rgb(51, 51, 51);" cx="957.6666666666666" cy="489.5"></circle><circle r="14.5" class="node" id="1_1" style="fill: rgb(255, 255, 255); stroke: rgb(51, 51, 51);" cx="957.6666666666666" cy="570.5"></circle><circle r="14.5" class="node" id="1_2" style="fill: rgb(255, 255, 255); stroke: rgb(51, 51, 51);" cx="957.6666666666666" cy="651.5"></circle><circle r="14.5" class="node" id="1_3" style="fill: rgb(255, 255, 255); stroke: rgb(51, 51, 51);" cx="957.6666666666666" cy="732.5"></circle><circle r="14.5" class="node" id="1_4" style="fill: rgb(255, 255, 255); stroke: rgb(51, 51, 51);" cx="957.6666666666666" cy="813.5"></circle><circle r="14.5" class="node" id="2_0" style="fill: rgb(255, 255, 255); stroke: rgb(51, 51, 51);" cx="1146.6666666666667" cy="573.5"></circle><circle r="14.5" class="node" id="2_1" style="fill: rgb(255, 255, 255); stroke: rgb(51, 51, 51);" cx="1146.6666666666667" cy="651.5"></circle><circle r="14.5" class="node" id="2_2" style="fill: rgb(255, 255, 255); stroke: rgb(51, 51, 51);" cx="1146.6666666666667" cy="729.5"></circle><circle r="14.5" class="node" id="3_0" style="fill: rgb(255, 255, 255); stroke: rgb(51, 51, 51);" cx="1335.6666666666667" cy="651.5"></circle><text class="text" dy=".35em" style="font-size: 14px;" x="733.6666666666666" y="862.5">Camada de Entrada (ℝ²) </text><text class="text" dy=".35em" style="font-size: 14px;" x="922.6666666666666" y="862.5">Camada Oculta (ℝ⁵)</text><text class="text" dy=".35em" style="font-size: 14px;" x="1111.6666666666667" y="862.5">Camada Oculta (ℝ³)</text><text class="text" dy=".35em" style="font-size: 14px;" x="1320.6666666666667" y="862.5"> Saída (ℝ¹)</text></g><defs><marker id="arrow" viewBox="0 -5 10 10" markerWidth="7" markerHeight="7" orient="auto" refX="52.599999999999994"><path d="M0,-5L10,0L0,5" style="stroke: rgb(80, 80, 80); fill: none;"></path></marker></defs></svg>
</div>

Matematicamente, cada **camada** da rede pode ser interpretada como um **bloco funcional** que transforma um vetor de entrada $\mathbf{a}^{(\ell-1)} \in \mathbb{R}^{m_{\ell-1}}$ em um vetor de saída $\mathbf{a}^{(\ell)} \in \mathbb{R}^{m_\ell}$, por meio de uma transformação afim seguida de uma função ativação:

$$
\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)} \\
\mathbf{a}^{(\ell)} = f^{(\ell)}\left(\mathbf{z}^{(\ell)}\\right)
$$

em que:

1. $\mathbf{W}^{(\ell)} \in \mathbb{R}^{m_\ell \\times m_{\ell-1}}$ é a **matriz de pesos** da camada $\ell$;
2. $\mathbf{b}^{(\ell)} \in \mathbb{R}^{m_\ell}$ é o **vetor de vieses**;
3. $f^{(\ell)}$ é a **função de ativação**, aplicada elemento a elemento.

A propagação da informação ao longo da de camadas sucessivas rede ocorre de forma recursiva:

1. Inicialização da entrada:
   $$
   \mathbf{a}^{(0)} = \mathbf{x}
   $$

2. Para cada camada $\ell = 1, 2, \dots, L$:
   $$
   \mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)} \\
   \mathbf{a}^{(\ell)} = f^{(\ell)}\left(\mathbf{z}^{(\ell)}\\right)
   $$

O vetor final $\hat{\mathbf{y}} = \mathbf{a}^{(L)}$ representa a saída da rede e é utilizado para realizar a tarefa desejada, como por exemplo, realizar uma classificação ou regressão.


Equivalente ao processo recursivo, a MLP pode ser representada como uma **função composta** (*composite function*) de operações parametrizadas:

$$
\hat{\mathbf{y}} = f^{(L)} \circ f^{(L-1)} \circ \cdots \circ f^{(1)} (\mathbf{x})
$$

Essa estrutura modular e hierárquica é o que permite às redes neurais aprenderem **funções complexas**, ajustando seus parâmetros com base em dados de treinamento.

Como mencionado no capítulo anterior, o treinamento de uma MLP é realizado por meio de um processo de **aprendizado supervisionado**, com base em um conjunto de exemplos rotulados. O processo envolve:

1. A definição de uma **função de custo** $\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y})$, que quantifica o erro entre a saída prevista $\hat{\mathbf{y}}$ e o valor verdadeiro $\mathbf{y}$;
2. A aplicação de um **algoritmo de otimização** (como o **gradiente descendente estocástico**, ou *Stochastic Gradient Descent – SGD*) para atualizar os parâmetros da rede;
3. O uso da **retropropagação do erro** (*backpropagation*), um método eficiente para computar os gradientes de $\mathcal{L}$ em relação a todos os pesos e vieses, camada por camada, utilizando a regra da cadeia.

Em síntese, as redes neurais densas de múltiplas camadas constituem uma classe de redes neurais que agem como um transformação de vetores por meio de operações matriciais e aplicações de funções de ativação. Sua arquitetura recursiva e composicional permite representar uma ampla variedade de funções e padrões, fazendo de tais redes uma base conceitual sólida e uma ferramenta prática amplamente utilizada na modelagem matemática de dados.

A seguir, apresentamos com mais detalhes cada um desses componentes fundamentais.       
        </div>'''
    },
  {
    'titulo': 'Treinamento Em Redes de Neurais Artificiais',
    'conteudo': '''<div align="justify">

Como já mencionamos anteriormente, o **treinamento** de uma rede neural consiste em encontrar os valores adequados dos pesos e vieses tendo em vista  a realização de alguma tarefa específica para a rede.
Matematicamente falando, o processo de treinamento nada mais é do que encontrar os valores dos parâmetros da rede que minimizam a distância entre o valor predito pela rede e o valor real observado.        

Para fundamentar essa ideia, consideremos uma rede cuja última camada contém um único neurônio em que a função de perda é: 
1. Para problemas de classificação: 
$$
\mathcal{L}(\hat{y},y) = -y \log \hat{y} - (1-y) \log (1-\hat{y});
$$ 

2. Para problemas de regressão: 

$$
\mathcal{L}(\hat{y},y)=\\frac{1}{2}(y-\hat{y})^2.
$$


##### 1 -  Gradientes da Última Camada

Considere os pesos e viés da camada de saída, indexada por $L$, definidos por:  

$$
\mathbf{W}^{[L]} = (w_1^{[L]}, w_2^{[L]}, \dots, w_{n_{L-1}}^{[L]}), \quad
b^{[L]} \in \mathbb{R}
$$ 
Pela estrutura de redes neurais densas, a última camada recebe como entrada as ativações da camada anterior, indexada por $L-1$, representadas pelo vetor 

$$
\mathbf{a}^{[L-1]} = (a_1^{[L-1]}, a_2^{[L-1]}, \dots, a_{n_{L-1}}^{[L-1]}) \in \mathbb{R}^{n_{L-1}}
$$

Nesse caso, a pré-ativação $z^{[L]}$, ativação da camada de saída $a^{[L]}$ e a predição $\hat{y}$ são dadas respectivamente por:
$$
  z^{[L]} = \sum_{j=1}^{n_{L-1}} w_j^{[L]} a_j^{[L-1]} + b^{[L]}, \qquad a^{[L]} = f^{[L]}(z^{[L]}), \qquad \hat{y} = a^{[L]}
$$
em que $f^{[L]}$ é a função de ativação da camada de saída. No caso de regressão, $f^{[L]}$ é a função identidade, enquanto que em problemas de classificação, $f^{[L]}$ é a função sigmoide.

As derivadas parciais da função de perda em relação aos pesos e ao viés da camada de saída podem ser obtidos por meio da regra da cadeia, que nos permite decompor a derivada em termos de outras derivadas mais simples. 

Assim, a derivada da perda em relação a um peso específico $w_j^{[L]}$ é:

$$
\\frac{\partial \mathcal{L}}{\partial w_j^{[L]}} = 
\\frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot 
\\frac{d a^{[L]}}{d z^{[L]}} \cdot 
\\frac{\partial z^{[L]}}{\partial w_j^{[L]}}
$$

em que:
 
   $$
   \\frac{d a^{[L]}}{d z^{[L]}} = f'^{[L]}(z^{[L]}), \quad
   \\frac{\partial z^{[L]}}{\partial w_j^{[L]}} = a_j^{[L-1]}
   $$

e, portanto,

$$
\\frac{\partial \mathcal{L}}{\partial w_j^{[L]}} = 
\\frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot f'^{[L]}(z^{[L]}) \cdot a_j^{[L-1]}
$$


De forma análoga, a derivada da função de perda $\mathcal{L}$ em relação ao viés $b^{[L]}$ também é obtida pela regra da cadeia:

$$
\\frac{\partial \mathcal{L}}{\partial b^{[L]}} =
\\frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot 
\\frac{d a^{[L]}}{d z^{[L]}} \cdot 
\\frac{\partial z^{[L]}}{\partial b^{[L]}}
$$

Novamente, sabemos que 
$$ 
\dfrac{d a^{[L]}}{d z^{[L]}} = f'^{[L]}(z^{[L]}),\quad\dfrac{\partial z^{[L]}}{\partial b^{[L]}} = 1 
$$
e, portanto, a derivada da função de perda em relação ao viés da camada de saída é:

$$
\\frac{\partial \mathcal{L}}{\partial b^{[L]}} = \\frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot f'^{[L]}(z^{[L]})
$$

Por fim, podemos reescrever o cálculo do gradiente em **forma vetorial**, para obter o gradiente completo em relação à matriz de pesos $ \mathbf{W}^{[L]} \in \mathbb{R}^{1 \\times n_{L-1}} $.
Para isso, consideramos o vetor de **deltas** da camada de saída:

$$ 
\delta^{[L]} = \\frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot f'^{[L]}(z^{[L]}) \in \mathbb{R}
$$

Assim, a derivada da perda em relação à matriz de pesos da última camada é:

$$
\\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[L]}} = \delta^{[L]} \cdot \left( \mathbf{a}^{[L-1]} \\right)^\\top
$$

cujo resultado é um vetor linha de dimensão $ 1 \\times n_{L-1} $, igual à forma de $ \mathbf{W}^{[L]} $.

Analogamente, o gradiente da função de perda em relação ao viés da camada de saída é:

$$
\\frac{\partial \mathcal{L}}{\partial b^{[L]}} = \delta^{[L]}
$$

---

###### Camadas Intermediárias

Nosso objetivo é calcular:

$$
\\frac{\partial \mathcal{L}}{\partial w_{ij}^{[L-1]}}
$$

Lembrando que a rede é composta por uma sequência de transformações, aplicamos a **regra da cadeia** para decompor essa derivada em três fatores:

$$
\\frac{\partial \mathcal{L}}{\partial w_{ij}^{[L-1]}} = 
\\frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot 
\\frac{\partial \hat{y}}{\partial z^{[L]}} \cdot 
\\frac{\partial z^{[L]}}{\partial a_i^{[L-1]}} \cdot 
\\frac{\partial a_i^{[L-1]}}{\partial z_i^{[L-1]}} \cdot 
\\frac{\partial z_i^{[L-1]}}{\partial w_{ij}^{[L-1]}}
$$

Observando que:

1. Erro da predição, como anteriormente definido:
   $$
   \\frac{\partial \mathcal{L}}{\partial \hat{y}} = a^{[L]} - y
   $$

2. Derivada da função de ativação da saída:
   $$
   \\frac{\partial \hat{y}}{\partial z^{[L]}} = f'^{[L]}(z^{[L]})
   $$

3. Derivada da pré-ativação da saída em relação à ativação da camada $ L-1 $:
   $$
   \\frac{\partial z^{[L]}}{\partial a_i^{[L-1]}} = w_i^{[L]}
   $$

4. Derivada da ativação da camada $ L-1$:
   $$
   \\frac{\partial a_i^{[L-1]}}{\partial z_i^{[L-1]}} = f'^{[L-1]}(z_i^{[L-1]})
   $$

5. Derivada da pré-ativação $ z_i^{[L-1]}$ com respeito ao peso $w_{ij}^{[L-1]}$:

   $$
   \\frac{\partial z_i^{[L-1]}}{\partial w_{ij}^{[L-1]}} = a_j^{[L-2]}
   $$

obtemos:

$$
\\frac{\partial \mathcal{L}}{\partial w_{ij}^{[L-1]}} =
(a^{[L]} - y) \cdot f'^{[L]}(z^{[L]}) \cdot w_i^{[L]} \cdot f'^{[L-1]}(z_i^{[L-1]}) \cdot a_j^{[L-2]}
$$


Agora, seja $ b_i^{[L-1]}$ o viés do neurônio $i$ da camada $L-1$. Novamente pela regra da cadeia, temos:

$$
\\frac{\partial \mathcal{L}}{\partial b_i^{[L-1]}} =
(a^{[L]} - y) \cdot f'^{[L]}(z^{[L]}) \cdot w_i^{[L]} \cdot f'^{[L-1]}(z_i^{[L-1]})
$$


Podemos ainda escrever essas operações em forma matricial. Para simplificar a notação, consideramos as seguintes definições:
1. $ \delta^{[L]} := (a^{[L]} - y) \cdot f'^{[L]}(z^{[L]}) \in \mathbb{R} $
2. $ \\boldsymbol{\delta}^{[L-1]} := \delta^{[L]} \cdot \left( \mathbf{w}^{[L]} \odot f'^{[L-1]}(\mathbf{z}^{[L-1]}) \\right) \in \mathbb{R}^{n_{L-1}} $
3. $ \mathbf{a}^{[L-2]} \in \mathbb{R}^{n_{L-2}} $

Daí, temos que as derivadas da função de perda em relação aos pesos $w_{ij}^{[L-1]}$ resultam em:
$$
\\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[L-1]}} = \\boldsymbol{\delta}^{[L-1]} \cdot \left( \mathbf{a}^{[L-2]} \\right)^\\top
$$

e as derivadas da função de perda em relação ao svieses $b^{[L-1]}$ resultam em:
$$
\\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[L-1]}} = \\boldsymbol{\delta}^{[L-1]}
$$

###### Retropropagação do Erro

Por fim, podemos calcular os gradientes para as camadas anteriores, seguindo a mesma lógica de decomposição e aplicação da regra da cadeia. A cada camada, o vetor de deltas é atualizado com base nos pesos e nas ativações da camada subsequente.
Esse procedimento é conhecido como **retropropagação do erro** (*backpropagation*) e é fundamental para o treinamento eficiente de redes neurais profundas. 

Isto é, dada uma rede com camadas $ \ell = 1, 2, \dots, L $, definimos:

- $ \mathbf{z}^{[\ell]} = \mathbf{W}^{[\ell]} \mathbf{a}^{[\ell-1]} + \mathbf{b}^{[\ell]} $
- $ \mathbf{a}^{[\ell]} = f^{[\ell]}(\mathbf{z}^{[\ell]}) $

Para a camada de saída $ L $:

$$
\delta^{[L]} = \\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \odot f'^{[L]}(\mathbf{z}^{[L]})
$$

- Para regressão com ativação linear:
  \( \delta^{[L]} = \mathbf{a}^{[L]} - \mathbf{y} \)

---
Recorrência para $ \ell = L-1, \dots, 1 $

$$
\delta^{[\ell]} = \left( \mathbf{W}^{[\ell+1]^\\top} \delta^{[\ell+1]} \\right) \odot f'^{[\ell]}(\mathbf{z}^{[\ell]})
$$

---

#### 🔹 Passo 3: Cálculo dos Gradientes

- Gradiente dos pesos:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[\ell]}} = \delta^{[\ell]} \cdot \left( \mathbf{a}^{[\ell-1]} \right)^\top
$$

- Gradiente dos vieses:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[\ell]}} = \delta^{[\ell]}
$$

---

Essa recursão permite computar os gradientes de todas as camadas de maneira eficiente e vetorizada.

---

###### Generalização para Múltiplas Amostras

Os resultados acima se referem ao cálculo dos gradientes **para uma única amostra**. Para generalizar para um conjunto de dados com $m$ amostras, basta calcular essas expressões para cada amostra individual e, em seguida, **tomar a média** dos resultados:

1. Derivada da função de perda em relação ao peso $w_j^{[L]}$ para $m$ amostras:  
  $$
  \\frac{\partial \mathcal{L}}{\partial w_j^{[L]}} = 
  \\frac{1}{m} \sum_{i=1}^m (a^{[L](i)} - y^{(i)}) \cdot f'^{[L]}(z^{[L](i)}) \cdot a_j^{[L-1](i)}
  $$

2. Derivada da função de perda em relação ao viés $b^{[L]}$ para $m$ amostras:  
  $$
  \\frac{\partial \mathcal{L}}{\partial b^{[L]}} = 
  \\frac{1}{m} \sum_{i=1}^m (a^{[L](i)} - y^{(i)}) \cdot f'^{[L]}(z^{[L](i)})
  $$

Essas formulas são essenciais para o treinamento eficiente de redes neurais com múltiplos exemplos simultaneamente (batch learning).

Por fim, considerando:

1. a matriz cujas colunas são as ativações da última camada oculta para cada uma das $m$ amostras: 

$$
\mathbf{A}^{[L-1]} \in \mathbb{R}^{n_{L-1} \\times m}
$$ 

2. O vetor de pré-ativações da camada de saída:

$$
\mathbf{Z}^{[L]} = \mathbf{w}^{[L] \\top} \mathbf{A}^{[L-1]} + b^{[L]}
$$ 

3. o vetor das predições da rede: 

$$
\mathbf{A}^{[L]} = f^{[L]}(\mathbf{Z}^{[L]})
$$

4. o vetor com os valores reais esperados: 

$$
\mathbf{Y} \in \mathbb{R}^{1 \\times m}
$$ 

5. A função de perda média: 

$$
\mathcal{L} = \dfrac{1}{2m} \sum_{i=1}^m \left(a^{[L](i)} - y^{(i)}\\right)^2
$$ 

a derivada da função de perda média em relação ao vetor de pesos da última camada $\mathbf{w}^{[L]} \in \mathbb{R}^{n_{L-1}}$ é:

$$
\\frac{\partial \mathcal{L}}{\partial \mathbf{w}^{[L]}} = \\frac{1}{m} \cdot \mathbf{A}^{[L-1]} \cdot \left[ \left( \mathbf{A}^{[L]} - \mathbf{Y} \\right) \odot f'^{[L]}(\mathbf{Z}^{[L]}) \\right]^\\top
$$
enquanto que a derivada da função de perda média em relação ao viés escalar $b^{[L]}$ é:

$$
\\frac{\partial \mathcal{L}}{\partial b^{[L]}} = \\frac{1}{m} \cdot \mathbf{1}^\\top \cdot \left[ \left( \mathbf{A}^{[L]} - \mathbf{Y} \\right) \odot f'^{[L]}(\mathbf{Z}^{[L]}) \\right]
$$

em que $\mathbf{1} \in \mathbb{R}^{m \\times 1}$ é um vetor coluna de uns.

Essas expressões vetoriais são altamente eficientes e podem ser implementadas diretamente em frameworks de aprendizado profundo. Elas são a base para o cálculo de gradientes na **fase de retropropagação** (backpropagation) durante o treinamento.

</div> '''
  },
  {
    'titulo': 'Treinamento de Redes Neurais',
    'conteudo': '''<div align="justify">
    
O processo de **treinamento** de uma Rede Neural Artificial (RNA) consiste na **ajuste iterativo dos pesos sinápticos** de modo a minimizar uma função de perda que quantifica o erro entre a saída prevista e a saída desejada.

Seja uma rede com parâmetros $\\theta$ (conjunto de pesos e vieses), a função de perda global é representada por:

$$
\mathcal{L}(\\theta) = \\frac{1}{n} \sum_{i=1}^n \ell(\hat{y}^{(i)}, y^{(i)})
$$

onde:

- $n$ é o número de amostras de treinamento,
- $y^{(i)}$ é a saída esperada,
- $\hat{y}^{(i)}$ é a saída da rede para a entrada $\mathbf{x}^{(i)}$,
- $\ell$ é a função de perda (como MSE ou entropia cruzada).

### Algoritmo de Retropropagação (Backpropagation)

O **algoritmo de retropropagação** é um método eficiente de **cálculo do gradiente** da função de perda com respeito a cada peso da rede, utilizando a **regra da cadeia** do cálculo diferencial.

#### Etapas do Algoritmo

1. **Propagação direta (Forward pass):**

   Calcula-se a saída da rede para uma dada entrada $\mathbf{x}^{(i)}$, armazenando os valores intermediários de ativações e pré-ativações em cada camada.

2. **Cálculo do erro na saída:**

   Para a última camada (camada de saída), o erro local é dado por:

   $$
   \delta^{[L]} = \\nabla_{\hat{y}} \ell(\hat{y}, y) \odot f'^{[L]}(z^{[L]})
   $$

   onde:

   - $f'^{[L]}$ é a derivada da função de ativação da camada de saída,
   - $z^{[L]}$ é a entrada linear (pré-ativação) da última camada,
   - $\odot$ denota o produto de Hadamard (elemento a elemento).

3. **Propagação do erro para camadas anteriores:**

   Para cada camada $l = L-1, ..., 1$, o erro é retropropagado pela fórmula:

   $$
   \delta^{[l]} = (W^{[l+1]})^T \delta^{[l+1]} \odot f'^{[l]}(z^{[l]})
   $$

4. **Cálculo dos gradientes:**

   Os gradientes da perda em relação aos pesos e vieses são obtidos por:

   $$
   \\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \delta^{[l]} (a^{[l-1]})^T, \\
   \\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \delta^{[l]}
   $$

5. **Atualização dos parâmetros (Gradiente Descendente):**

   Com uma taxa de aprendizado $\eta > 0$, os pesos são atualizados conforme:

   $$
   W^{[l]} \leftarrow W^{[l]} - \eta \\frac{\partial \mathcal{L}}{\partial W^{[l]}}, \\
   b^{[l]} \leftarrow b^{[l]} - \eta \\frac{\partial \mathcal{L}}{\partial b^{[l]}}
   $$

---

### Considerações Finais

O **algoritmo de retropropagação** é a base do treinamento supervisionado em redes neurais profundas. Ele permite que a informação de erro flua da saída até as primeiras camadas, ajustando os pesos para melhorar gradualmente a performance da rede no conjunto de treinamento.

A compreensão da derivação matemática e do funcionamento recursivo da retropropagação é essencial para o domínio teórico e prático do treinamento de RNAs.

---
### Cálculo das Derivadas na Camada de Saída: Aplicando a Regra da Cadeia

O treinamento de uma rede neural baseia-se na minimização de uma função de perda, e isso requer o cálculo de derivadas parciais da perda em relação aos parâmetros da rede. Um dos passos fundamentais nesse processo é o cálculo da derivada da função de perda em relação à **entrada linear** da camada de saída — isto é, os valores antes da aplicação da função de ativação.

Nesta seção, explicamos esse procedimento com base na **regra da cadeia**, de forma rigorosa e acessível.

---

#### Notação e Estrutura

Considere a última camada de uma rede neural (denotada por índice $L$):

- A entrada da rede é $\mathbf{x}$;
- A saída da penúltima camada (ou última camada oculta) é o vetor $\mathbf{a}^{[L-1]}$;
- Os pesos da camada de saída são $W^{[L]}$ e os vieses são $b^{[L]}$;
- A entrada linear (pré-ativação) da camada de saída é:
  
  $$
  \mathbf{z}^{[L]} = W^{[L]} \mathbf{a}^{[L-1]} + b^{[L]}
  $$

- A função de ativação $f^{[L]}$ é aplicada elemento a elemento sobre $\mathbf{z}^{[L]}$, produzindo a saída da rede:

  $$
  \mathbf{a}^{[L]} = f^{[L]}(\mathbf{z}^{[L]})
  $$

- A função de perda (loss function), denotada por $\mathcal{L}$, compara $\mathbf{a}^{[L]}$ com o valor alvo $\mathbf{y}$.

---

#### Aplicando a Regra da Cadeia

Nosso objetivo é calcular:

$$
\\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[L]}}
$$

Ou seja, como pequenas variações na entrada linear da última camada ($\mathbf{z}^{[L]}$) afetam o valor da perda.

Para isso, usamos a **regra da cadeia**, considerando que:

- $\mathcal{L}$ depende de $\mathbf{z}^{[L]}$ apenas indiretamente, por meio de $\mathbf{a}^{[L]}$,
- $\mathbf{a}^{[L]} = f^{[L]}(\mathbf{z}^{[L]})$.

Logo, temos:

$$
\\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[L]}} = \\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \odot \\frac{\partial \mathbf{a}^{[L]}}{\partial \mathbf{z}^{[L]}} = \\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \odot f'^{[L]}(\mathbf{z}^{[L]})
$$

Este é o ponto central: a derivada da perda em relação à entrada linear $\mathbf{z}^{[L]}$ é o **produto de Hadamard** (produto componente a componente) entre dois vetores:

- o vetor do gradiente da perda em relação à saída da rede ($\partial \mathcal{L} / \partial \mathbf{a}^{[L]}$),
- e o vetor derivada da função de ativação avaliada em $\mathbf{z}^{[L]}$.

---

#### Interpretação

Cada componente da derivada $\partial \mathcal{L} / \partial z^{[L]}_i$ depende apenas do correspondente $a^{[L]}_i$ e $z^{[L]}_i$. Isso ocorre porque a ativação $f^{[L]}$ é aplicada **individualmente a cada neurônio** da camada de saída. Essa independência entre neurônios justifica o uso do **produto de Hadamard**.


### Derivadas na Camada de Saída: Um Caminho Componente a Componente

Para entender de forma rigorosa o cálculo da derivada da função de perda em relação à entrada linear da **camada de saída**, começamos examinando o problema componente a componente, antes de generalizar a notação para vetores.

Considere uma rede neural com camada de saída indexada por $L$, e saída vetorial $\mathbf{a}^{[L]} \in \mathbb{R}^{n_L}$. A entrada linear (pré-ativação) de cada neurônio da saída é:

$$
z^{[L]}_i = \sum_{j=1}^{n_{L-1}} w^{[L]}_{ij} a^{[L-1]}_j + b^{[L]}_i
$$

Após a aplicação da função de ativação $f^{[L]}$ (aplicada ponto a ponto), temos:

$$
a^{[L]}_i = f^{[L]}(z^{[L]}_i)
$$

A função de perda $\mathcal{L}$ depende da predição da rede, isto é, de todos os $a^{[L]}_i$. Nosso objetivo é calcular:

$$
\frac{\partial \mathcal{L}}{\partial z^{[L]}_i}
$$

---

#### Passo 1: Regra da Cadeia (componente por componente)

Aplicando a **regra da cadeia** à função composta $\mathcal{L}(a^{[L]}_i(z^{[L]}_i))$, temos:

$$
\frac{\partial \mathcal{L}}{\partial z^{[L]}_i} = \frac{\partial \mathcal{L}}{\partial a^{[L]}_i} \cdot \frac{\partial a^{[L]}_i}{\partial z^{[L]}_i}
$$

A primeira derivada, $\frac{\partial \mathcal{L}}{\partial a^{[L]}_i}$, depende da forma da função de perda utilizada (por exemplo, erro quadrático ou entropia cruzada).

A segunda derivada é dada pela derivada da função de ativação:

$$
\frac{\partial a^{[L]}_i}{\partial z^{[L]}_i} = f'^{[L]}(z^{[L]}_i)
$$

Portanto:

$$
\frac{\partial \mathcal{L}}{\partial z^{[L]}_i} = \frac{\partial \mathcal{L}}{\partial a^{[L]}_i} \cdot f'^{[L]}(z^{[L]}_i)
$$

---

#### Passo 2: Generalização Vetorial

O resultado acima vale para cada neurônio $i = 1, \dots, n_L$. Podemos agora reunir todos os termos em um único vetor de derivadas:

- $\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[L]}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial z^{[L]}_1} \\ \vdots \\ \frac{\partial \mathcal{L}}{\partial z^{[L]}_{n_L}} \end{bmatrix}$  
- $\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial a^{[L]}_1} \\ \vdots \\ \frac{\partial \mathcal{L}}{\partial a^{[L]}_{n_L}} \end{bmatrix}$  
- $f'^{[L]}(\mathbf{z}^{[L]}) = \begin{bmatrix} f'^{[L]}(z^{[L]}_1) \\ \vdots \\ f'^{[L]}(z^{[L]}_{n_L}) \end{bmatrix}$

Juntando essas expressões, obtemos:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[L]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \odot f'^{[L]}(\mathbf{z}^{[L]})
$$

onde $\odot$ representa o **produto de Hadamard**, ou seja, o produto elemento a elemento entre vetores.

---

### Conclusão

A fórmula:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[L]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \odot f'^{[L]}(\mathbf{z}^{[L]})
$$

é a **versão vetorial da regra da cadeia**, e surge naturalmente ao calcular as derivadas **componente a componente**. Esse resultado é fundamental para o algoritmo de retropropagação, pois estabelece o elo entre o erro na predição e os ajustes que devem ser feitos nos pesos da rede.

---

#### Exemplo com Classificação Binária

Suponha uma tarefa de classificação binária com:

- Um único neurônio na saída ($n_L = 1$),
- Função de ativação $f^{[L]} = \sigma$ (sigmoide),
- Função de perda $\mathcal{L}$ como entropia cruzada binária:

  $$
  \mathcal{L} = - \left[y \log a^{[L]} + (1 - y) \log(1 - a^{[L]})\right]
  $$

Neste caso, a derivada da perda em relação à saída é:

$$
\frac{\partial \mathcal{L}}{\partial a^{[L]}} = -\frac{y}{a^{[L]}} + \frac{1 - y}{1 - a^{[L]}}
$$

A derivada da função sigmoide é:

$$
f'^{[L]}(z^{[L]}) = a^{[L]}(1 - a^{[L]})
$$

Multiplicando os dois, temos a derivada em relação à pré-ativação:

$$
\frac{\partial \mathcal{L}}{\partial z^{[L]}} = \left( -\frac{y}{a^{[L]}} + \frac{1 - y}{1 - a^{[L]}} \right) \cdot a^{[L]}(1 - a^{[L]})
$$

Em muitos casos, isso se simplifica para:

$$
\frac{\partial \mathcal{L}}{\partial z^{[L]}} = a^{[L]} - y
$$

um resultado amplamente utilizado na prática por sua eficiência computacional.

---

### Conclusão

O uso da **regra da cadeia** para obter as derivadas da camada de saída é uma etapa essencial da retropropagação. A presença do **produto de Hadamard** reflete a estrutura vetorial da rede e a independência entre os neurônios da mesma camada. A compreensão detalhada desse processo é crucial para dominar a teoria e a implementação do treinamento de redes neurais.



    </div>'''
  }

]
