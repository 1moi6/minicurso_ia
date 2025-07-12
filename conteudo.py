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

introducao='''### 1. Introdução: A Convergência de IA, ML e Matemática

#### 1.1. O que é Inteligência Artificial (IA)?

A **Inteligência Artificial (IA)** constitui um campo de estudo em constante expansão, cujo objetivo central é capacitar máquinas a replicar competências cognitivas humanas, tais como raciocínio, aprendizagem, planejamento e criatividade. Os sistemas de IA são concebidos para interagir com o ambiente, processando dados — sejam eles previamente preparados ou coletados por sensores, como câmeras — e gerando respostas adaptativas.

Uma característica distintiva desses sistemas reside na sua capacidade de modificar o comportamento de forma autônoma, analisando os efeitos de ações passadas.  

Esta emulação da cognição humana e a autonomia conferem à IA um papel que transcende a mera programação complexa, posicionando-a como uma disciplina que busca construir sistemas capazes de **aprender e evoluir** com base na experiência, de maneira análoga aos sistemas biológicos.

Essa abordagem ressalta a natureza **interdisciplinar** da IA, que se nutre de conhecimentos da ciência cognitiva, ciência da computação e, fundamentalmente, da **matemática**.

No entanto, os primórdios da IA enfrentaram um desafio significativo: lidar com tarefas intuitivamente fáceis para humanos, mas difíceis de formalizar para computadores, como reconhecimento de fala ou rostos. Projetos como o **Cyc** tentaram codificar o conhecimento do mundo em regras formais, mas falharam em capturar a complexidade do mundo real. Um exemplo clássico é o fracasso do Cyc em compreender uma história simples envolvendo um barbeador elétrico, devido à limitação de suas representações formais.

Essas dificuldades impulsionaram a necessidade de sistemas que **aprendem com dados**, ao invés de dependerem exclusivamente de regras explícitas. Isso deu origem ao **Machine Learning (ML)**, uma transição fundamental marcada pela adoção de ferramentas matemáticas como **probabilidade** e **otimização**.

---

#### 1.2. O que é Machine Learning (ML)?

O **Machine Learning (ML)** é um subcampo crucial da IA que capacita computadores a aprenderem **sem programação explícita**, executando tarefas de forma autônoma. Através de algoritmos, o ML permite identificar **padrões em grandes volumes de dados** e fazer previsões — caracterizando a chamada **análise preditiva**.

O aprendizado em ML é contínuo: o desempenho melhora conforme mais dados e experiências são incorporados.  

A distinção entre ML e IA é importante: o ML fornece os métodos que tornam a IA possível, especialmente no que diz respeito à **capacidade de aprender com dados**.

A eficácia de algoritmos de ML — mesmo os mais simples, como **regressão logística** — depende criticamente da **representação dos dados**. Por exemplo, prever uma cesariana a partir de uma variável como “cicatriz uterina” é viável com regressão logística. Já uma imagem de ressonância magnética (MRI) bruta, sem representação adequada, não oferece pistas úteis ao algoritmo.

Historicamente, a solução era projetar manualmente **features** (características) adequadas para o problema. No entanto, tarefas como detectar "rodas" em imagens sofrem com variações de sombra, brilho e oclusões, tornando a **engenharia manual de features** ineficaz.

Isso motivou o desenvolvimento do **aprendizado de representações** (representation learning), em que os próprios algoritmos de ML aprendem **quais características são úteis**. Esse avanço reduziu a dependência de intervenção humana e viabilizou aplicações mais adaptáveis e poderosas.

---

#### 1.3. Breve Histórico das Redes Neurais Artificiais (RNAs)

As **Redes Neurais Artificiais (RNAs)** têm uma história marcada por ciclos de entusiasmo, frustração e renascimento. Essa trajetória pode ser dividida em três ondas:

#### Primeira Onda — Cibernética (1940s–1960s)

- Criação do **neurônio de McCulloch e Pitts** (1943)
- Regra de Hebb (1949)
- Modelos como o **Perceptron** (Rosenblatt, 1958) e o **ADALINE** (Widrow & Hoff, 1960)
- Treinamento com base em **gradiente descendente estocástico**
- Críticas de **Minsky e Papert** (1969) sobre a incapacidade de aprender funções como XOR causaram um declínio da área

#### Segunda Onda — Conexionismo (1980s–1990s)

- Popularização do **algoritmo de retropropagação** (backpropagation) por Rumelhart et al. (1986)
- Conceito de **representação distribuída** (Hinton et al., 1986)
- Introdução das **LSTM** (Hochreiter & Schmidhuber, 1997)
- Declínio novamente nos anos 1990, com o avanço de outras técnicas como **máquinas de vetor de suporte** e **modelos gráficos probabilísticos**

#### Terceira Onda — Deep Learning (a partir de 2006)

- Avanço decisivo de **Geoffrey Hinton** (2006) com **deep belief networks**
- Estratégia de **pré-treinamento guloso camada a camada**
- Popularização do termo **Deep Learning**, com ênfase na **profundidade das redes**
- Crescimento impulsionado por:
  - Expansão de **dados disponíveis**
  - Avanços em **hardware e software**
  - Aumento da **precisão das aplicações no mundo real**

---

A história das RNAs e da IA mostra como **avanços teóricos exigem suporte tecnológico** para serem plenamente realizados. A matemática fornece as bases, mas é a engenharia que possibilita a aplicação dessas ideias em larga escala, conectando teoria e prática de forma indissociável.
'''

introducao2 = f'''### 1. Introdução: a convergência entre IA, ML e Matemática

<div style="text-align: center;">
        <img src="https://github.com/1moi6/tccs/blob/main/pietra/tcc_pietra/Figure%206.png?raw=true" width="400"/>
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
As **Redes Neurais Artificiais (RNAs)** possuem uma trajetória marcada por altos e baixos, refletindo tanto os avanços teóricos quanto as limitações tecnológicas de cada época. Essa história pode ser dividida em três grandes ondas:
</div>
#### Primeira Onda — Cibernética (décadas de 1940–1960)

- Criação do **neurônio de McCulloch e Pitts** (1943) e da **regra de Hebb** (1949)
- Desenvolvimento dos modelos **Perceptron** (Rosenblatt, 1958) e **ADALINE** (Widrow & Hoff, 1960)
- Introdução do **gradiente descendente estocástico**, base de muitos algoritmos atuais

Esses modelos, embora pioneiros, eram limitados a funções lineares. A incapacidade de resolver problemas simples, como a função XOR, levou à crítica de **Minsky e Papert (1969)** e ao subsequente declínio da área.

#### Segunda Onda — Conexionismo (décadas de 1980–1990)

- Redescoberta e popularização do algoritmo de **retropropagação** (*backpropagation*) por **Rumelhart et al. (1986)**
- Introdução da **representação distribuída** (Hinton et al., 1986)
- Desenvolvimento das **LSTM** (Long Short-Term Memory) por **Hochreiter e Schmidhuber (1997)** para lidar com dependências temporais

<div align="justify">
Apesar desses avanços, o campo perdeu tração frente ao crescimento de técnicas alternativas, como **máquinas de vetores de suporte** e **modelos probabilísticos gráficos**.
</div>

#### Terceira Onda — Deep Learning (a partir de 2006)

- Avanço decisivo de **Geoffrey Hinton**, com as **deep belief networks** treinadas via pré-treinamento camada a camada
- Consolidação do termo **Deep Learning**, destacando a importância das redes **profundas**
- Expansão acelerada graças a:
  - Disponibilidade de **grandes volumes de dados**
  - Avanços em **hardware (GPUs)** e bibliotecas de software
  - Aumento da precisão de modelos aplicados em **visão computacional, linguagem natural e robótica**

---
<div align="justify">
A evolução das RNAs ilustra a interdependência entre **teoria matemática e capacidade computacional**. A matemática fornece os fundamentos para os modelos, mas é o avanço tecnológico que permite sua implementação em escala real. O renascimento das RNAs com o Deep Learning mostra que **inovações significativas exigem tanto compreensão teórica quanto infraestrutura para serem efetivamente aplicadas.**
</div>
---
'''