# EduAI: Sistema Inteligente de Análise Educacional

O **EduAI** é um sistema de suporte à decisão focado no ambiente acadêmico. Ele foi desenvolvido para auxiliar professores e coordenadores a identificar alunos com dificuldades de aprendizagem, prever riscos de reprovação e sugerir abordagens pedagógicas personalizadas. O projeto integra diversas subáreas da Inteligência Artificial em uma interface amigável e focada na experiência do educador.

## 🌟 Funcionalidades Principais

*   **Entrada de dados intuitiva:** As métricas do aluno estão agrupadas em categorias claras, como "Desempenho Acadêmico" e "Assiduidade e Hábitos", utilizando controles deslizantes para facilitar e agilizar o preenchimento[cite: 2].
*   **Diagnóstico consolidado imediato:** A decisão central do Agente Inteligente é exibida com destaque logo após a análise, informando rapidamente o nível de risco atual e a nota final projetada[cite: 2].
*   **Visualização gráfica avançada:** O painel conta com gráficos dinâmicos, incluindo um indicador de "Nível de risco" em formato de velocímetro e um gráfico de radar que compara o desempenho atual do aluno com o perfil ideal da escola[cite: 3].
*   **Recomendações pedagógicas práticas:** Além de apontar o problema (ex: o ponto mais fraco do aluno), o sistema sugere abordagens de ensino acionáveis, como aulas de reforço individuais ou explicações com exemplos concretos[cite: 3].
*   **Inteligência Artificial acessível:** Os jargões técnicos foram traduzidos para a realidade escolar; por exemplo, a classificação por KNN é mostrada como "Alunos parecidos", e o agrupamento K-Means como "Perfil de comportamento", incluindo ícones informativos de ajuda[cite: 4].
*   **Análise histórica em linguagem natural:** O motor de mineração de dados extrai regras de associação (Apriori) e as apresenta em texto corrido, alertando o professor sobre a porcentagem de alunos no passado que reprovaram apresentando o mesmo padrão de notas e tarefas[cite: 5].

## 🧠 Técnicas de IA Integradas

O Agente Inteligente processa os dados simultaneamente através de 7 motores distintos:
1.  **Sistema Especialista:** Base de regras fixas sobre políticas da escola.
2.  **Lógica Fuzzy:** Tratamento de incertezas para definir o nível de atenção necessário.
3.  **Árvore de Decisão:** Classificação do aluno em níveis de alerta.
4.  **KNN (K-Nearest Neighbors):** Busca no banco de dados por alunos históricos com comportamento semelhante.
5.  **K-Means:** Agrupamento para descobrir o perfil comportamental do aluno.
6.  **Regressão Linear:** Projeção estatística da nota final esperada.
7.  **Apriori:** Mineração de regras de associação para encontrar padrões de risco.

## 🚀 Como executar o projeto

Certifique-se de ter o Python instalado na sua máquina. Siga os passos abaixo para instalar as dependências e rodar a aplicação:

**1. Instale as dependências necessárias:**
```bash
pip install streamlit scikit-learn scikit-fuzzy pandas numpy scipy networkx plotly anthropic
streamlit run app.py ou python -m streamlit run app.py   