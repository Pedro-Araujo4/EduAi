"""
Sistema Inteligente de Análise Educacional
Executar:  streamlit run app.py
Dependências: pip install streamlit scikit-learn scikit-fuzzy pandas numpy scipy networkx
"""
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import skfuzzy as fuzz
import streamlit as st
from skfuzzy import control as ctrl
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Análise Educacional", page_icon="🎓", layout="wide")

FEATURES = ["nota", "frequencia", "tarefas", "estudo"]
ANTECEDENTES = ["Nota_Baixa", "Faltas_Altas", "Tarefas_Baixas", "Estudo_Baixo", "Faltas_Segunda"]
IDEAL = {"Nota": 80, "Frequência": 90, "Tarefas": 90, "Estudo extra": 70}  # escala 0-100


def valores_radar(nota, freq, tarefas, estudo):
    return {"Nota": nota * 10, "Frequência": freq, "Tarefas": tarefas,
            "Estudo extra": min(estudo / 10 * 100, 100)}


# ----------------------------------------------------------------- DADOS
def categoria(final):
    if final < 5:
        return "Risco de Reprovação"
    if final < 7:
        return "Atenção / Recuperação"
    if final < 9:
        return "Aprovado"
    return "Aprovado com Louvor"


def itens_do_aluno(nota, freq, tarefas, estudo, seg):
    itens = {
        "Nota_Baixa": nota < 6,
        "Faltas_Altas": freq < 75,
        "Tarefas_Baixas": tarefas < 50,
        "Estudo_Baixo": estudo < 3,
        "Faltas_Segunda": seg,
    }
    return {k for k, v in itens.items() if v}


@st.cache_data
def gerar_dataset(n=400, seed=42):
    rng = np.random.default_rng(seed)
    base = rng.beta(4, 2, n)  # "perfil" latente do aluno
    df = pd.DataFrame({
        "nota": np.clip(base * 10 + rng.normal(0, 1.2, n), 0, 10),
        "frequencia": np.clip(55 + base * 45 + rng.normal(0, 8, n), 30, 100),
        "tarefas": np.clip(base * 100 + rng.normal(0, 18, n), 0, 100),
        "estudo": np.clip(base * 12 + rng.normal(0, 3, n), 0, 20),
    })
    df["faltas_segunda"] = rng.random(n) < np.where(df.frequencia < 75, 0.65, 0.12)
    df["nota_final"] = np.clip(
        0.6 * df.nota + 0.02 * df.tarefas + 0.15 * df.estudo + 0.01 * (df.frequencia - 70)
        + rng.normal(0, 0.5, n), 0, 10)
    df["resultado"] = df.nota_final.apply(categoria)
    return df.round(2)


# --------------------------------------------------------------- MOTORES
def sistema_especialista(nota, freq, tarefas, estudo, seg):
    regras = [
        (nota < 6 and freq > 75, "R1: Recomendar revisão do conteúdo."),
        (nota < 5 and tarefas < 50, "R2: Recomendar estudo intensivo e notificar responsáveis."),
        (freq < 75 and nota < 6, "R3: Notificar coordenação sobre infrequência associada a baixo rendimento."),
        (tarefas < 50 and nota >= 6, "R4: Reforçar acompanhamento da entrega de tarefas."),
        (estudo < 2 and nota < 7, "R5: Orientar construção de rotina de estudo extra-classe."),
        (nota >= 8 and tarefas >= 80 and freq >= 90, "R6: Propor atividades de enriquecimento/monitoria."),
    ]
    return [msg for cond, msg in regras if cond]


@st.cache_resource
def sistema_fuzzy():
    nota = ctrl.Antecedent(np.arange(0, 10.01, 0.1), "nota")
    est = ctrl.Antecedent(np.arange(0, 20.01, 0.1), "estudo")
    out = ctrl.Consequent(np.arange(0, 100.01, 1), "interv")
    nota["baixa"], nota["media"], nota["alta"] = (
        fuzz.trimf(nota.universe, [0, 0, 5]), fuzz.trimf(nota.universe, [3, 5, 7]),
        fuzz.trimf(nota.universe, [5, 10, 10]))
    est["baixo"], est["medio"], est["alto"] = (
        fuzz.trimf(est.universe, [0, 0, 6]), fuzz.trimf(est.universe, [2, 7, 12]),
        fuzz.trimf(est.universe, [8, 20, 20]))
    out["baixa"], out["media"], out["alta"] = (
        fuzz.trimf(out.universe, [0, 0, 50]), fuzz.trimf(out.universe, [25, 50, 75]),
        fuzz.trimf(out.universe, [50, 100, 100]))
    tabela = [
        ("baixa", "baixo", "alta"), ("baixa", "medio", "alta"),
        ("baixa", "alto", "media"),  # dificuldade de aprendizado, não falta de esforço
        ("media", "baixo", "media"), ("media", "medio", "baixa"), ("media", "alto", "baixa"),
        ("alta", "baixo", "baixa"), ("alta", "medio", "baixa"), ("alta", "alto", "baixa"),
    ]
    return ctrl.ControlSystem([ctrl.Rule(nota[a] & est[b], out[c]) for a, b, c in tabela])


def logica_fuzzy(nota, estudo):
    sim = ctrl.ControlSystemSimulation(sistema_fuzzy())
    sim.input["nota"] = float(np.clip(nota, 0, 10))
    sim.input["estudo"] = float(np.clip(estudo, 0, 20))
    sim.compute()
    v = float(sim.output["interv"])
    nivel = "Baixa" if v < 35 else "Média" if v < 65 else "Alta"
    dica = ""
    if nota < 5 and estudo >= 8:
        dica = "Alto esforço com nota baixa: possível dificuldade de aprendizado, não falta de esforço."
    return v, nivel, dica


@st.cache_resource
def treinar_modelos(n):
    df = gerar_dataset(n)
    X = df[FEATURES]
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    arvore = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, df.resultado)
    knn = NearestNeighbors(n_neighbors=3).fit(Xs)
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Xs)
    reg = LinearRegression().fit(X, df.nota_final)

    # nomeia clusters pelo centróide: maior nota = engajados; entre os demais, maior engajamento = esforçados
    cent = scaler.inverse_transform(km.cluster_centers_)
    eng = cent[:, 1] / 100 + cent[:, 2] / 100 + cent[:, 3] / 20
    melhor = int(np.argmax(cent[:, 0]))
    resto = sorted([i for i in range(3) if i != melhor], key=lambda i: -eng[i])
    nomes = {melhor: "Alto desempenho e engajados", resto[0]: "Esforçados com dificuldade",
             resto[1]: "Desengajados"}
    return dict(df=df, scaler=scaler, arvore=arvore, knn=knn, km=km, reg=reg, nomes=nomes)


def arvore_decisao(m, x):
    p = m["arvore"].predict_proba(x)[0]
    i = int(np.argmax(p))
    return m["arvore"].classes_[i], float(p[i])


def knn_vizinhos(m, x):
    _, idx = m["knn"].kneighbors(m["scaler"].transform(x))
    return m["df"].iloc[idx[0]][FEATURES + ["nota_final", "resultado"]]


def kmeans_cluster(m, x):
    return m["nomes"][int(m["km"].predict(m["scaler"].transform(x))[0])]


def regressao(m, x):
    return float(np.clip(m["reg"].predict(x)[0], 0, 10))


def apriori(m, itens_aluno, sup_min=0.05, conf_min=0.6):
    """Apriori simplificado (itemsets até 3 itens) com consequente 'Reprovação'."""
    df = m["df"].copy()
    d = pd.DataFrame({
        "Nota_Baixa": df.nota < 6, "Faltas_Altas": df.frequencia < 75,
        "Tarefas_Baixas": df.tarefas < 50, "Estudo_Baixo": df.estudo < 3,
        "Faltas_Segunda": df.faltas_segunda})
    reprov = (df.nota_final < 5).values
    regras = []
    for k in (1, 2, 3):
        for comb in combinations(ANTECEDENTES, k):
            mask = d[list(comb)].all(axis=1).values
            sup = mask.mean()
            if sup >= sup_min and mask.sum() > 0:
                conf = reprov[mask].mean()
                if conf >= conf_min:
                    regras.append((set(comb), sup, conf, int(mask.sum())))
    aplicaveis = [r for r in regras if r[0] <= itens_aluno]
    return sorted(aplicaveis, key=lambda r: -r[2])[:4]


# ---------------------------------------------------------------- AGENTE
def agente(nota, freq, tarefas, estudo, seg, m, conf_min):
    x = pd.DataFrame([[nota, freq, tarefas, estudo]], columns=FEATURES)
    itens = itens_do_aluno(nota, freq, tarefas, estudo, seg)
    # Processamento simultâneo dos 7 motores
    with ThreadPoolExecutor(max_workers=7) as ex:
        f = {
            "especialista": ex.submit(sistema_especialista, nota, freq, tarefas, estudo, seg),
            "fuzzy": ex.submit(logica_fuzzy, nota, estudo),
            "arvore": ex.submit(arvore_decisao, m, x),
            "knn": ex.submit(knn_vizinhos, m, x),
            "kmeans": ex.submit(kmeans_cluster, m, x),
            "reg": ex.submit(regressao, m, x),
            "apriori": ex.submit(apriori, m, itens, 0.05, conf_min),
        }
        r = {k: v.result() for k, v in f.items()}

    # Decisão: score de risco ponderado (0-100)
    cat, _ = r["arvore"]
    risco_arvore = {"Risco de Reprovação": 1, "Atenção / Recuperação": 0.5}.get(cat, 0)
    knn_risco = (r["knn"].resultado == "Risco de Reprovação").mean()
    conf_ap = max([c for _, _, c, _ in r["apriori"]], default=0)
    score = (r["fuzzy"][0] * 0.25 + risco_arvore * 25 + knn_risco * 20
             + np.clip((7 - r["reg"]) / 7, 0, 1) * 20 + conf_ap * 10)
    r["score"] = float(score)
    r["nivel"] = "ALTO" if score >= 55 else "MODERADO" if score >= 30 else "BAIXO"

    abordagem = {
        "Esforçados com dificuldade": "Aula de reforço individual, explicações com exemplos concretos e "
                                      "atividades escalonadas; evitar apenas aumentar a carga de estudo.",
        "Desengajados": "Reengajar com metas curtas e semanais, tarefas em grupo/projetos, contato "
                        "com responsáveis e acompanhamento da frequência.",
        "Alto desempenho e engajados": "Manter o ritmo com desafios extras, monitoria entre pares e "
                                       "atividades de aprofundamento.",
    }[r["kmeans"]]
    r["abordagem"] = abordagem
    vals = valores_radar(nota, freq, tarefas, estudo)
    r["ponto_fraco"] = min(vals, key=lambda k: vals[k] / IDEAL[k])
    r["entrada"] = dict(nota=nota, freq=freq, tarefas=tarefas, estudo=estudo, seg=seg)
    frases = {"ALTO": "Intervenção prioritária recomendada.",
              "MODERADO": "Acompanhamento mais próximo recomendado.",
              "BAIXO": "Sem sinais de alerta relevantes no momento."}
    r["diagnostico"] = (f"{frases[r['nivel']]} Perfil: <b>{r['kmeans']}</b>. Nota final projetada: "
                        f"<b>{r['reg']:.1f}</b>. Maior ponto fraco: <b>{r['ponto_fraco']}</b>.")
    return r


# ------------------------------------------------------------------- UI
ESTADOS = {  # (fundo, borda, ícone) — a cor indica o ESTADO do aluno, não o algoritmo
    "ok": ("#dcfce7", "#16a34a", "🟢"),
    "atencao": ("#ffedd5", "#f59e0b", "🟠"),
    "alerta": ("#fee2e2", "#dc2626", "🔴"),
}
NIVEL_ESTADO = {"BAIXO": "ok", "MODERADO": "atencao", "ALTO": "alerta"}
ITEM_TXT = {
    "Nota_Baixa": "nota abaixo de 6",
    "Faltas_Altas": "frequência abaixo de 75%",
    "Tarefas_Baixas": "menos da metade das tarefas entregues",
    "Estudo_Baixo": "menos de 3h de estudo por semana",
    "Faltas_Segunda": "faltas frequentes às segundas-feiras",
}
MODELO = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0f172a}
[data-testid="stSidebar"] *{color:#e2e8f0 !important}
[data-testid="stSidebar"] input,[data-testid="stSidebar"] textarea{color:#0f172a !important}
[data-testid="stSidebar"] div.stButton>button{background:transparent;border:1px solid #334155}
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button{background:#2563eb;color:#fff !important}
.card{border-radius:10px;padding:.9rem 1.1rem;min-height:200px;margin-bottom:1rem;color:#1e293b;
      box-shadow:0 1px 4px rgba(0,0,0,.12)}
.card h4{margin:0 0 .5rem 0;color:#1e293b}
.card li{margin-bottom:.3rem}
.agente{color:#fff;padding:1.1rem 1.5rem;border-radius:12px;margin:.5rem 0 .8rem 0}
.agente h3,.agente p{color:#fff;margin:.2rem 0}
div.stButton>button[kind="primary"]{background:#2563eb;color:#fff;font-weight:600;border:none}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------- Gráficos
def grafico_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": "/100"},
        title={"text": "Nível de risco"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1e293b", "thickness": 0.25},
               "steps": [{"range": [0, 30], "color": "#22c55e"},
                         {"range": [30, 55], "color": "#facc15"},
                         {"range": [55, 100], "color": "#ef4444"}]}))
    fig.update_layout(height=270, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def grafico_radar(e):
    cats = list(IDEAL)
    vals = valores_radar(e["nota"], e["freq"], e["tarefas"], e["estudo"])
    fechar = lambda l: l + [l[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=fechar(list(IDEAL.values())), theta=fechar(cats), name="Ideal",
                                  line=dict(color="#16a34a", dash="dash"), fill="none"))
    fig.add_trace(go.Scatterpolar(r=fechar(list(vals.values())), theta=fechar(cats), name="Aluno",
                                  fill="toself", line=dict(color="#2563eb")))
    fig.update_layout(height=270, margin=dict(l=40, r=40, t=30, b=10),
                      polar=dict(radialaxis=dict(range=[0, 100], showticklabels=False)),
                      legend=dict(orientation="h", y=-0.1), paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ------------------------------------------------------- Cards semafóricos
def card(titulo, estado, corpo, dica=""):
    bg, bd, ic = ESTADOS[estado]
    tip = f' <span title="{dica}" style="cursor:help">ℹ️</span>' if dica else ""
    return (f'<div class="card" style="background:{bg};border-left:8px solid {bd}">'
            f"<h4>{ic} {titulo}{tip}</h4>{corpo}</div>")


def lista(itens):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in itens) + "</ul>"


def estado_especialista(esp):
    neg = [e for e in esp if not e.startswith("R6")]
    if not neg:
        return "ok"
    return "alerta" if len(neg) >= 3 or any(e.startswith("R2") for e in neg) else "atencao"


def estado_knn(viz):
    risco = (viz.resultado == "Risco de Reprovação").sum()
    atenc = (viz.resultado == "Atenção / Recuperação").sum()
    return "alerta" if risco >= 2 else "atencao" if risco == 1 or atenc >= 2 else "ok"


# ------------------------------------------------------- Navegação (sidebar)
st.session_state.setdefault("pagina", "Início")
st.session_state.setdefault("conf_min", 0.6)
st.session_state.setdefault("n_hist", 400)
st.session_state.setdefault("chat", [])
res = st.session_state.get("res")

with st.sidebar:
    st.markdown("## 🎓 EduAI")
    for p in ["Início", "Configurações", "Sobre", "Sair"]:
        if st.button(p, use_container_width=True, key=f"nav_{p}"):
            st.session_state.pagina = p
    pagina = st.session_state.pagina


if pagina == "Sair":
    st.title("Até logo! 👋")
    st.info("Sessão encerrada. Clique em 'Início' no menu para voltar.")
    st.stop()

if pagina == "Sobre":
    st.title("Sobre")
    st.write("Sistema que integra Sistema Especialista, Lógica Fuzzy, Árvore de Decisão, KNN, "
             "K-Means, Regressão Linear e Apriori, consolidados por um Agente Inteligente. "
             "O histórico de alunos é **simulado** para fins demonstrativos.")
    st.stop()

if pagina == "Configurações":
    st.title("Configurações")
    st.session_state.n_hist = st.slider("Alunos no histórico simulado", 100, 1000,
                                        st.session_state.n_hist, 50)
    st.session_state.conf_min = st.slider("Percentual mínimo de reprovação para alertar (Apriori)",
                                          0.3, 0.95, st.session_state.conf_min, 0.05)
    st.stop()

# ------------------------------------------------------- Início: entrada
st.title("Análise Educacional Inteligente")
modelos = treinar_modelos(st.session_state.n_hist)

ca, cb = st.columns(2)
with ca.container(border=True):
    st.markdown("#### 📘 Desempenho Acadêmico")
    nota = st.number_input("Nota atual (0-10)", 0.0, 10.0, 5.5, 0.1)
    tarefas = st.slider("Entrega de tarefas (%)", 0, 100, 45)
with cb.container(border=True):
    st.markdown("#### 🕒 Assiduidade e Hábitos")
    freq = st.slider("Frequência escolar (%)", 0, 100, 80)
    estudo = st.number_input("Estudo extra-classe (horas/semana)", 0.0, 20.0, 6.0, 0.5)
    seg = st.toggle("Costuma faltar às segundas-feiras?")

if st.button("Analisar", type="primary", use_container_width=True):
    st.session_state.res = agente(nota, freq, tarefas, estudo, seg, modelos, st.session_state.conf_min)
    st.session_state.chat = []
    st.rerun()

if res is None:
    st.info("Preencha os dados e clique em **Analisar**.")
    st.stop()

# ------------------------------------------------------- 1) CONCLUSÃO PRIMEIRO
estado_geral = NIVEL_ESTADO[res["nivel"]]
_, cor_borda, icone = ESTADOS[estado_geral]
st.markdown(f"""
<div class="agente" style="background:{cor_borda}">
<h3>🤖 Decisão do Agente Inteligente — Risco {res['nivel']}</h3>
<p>{res['diagnostico']}</p></div>""", unsafe_allow_html=True)

g, r_, t = st.columns(3)
g.plotly_chart(grafico_gauge(res["score"]), use_container_width=True)
r_.plotly_chart(grafico_radar(res["entrada"]), use_container_width=True)
t.markdown(card("Abordagem pedagógica sugerida", estado_geral,
                f"{res['abordagem']}<br><br><b>Ponto mais fraco:</b> {res['ponto_fraco']}"),
           unsafe_allow_html=True)
st.caption("Radar: linha tracejada = perfil ideal. Estudo extra: 10 h/semana equivale a 100%.")

# ------------------------------------------------------- 2) JUSTIFICATIVAS
esp, (v, nivel, dica), (cat, prob) = res["especialista"], res["fuzzy"], res["arvore"]
viz, ap = res["knn"], res["apriori"]

with st.expander("🔍 Por que o sistema chegou a esta decisão?", expanded=True):
    a, b = st.columns(2)
    a.markdown(card("Sistema Especialista", estado_especialista(esp),
                    lista(esp) if esp else "Nenhuma regra de alerta foi acionada.",
                    "Regras pedagógicas fixas definidas por especialistas."), unsafe_allow_html=True)
    b.markdown(card("Nível de atenção necessário", {"Baixa": "ok", "Média": "atencao", "Alta": "alerta"}[nivel],
                    f"Necessidade de intervenção: <b>{nivel}</b><br><small>{dica}</small>",
                    "Lógica fuzzy: trata zonas cinzentas, como nota média com pouco estudo."),
               unsafe_allow_html=True)

    certeza = "alta" if prob >= 0.8 else "média" if prob >= 0.6 else "baixa"
    estado_arv = ("alerta" if cat == "Risco de Reprovação"
                  else "atencao" if cat.startswith("Atenção") else "ok")
    a, b = st.columns(2)
    a.markdown(card("Classificação do aluno", estado_arv,
                    f"Com base em alunos de anos anteriores, o perfil se encaixa em: <b>{cat}</b>."
                    f"<br><small>Grau de certeza: {certeza}.</small>",
                    "A árvore aprende regras simples (ex.: nota baixa + poucas tarefas) a partir do histórico."),
               unsafe_allow_html=True)
    b.markdown(card("Alunos parecidos", estado_knn(viz),
                    "Os 3 alunos do histórico mais semelhantes terminaram assim:" + lista(
                        [f"Nota {r.nota:.1f}, freq. {r.frequencia:.0f}% → final {r.nota_final:.1f} "
                         f"(<b>{r.resultado}</b>)" for r in viz.itertuples()]),
                    "KNN: busca no histórico os casos mais próximos dos dados informados."),
               unsafe_allow_html=True)

    estado_reg = "ok" if res["reg"] >= 7 else "atencao" if res["reg"] >= 5 else "alerta"
    leitura = {"ok": "Projeção positiva.", "atencao": "Projeção em zona de atenção.",
               "alerta": "Projeção abaixo da média de aprovação."}[estado_reg]
    a, b = st.columns(2)
    a.markdown(card("Perfil de comportamento", {"Alto desempenho e engajados": "ok",
                                                "Esforçados com dificuldade": "atencao",
                                                "Desengajados": "alerta"}[res["kmeans"]],
                    f"O aluno se parece com o grupo: <b>{res['kmeans']}</b>.",
                    "K-Means: agrupa alunos com hábitos e resultados semelhantes."), unsafe_allow_html=True)
    b.markdown(card("Nota final prevista", estado_reg,
                    f"Projeção: <b>{res['reg']:.1f} de 10</b>. {leitura}",
                    "Regressão linear: estima a nota do fim do semestre a partir do desempenho atual."),
               unsafe_allow_html=True)

    if ap:
        corpo_ap = lista([
            f"<b>Atenção:</b> {c:.0%} dos alunos históricos com este padrão "
            f"({' + '.join(ITEM_TXT[i] for i in sorted(its))}) acabaram por reprovar. "
            f"<small>(base: {n} alunos)</small>" for its, _, c, n in ap])
        est_ap = "alerta" if max(c for _, _, c, _ in ap) >= 0.75 else "atencao"
    else:
        corpo_ap, est_ap = "Nenhum padrão de risco conhecido do histórico se aplica a este aluno.", "ok"
    st.markdown(card("Padrões de risco no histórico", est_ap, corpo_ap,
                     "Apriori: descobre combinações de comportamentos que costumam levar à reprovação."),
                unsafe_allow_html=True)