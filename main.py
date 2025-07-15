import streamlit as st
import os
import sys
import conteudo

def load_css(file_name: str):
    """Função para carregar CSS externo e aplicá-lo no Streamlit."""
    with open(file_name, "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# --- AJUSTE PARA NOVA ESTRUTURA ---
# Diretório raiz do projeto (onde main.py está)
project_root = os.path.dirname(os.path.abspath(__file__))

# Caminho para a pasta 'componentes'
componentes_dir = os.path.join(project_root, 'componentes')

# Adiciona a pasta 'componentes' ao sys.path para que o Python encontre 'botao' e 'navbar'
if componentes_dir not in sys.path:
    sys.path.append(componentes_dir)

# Importa as funções dos componentes das pastas locais
# from botao.my_component import my_component as botao_component # Renomeia para clareza
# from navbar.my_component import my_component as navbar_component # Renomeia para clareza
from componente.my_component import my_component as componente
# --- FIM DO AJUSTE ---

st.set_page_config(layout="wide")
load_css("assets/css/styles.css")

# st.title("App Principal com Múltiplos Componentes")

# --- Inicialização do Session State para a página ---
if 'page' not in st.session_state:
    st.session_state.page = "Home"
# --- Fim da Inicialização ---

# --- Renderização da Navbar e Atualização do Estado ---
# st.header("Navegação")
nav_items = ["Home", "Análise", "Configurações", "Sobre"]
nav_icons = ["home", "bar_chart", "settings", "info"]

# Chama o componente com ícones
# clicked_page = navbar_component(items=nav_items, icons=nav_icons,user_name="", key="main_navbar")

args_navbar = {
    "tipo": "navbar2",
    "opcoes": ["Resumo",'Introdução', "Fundamentos de ML", "Ciência de Dados", "Redes Neurais", "Aplicações"],
}

clicked_page = componente(**args_navbar, key="navbar_dinamica")
# Atualiza o estado da página se um item diferente foi clicado
if clicked_page is not None and clicked_page != st.session_state.page:
    st.session_state.page = clicked_page
    # O Streamlit geralmente faz o rerun automaticamente ao mudar o session_state
    # st.experimental_rerun() # Descomente se o rerun automático não ocorrer
# --- Fim da Navbar ---

if st.session_state.page == "Resumo":
    st.markdown(conteudo.resumo,unsafe_allow_html=True)

if st.session_state.page == "Introdução":
    st.markdown(conteudo.introducao,unsafe_allow_html=True)

if st.session_state.page == "Fundamentos de ML":
    st.markdown("## Fundamentos do Machine Learning Supervisionado", unsafe_allow_html=True)
    content = conteudo.fundamentos_dct
    cols = st.columns([0.15,0.85])
    for cnt in content:
        atalho = cnt['titulo'].replace(" ", "_").lower()
        cols[0].markdown(f"[{cnt['titulo']}](#{atalho})",unsafe_allow_html=True)
        with cols[1].expander(cnt['titulo'], expanded=False):
            st.markdown(f'''<a name={atalho}></a>''', unsafe_allow_html=True)
            st.markdown(cnt['conteudo'], unsafe_allow_html=True) #cnt.split("\n")[1:], unsafe_allow_html=True)
    
if st.session_state.page == "Ciência de Dados":
    content = conteudo.ciencia.split("####")
    cols = st.columns([0.15,0.85])
    cols[0].markdown("[Principais conceitos](#principais_conceitos)",unsafe_allow_html=True)
    cols[0].markdown("[Treinamento](#treinamento)",unsafe_allow_html=True)
    cols[1].markdown("## Fundamentos do Machine Learning Supervisionado", unsafe_allow_html=True)
    with cols[1].expander('#### Introdução',expanded=True):
        st.markdown(content[0], unsafe_allow_html=True)
    for cnt in content[1:]:
        with cols[1].expander(cnt.split("\n")[0].strip(), expanded=False):
            st.markdown("\n".join(cnt.split("\n")[1:]), unsafe_allow_html=True) #cnt.split("\n")[1:], unsafe_allow_html=True)
    # cols[1].markdown(conteudo.fundamentos,unsafe_allow_html=True)    


# args_btn = {
#     "tipo": "botao",
#     "texto": "clique aqui",
# }
# clicked_page2 = componente(**args_btn, key="btn_dinamica")
# st.write(f"Você clicou em: {clicked_page2}")
