import os
import streamlit.components.v1 as components

_RELEASE = False

if not _RELEASE:
    _component_func = components.declare_component(
        "componente", # Mantém o nome declarado
        url="http://localhost:3001",
     )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("componente", path=build_dir)

# --- FUNÇÃO WRAPPER REVISADA ---
# Aceita argumentos específicos em vez de um dicionário 'args'
def my_component(
    tipo: str, # Argumento obrigatório para definir o tipo
    texto: str | None = None, # Específico para tipo='botao'
    valor_retorno: any = None, # Específico para tipo='botao'
    opcoes: list[str] | None = None, # Específico para tipo='navbar'
    # Adicione outros argumentos para futuros tipos aqui, com default=None
    key=None
):
    """Cria uma instância dinâmica do componente.

    Parameters
    ----------
    tipo : str
        Tipo do componente a renderizar ('botao', 'navbar', etc.).
    texto : str, optional
        Texto para o botão.
    valor_retorno : any, optional
        Valor a ser retornado pelo botão.
    opcoes : list[str], optional
        Lista de opções para a navbar.
    key : str or None, optional
        Chave única para o componente.

    Returns
    -------
    any
        Valor retornado pela interação do componente (ex: item clicado).
    """
    # Passa os argumentos diretamente como keyword arguments.
    # O frontend receberá isso dentro de props.args
    component_value = _component_func(
        tipo=tipo,
        texto=texto,
        valor_retorno=valor_retorno,
        opcoes=opcoes,
        # Passe outros argumentos aqui se adicionar mais tipos
        key=key,
        default=None # Define um valor de retorno padrão
    )
    return component_value
