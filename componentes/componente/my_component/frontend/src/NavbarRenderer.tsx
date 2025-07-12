import React, { useEffect, useState, useCallback } from 'react';
import { Streamlit } from 'streamlit-component-lib';

interface NavbarRendererProps {
  items: string[];
  icons?: string[];
  selected?: string;
  user_name?: string;
}

export function NavbarRenderer({ items, icons = [], selected, user_name = "" }: NavbarRendererProps): React.ReactElement {

  // --- HOOKS TODOS NO TOPO ---
  const [activeItem, setActiveItem] = useState<string>(selected ?? "");
  const [isDarkMode, setIsDarkMode] = useState<boolean>(false);
  const [isSmallScreen, setIsSmallScreen] = useState<boolean>(false);

  // Hook para detectar tema e tamanho da tela
  useEffect(() => {
    const darkModeQuery = window.matchMedia("(prefers-color-scheme: dark)");
    setIsDarkMode(darkModeQuery.matches);
    const handleThemeChange = (e: MediaQueryListEvent) => setIsDarkMode(e.matches);
    darkModeQuery.addEventListener("change", handleThemeChange);

    const checkScreenSize = () => {
      setIsSmallScreen(window.innerWidth < 576);
    };
    checkScreenSize();
    window.addEventListener("resize", checkScreenSize);

    return () => {
      darkModeQuery.removeEventListener("change", handleThemeChange);
      window.removeEventListener("resize", checkScreenSize);
    };
  }, []);

  // Atualiza o item ativo se a prop 'selected' mudar
  useEffect(() => {
    setActiveItem(selected ?? "");
  }, [selected]);

  // Callback para clique no item (agora antes da validação)
  const onItemClick = useCallback((item: string): void => {
    setActiveItem(item);
    Streamlit.setComponentValue(item);
  }, []);
  // --- FIM DOS HOOKS ---

  // Validação DEPOIS dos hooks
  if (!Array.isArray(items)) {
    console.error("NavbarRenderer: prop 'items' não é um array.");
    return <div>Erro interno: items da navbar inválidos.</div>;
  }

  // Lógica restante...
  const isLoggedIn = user_name.trim() !== "";
  const userTooltip = isLoggedIn ? user_name : "Entrar";
  const userAction = isLoggedIn ? "logout" : "login";

  const bgColor = isDarkMode ? "#1e1e1e" : "#f8f9fa";
  const textColor = isDarkMode ? "#ffffff" : "#000000";
  const activeColor = isDarkMode ? "#00c0ff" : "#007bff";

  // Renderização
  return (
    <nav
      className="navbar navbar-expand"
      style={{
        padding: "0.5rem 1rem",
        borderRadius: "0.5rem",
        backgroundColor: bgColor,
        color: textColor,
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        flexWrap: "wrap",
      }}
    >
      <button
        className="btn btn-link"
        onClick={() => Streamlit.setComponentValue(userAction)}
        title={userTooltip}
        style={{
          color: textColor,
          display: "flex",
          alignItems: "center",
          gap: "0.3rem",
          fontSize: "1rem",
          textDecoration: "none",
          border: "none",
          background: "none",
          padding: 0,
          cursor: "pointer",
        }}
      >
        {!isSmallScreen && <span>{isLoggedIn ? user_name : "Entrar"}</span>}
      </button>
      <div className="d-flex justify-content-end">
        <ul className="navbar-nav d-flex flex-row gap-3">
          {items.map((item: string, index: number) => (
            <li className="nav-item" key={item}>
              <button
                className="nav-link btn btn-link"
                onClick={() => onItemClick(item)}
                style={{
                  cursor: "pointer",
                  color: activeItem === item ? activeColor : textColor,
                  fontWeight: activeItem === item ? "bold" : "normal",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.3rem",
                }}
              >
                {isSmallScreen && icons[index] ? (
                  <span
                    className="material-icons"
                    style={{ fontSize: "20px" }}
                    title={item}
                  >
                    {icons[index]}
                  </span>
                ) : (
                  item
                )}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
}
