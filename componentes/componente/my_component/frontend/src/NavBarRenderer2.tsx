import React, { useEffect, useState, useCallback } from 'react';
import { Streamlit } from 'streamlit-component-lib';

interface NavbarRendererProps {
  items: string[];
  icons?: string[];
  selected?: string;
}

export function NavbarRenderer2({ items, icons = [], selected }: NavbarRendererProps): React.ReactElement {
  const [activeItem, setActiveItem] = useState<string>(selected ?? "");
  const [isDarkMode, setIsDarkMode] = useState<boolean>(false);
  const [isSmallScreen, setIsSmallScreen] = useState<boolean>(false);

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

  useEffect(() => {
    setActiveItem(selected ?? "");
  }, [selected]);

  const onItemClick = useCallback((item: string): void => {
    setActiveItem(item);
    Streamlit.setComponentValue(item);
  }, []);

  if (!Array.isArray(items)) {
    console.error("NavbarRenderer: prop 'items' não é um array.");
    return <div>Erro interno: items da navbar inválidos.</div>;
  }

  const bgColor = isDarkMode ? "#1e1e1e" : "#f8f9fa";
  const textColor = isDarkMode ? "#ffffff" : "#000000";
  const activeColor = isDarkMode ? "#00c0ff" : "#007bff";

  return (
    <nav
      className="navbar navbar-expand"
      style={{
        padding: "0.5rem 1rem",
        borderRadius: "0.5rem",
        backgroundColor: bgColor,
        color: textColor,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        flexWrap: "wrap",
      }}
    >
      <ul
        className="navbar-nav d-flex w-100"
        style={{
          display: "flex",
          flexDirection: "row",
          justifyContent: "space-between",
          width: "100%",
          margin: 0,
          padding: 0,
          listStyleType: "none",
        }}
      >
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
    </nav>
  );
}
