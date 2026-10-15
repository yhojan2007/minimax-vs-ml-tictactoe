"""
view/gui_view.py
================
Vista del juego (Tkinter).

Responsabilidades:
- Renderizar el tablero 3x3
- Mostrar selector de modo (Humano / Minimax / ML)
- Mostrar métricas en tiempo real
- Capturar clics y delegar al Controlador

NO contiene lógica del juego.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from enum import Enum


# ============================================================
# Constantes de estilo
# ============================================================

COLOR_BG = "#ffffff"
COLOR_PANEL = "#f5f5f5"
COLOR_BOARD = "#ffffff"
COLOR_CELL = "#f0f0f0"
COLOR_CELL_HOVER = "#e0e0e0"
COLOR_TEXT = "#000000"
COLOR_X = "#1565C0"       # Azul para X (Humano)
COLOR_O = "#C62828"       # Rojo para O (IA)
COLOR_WIN = "#6AF678"     # Amarillo dorado para línea ganadora
COLOR_ACCENT = "#555555"

FONT_TITLE = ("Segoe UI", 18, "bold")
FONT_SUBTITLE = ("Segoe UI", 11)
FONT_CELL = ("Segoe UI", 32, "bold")
FONT_STATUS = ("Segoe UI", 12, "bold")
FONT_METRIC = ("Consolas", 10)


class GameMode(Enum):
    """Modos de juego disponibles."""
    HUMAN_VS_HUMAN = "Humano vs Humano"
    HUMAN_VS_MINIMAX = "Humano vs Minimax"
    HUMAN_VS_ML = "Humano vs ML"


# ============================================================
# Clase principal: GUIView
# ============================================================

class GUIView:
    """
    Vista principal del juego.

    El Controlador debe asignar callbacks a:
        - on_cell_click(index)
        - on_mode_change(mode)
        - on_reset()
        - on_new_game()
    """

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("🎮 Tres en Raya — Minimax vs Machine Learning")
        self.root.configure(bg=COLOR_BG)
        self.root.resizable(False, False)

        # Callbacks (asignados por el Controlador)
        self.on_cell_click: callable[[int], None] | None = None # Callback para clic en celda (0-8)
        self.on_mode_change: callable[[GameMode], None] | None = None # Callback para cambio de modo
        self.on_reset: callable[[], None] | None = None # Callback para reiniciar partida
        self.on_new_game: callable[[], None] | None = None # Callback para nueva partida

        # Referencias a widgets
        self.cell_buttons: dict[int, tk.Button] = {}       # Botones de las celdas (0-8)
        self.status_label: tk.Label | None = None      # Label de estado (Turno, Ganador, etc.)
        self.mode_var: tk.StringVar | None = None   # Variable para el modo de juego seleccionado
        self.metric_labels: dict[str, tk.Label] = {}    # Labels para métricas de IA (nodes, time, accuracy)

        self._build_ui()
        self._center_window()

    # --------------------------------------------------------
    # Construcción de la interfaz
    # --------------------------------------------------------

    def _build_ui(self) -> None:
        """Construye toda la interfaz."""
        # -------- Contenedor principal --------
        main_frame = tk.Frame(self.root, bg=COLOR_BG, padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)

        # -------- Título --------
        title = tk.Label(
            main_frame,
            text="🎮 Tres en Raya",
            font=FONT_TITLE,
            bg=COLOR_BG,
            fg=COLOR_TEXT,
        )
        title.grid(row=0, column=0, columnspan=2, pady=(0, 4))

        subtitle = tk.Label(
            main_frame,
            text="Minimax con backtracking  y  MachineLearning",
            font=FONT_SUBTITLE,
            bg=COLOR_BG,
            fg=COLOR_ACCENT,
        )
        subtitle.grid(row=1, column=0, columnspan=2, pady=(0, 16))

        # -------- Tablero (izquierda) --------
        board_frame = tk.Frame(main_frame, bg=COLOR_BOARD, padx=10, pady=10)
        board_frame.grid(row=2, column=0, padx=(0, 16), sticky="n")

        for i in range(9):
            row, col = divmod(i, 3)
            btn = tk.Button(
                board_frame,
                text="",
                font=FONT_CELL,
                width=4,
                height=2,
                bg=COLOR_CELL,
                fg=COLOR_TEXT,
                activebackground=COLOR_CELL_HOVER,
                activeforeground=COLOR_TEXT,
                relief="flat",
                bd=0,
                cursor="hand2",
                command=lambda idx=i: self._handle_cell_click(idx),
            )
            btn.grid(row=row, column=col, padx=3, pady=3)
            btn.bind("<Enter>", lambda e, b=btn: self._on_hover(b, True))
            btn.bind("<Leave>", lambda e, b=btn: self._on_hover(b, False))
            self.cell_buttons[i] = btn

        # -------- Panel lateral (derecha) --------
        panel = tk.Frame(main_frame, bg=COLOR_PANEL, padx=16, pady=16)
        panel.grid(row=2, column=1, sticky="n")

        # --- Selector de modo ---
        tk.Label(
            panel,
            text="Modo de Juego",
            font=("Segoe UI", 12, "bold"),
            bg=COLOR_PANEL,
            fg=COLOR_TEXT,
        ).pack(anchor="w", pady=(0, 8))

        self.mode_var = tk.StringVar(value=GameMode.HUMAN_VS_HUMAN.value)
        for mode in GameMode:
            rb = tk.Radiobutton(
                panel,
                text=mode.value,
                variable=self.mode_var,
                value=mode.value,
                command=self._handle_mode_change,
                bg=COLOR_PANEL,
                fg=COLOR_TEXT,
                selectcolor=COLOR_BOARD,
                activebackground=COLOR_PANEL,
                activeforeground=COLOR_ACCENT,
                font=("Segoe UI", 10),
                cursor="hand2",
            )
            rb.pack(anchor="w", pady=2)

        # --- Separador ---
        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=12)

        # --- Estado de la partida ---
        tk.Label(
            panel,
            text="Estado",
            font=("Segoe UI", 12, "bold"),
            bg=COLOR_PANEL,
            fg=COLOR_TEXT,
        ).pack(anchor="w", pady=(0, 8))

        self.status_label = tk.Label(
            panel,
            text="Turno: X",
            font=FONT_STATUS,
            bg=COLOR_PANEL,
            fg=COLOR_X,
            wraplength=200,
            justify="left",
        )
        self.status_label.pack(anchor="w", pady=(0, 12))

        # --- Métricas ---
        tk.Label(
            panel,
            text="Metricas de IA",
            font=("Segoe UI", 12, "bold"),
            bg=COLOR_PANEL,
            fg=COLOR_TEXT,
        ).pack(anchor="w", pady=(0, 8))

        metrics_frame = tk.Frame(panel, bg=COLOR_PANEL)
        metrics_frame.pack(fill="x", pady=(0, 12))

        metrics = [
            ("Nodos explorados:", "nodes"),
            ("Tiempo respuesta:", "time"),
            ("Precisión ML:", "accuracy"),
        ]
        for label_text, key in metrics:
            row = tk.Frame(metrics_frame, bg=COLOR_PANEL)
            row.pack(fill="x", pady=2)
            tk.Label(
                row,
                text=label_text,
                font=FONT_METRIC,
                bg=COLOR_PANEL,
                fg=COLOR_TEXT,
            ).pack(side="left")
            value_lbl = tk.Label(
                row,
                text="—",
                font=FONT_METRIC,
                bg=COLOR_PANEL,
                fg=COLOR_ACCENT,
            )
            value_lbl.pack(side="right")
            self.metric_labels[key] = value_lbl

        # --- Separador ---
        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=12)

        # --- Botones de acción ---

        new_btn = tk.Button(
            panel,
            text="Nueva Partida",
            font=("Segoe UI", 10),
            bg=COLOR_CELL,
            fg=COLOR_TEXT,
            activebackground=COLOR_CELL_HOVER,
            activeforeground=COLOR_TEXT,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=10,
            pady=6,
            command=self._handle_reset,
        )
        new_btn.pack(fill="x")

        # --- Footer ---
        footer = tk.Label(
            panel,
            text="Aqui voy a poner los botones nivel ia",
            font=("Segoe UI", 8),
            bg=COLOR_PANEL,
            fg="#888888",
        )
        footer.pack(side="bottom", pady=(16, 0))

    def _center_window(self) -> None:
        """Centra la ventana en la pantalla."""
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"+{x}+{y}")

    # --------------------------------------------------------
    # Handlers internos (delegan al Controlador)
    # --------------------------------------------------------

    def _handle_cell_click(self, index: int) -> None:
        if self.on_cell_click:
            self.on_cell_click(index)

    def _handle_mode_change(self) -> None:
        if self.on_mode_change and self.mode_var:
            mode_str = self.mode_var.get()
            for mode in GameMode:
                if mode.value == mode_str:
                    self.on_mode_change(mode)
                    break

    def _handle_reset(self) -> None:
        if self.on_reset:
            self.on_reset()

    def _handle_new_game(self) -> None:
        if self.on_new_game:
            self.on_new_game()
        else:
            self._handle_reset()

    def _on_hover(self, btn: tk.Button, entering: bool) -> None:
        """Efecto hover sobre las celdas."""
        if btn["state"] == "disabled":
            return
        btn.configure(bg=COLOR_CELL_HOVER if entering else COLOR_CELL)

    # --------------------------------------------------------
    # API pública para el Controlador
    # --------------------------------------------------------

    def render_board(self,board: list[str], winning_line: tuple[int, int, int] | None, game_over: bool = False) -> None:
        """Actualiza el tablero. `board` es una lista de 9 strings: "", "X" u "O"."""
        colors = {"": COLOR_TEXT, "X": COLOR_X, "O": COLOR_O}

        for i, symbol in enumerate(board):
            btn = self.cell_buttons[i]
            btn.configure(
                text=symbol,
                fg=colors[symbol],
                disabledforeground=colors[symbol],  # si no, X y O se ven grises al bloquearse
                # Solo se puede pulsar una celda vacía y mientras la partida siga
                state="normal" if symbol == "" and not game_over else "disabled",
                bg=COLOR_WIN if winning_line and i in winning_line else COLOR_CELL,
            )

    def set_status(self, text: str, color: str = COLOR_TEXT) -> None:
        """Actualiza el mensaje de estado."""
        if self.status_label:
            self.status_label.configure(text=text, fg=color)

    def set_metric(self, key: str, value: str) -> None:
        """Actualiza una métrica específica ('nodes', 'time', 'accuracy')."""
        if key in self.metric_labels:
            self.metric_labels[key].configure(text=value)

    def reset_metrics(self) -> None:
        """Limpia todas las métricas."""
        for lbl in self.metric_labels.values():
            lbl.configure(text="—")

    def set_board_enabled(self, enabled: bool) -> None:
        """Habilita/deshabilita los clics del tablero."""
        for btn in self.cell_buttons.values():
            btn.configure(state="normal" if enabled else "disabled")

    def get_selected_mode(self) -> GameMode:
        """Devuelve el modo seleccionado actualmente."""
        mode_str = self.mode_var.get() if self.mode_var else ""
        for mode in GameMode:
            if mode.value == mode_str:
                return mode
        return GameMode.HUMAN_VS_MINIMAX

    # --------------------------------------------------------
    # Ciclo de vida
    # --------------------------------------------------------

    def run(self) -> None:
        """Inicia el bucle principal de Tkinter."""
        self.root.mainloop()

    def destroy(self) -> None:
        """Cierra la ventana."""
        self.root.destroy()