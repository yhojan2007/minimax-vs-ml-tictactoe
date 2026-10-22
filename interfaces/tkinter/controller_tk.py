"""
Controlador que conecta Modelo ↔ Vista.

Sincroniza eventos del usuario con el estado del juego
y coordina las llamadas a los agentes de IA.
"""
import time

from core.game_model import GameModel, GameState, PLAYER_X, PLAYER_O
from core.minimax_agent import MinimaxAgent
from core.ml_agent import MLAgent
from interfaces.tkinter.view_tk import GUIView, GameMode, COLOR_ACCENT

class GameController:
    """Coordinador MVC."""

    HUMAN_SYMBOL: str = PLAYER_X
    AI_SYMBOL: str = PLAYER_O
    AI_DELAY_MS: int = 400  # Retraso visual para simular que la IA "piensa"

    def __init__(self, view: GUIView) -> None:
        self.view = view
        self.model = GameModel()
        self.mode: GameMode = view.get_selected_mode()
        self.minimax_agent = MinimaxAgent(self.AI_SYMBOL, self.HUMAN_SYMBOL)
        self.ml_agent: MLAgent | None = None  # Se carga la primera vez que se elige el modo ML
        self.agent = self.minimax_agent       # Agente activo según el modo
        self._ia_job: str | None = None # None =

        # Conectar callbacks
        self.view.on_cell_click = self.handle_cell_click
        self.view.on_mode_change = self.handle_mode_change
        self.view.on_reset = self.handle_reset
        self.view.on_new_game = self.handle_reset  # Vincula el botón "Nuevo juego" al mismo reinicio

        # Inicializar dificultad en difícil por defecto
        self.difficulty: str = "dificil"
        self.minimax_agent = MinimaxAgent(self.AI_SYMBOL, self.HUMAN_SYMBOL, difficulty=self.difficulty)
        
        # Conectar el callback de dificultad con el controlador
        self.view.on_difficulty_change = self.handle_difficulty_change

        self._refresh_view()

    # --------------------------------------------------------
    # Ciclo de vida
    # --------------------------------------------------------

    def start(self) -> None:
        """Inicia la aplicación."""
        self.view.run()

    # --------------------------------------------------------
    # Handlers
    # --------------------------------------------------------

    def handle_cell_click(self, index: int) -> None:
        """El usuario hizo clic en una celda."""
        if self.model.is_terminal() or self._ia_job is not None:
            return

        if not self.model.is_valid_move(index):
            return

        # Aplicar movimiento del jugador humano
        self.model.make_move(index)
        
        # Continuar con el flujo del juego (verifica victoria o turno de la IA)
        self._after_move()

    def handle_mode_change(self, mode: GameMode) -> None:
        """El usuario cambió el modo de juego."""
        # El modo ML necesita el modelo entrenado; si no existe, se vuelve al modo anterior
        if mode == GameMode.HUMAN_VS_ML and not self._load_ml_agent():
            self.view.set_selected_mode(self.mode)
            return

        self.mode = mode
        self.agent = self.ml_agent if mode == GameMode.HUMAN_VS_ML else self.minimax_agent
        
        # Mostrar controles de Minimax solo si el modo es Humano vs Minimax
        is_minimax = (mode == GameMode.HUMAN_VS_MINIMAX)
        self.view.show_minimax_options(is_minimax)

        if is_minimax:
            # Por defecto en Difícil al cambiar a este modo
            self.handle_difficulty_change("dificil")

        self.handle_reset()
        self.view.set_status(f"Modo: {mode.value}", "#cba6f7")

    def handle_reset(self) -> None:
        """Reinicia la partida actual."""
        self._cancel_ai_turn()
        self.model.reset()
        self.view.reset_metrics()
        self._show_ml_accuracy()
        self._refresh_view()

    # --------------------------------------------------------
    # Flujo de turnos y ejecución de la IA
    # --------------------------------------------------------

    def _after_move(self) -> None:
        """Se ejecuta después de CADA jugada (humana o de la IA)."""
        self._refresh_view()
        if self.model.is_terminal():
            return  # La partida terminó, no hacer nada más
        # Si el modo es contra Minimax y le toca el turno a la IA ('O')
        if self._vs_ai() and self.model.current_player == self.AI_SYMBOL:
            self._start_ai_turn()
 
    def _start_ai_turn(self) -> None:
        """Bloquea el tablero y programa la jugada de la IA."""
        self.view.set_board_enabled(False)
        self.view.set_status("La IA TA pensando..", COLOR_ACCENT)
        # Programar la ejecución de la IA después de un retraso visual
        self._ia_job = self.view.root.after(self.AI_DELAY_MS, self._play_ai_move)
 
    def _play_ai_move(self) -> None:
        """La IA elige y ejecuta su jugada; se miden nodos y tiempo."""
        self._ia_job = None # La jugada de la IA ya se ejecutó, no hay trabajo pendiente
 
        start = time.perf_counter()
        move = self.agent.choose_move(self.model)
        elapsed_ms = (time.perf_counter() - start) * 1000
 
        self.model.make_move(move)
        self.view.set_metric("nodes", f"{self.agent.nodes_evaluated:,}")
        self.view.set_metric("time", f"{elapsed_ms:.1f} ms")
        self._after_move()
 
    def _cancel_ai_turn(self) -> None:
        """Cancela la jugada programada de la IA (si existe)."""
        if self._ia_job is not None:
            self.view.root.after_cancel(self._ia_job)
            self._ia_job = None
    
    def _vs_ai(self) -> bool:
        """Devuelve True si el modo de juego activo es contra una IA (Minimax o ML)."""
        return self.mode in (GameMode.HUMAN_VS_MINIMAX, GameMode.HUMAN_VS_ML)

    def _load_ml_agent(self) -> bool:
        """Carga el modelo ML entrenado. Devuelve False si todavía no existe."""
        if self.ml_agent is not None:
            return True
        try:
            self.ml_agent = MLAgent.load()
        except FileNotFoundError:
            self.view.show_error("Modelo no encontrado")
            return False
        return True

    def _show_ml_accuracy(self) -> None:
        """En modo ML, muestra en el panel la precisión medida del modelo."""
        if self.mode == GameMode.HUMAN_VS_ML and self.ml_agent and self.ml_agent.accuracy is not None:
            self.view.set_metric("accuracy", f"{self.ml_agent.accuracy:.1%}")

    def handle_difficulty_change(self, diff: str) -> None:
        """Actualiza el nivel de dificultad en el agente Minimax."""
        self.difficulty = diff
        self.minimax_agent.set_difficulty(diff)
        self.view.set_difficulty(diff)
        self.handle_reset()
    # --------------------------------------------------------
    # Renderizado
    # --------------------------------------------------------

    def _refresh_view(self) -> None:
        """Sincroniza la Vista con el Modelo."""
        self.view.render_board(
            self.model.get_flat_board(),
            self.model.get_winning_line(),
            self.model.is_terminal(),
        )
        state = self.model.get_state()

        if state == GameState.IN_PROGRESS:
            turn = "X" if self.model.current_player == PLAYER_X else "O"
            color = "#4f89e6" if turn == "X" else "#da436d"
            self.view.set_status(f"Turno: {turn}", color)
        elif state == GameState.X_WINS:
            self.view.set_status("¡Ganó X!", "#68e35d")
        elif state == GameState.O_WINS:
            self.view.set_status("¡Ganó O!", "#68e35d")
        elif state == GameState.DRAW:
            self.view.set_status("¡Empate!", "#ebc167")