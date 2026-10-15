"""
Modelo del núcleo del juego (Modelo MVC) basado en una Matriz 3x3.

Contiene:
- Representación estructural del tablero (matriz 3x3)
- Validaciones de movimientos
- Detección de victorias (filas, columnas, diagonales)
- Control de empates y estados terminales
- Soporte para Backtracking (hacer/deshacer movimientos)
- Utilidades para Minimax y Machine Learning

"""

from enum import Enum


# ============================================================
# Constantes del juego
# ============================================================

EMPTY: str = ""
PLAYER_X: str = "X"
PLAYER_O: str = "O"

# Combinaciones ganadoras como índices lineales (0-8)
# (Se usan para get_winning_line y para la UI)
WINNING_LINES: list[tuple[int, int, int]] = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # Filas
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # Columnas
    (0, 4, 8), (2, 4, 6),              # Diagonales
]

# Mapeo símbolo → valor numérico (para ML)
SYMBOL_TO_INT: dict[str, int] = {EMPTY: 0, PLAYER_X: 1, PLAYER_O: -1}


class GameState(Enum):
    """Estados posibles de la partida."""
    IN_PROGRESS = "in_progress"
    X_WINS = "x_wins"
    O_WINS = "o_wins"
    DRAW = "draw"

    def __str__(self) -> str:
        return {
            GameState.IN_PROGRESS: "En curso",
            GameState.X_WINS: "Ganó X",
            GameState.O_WINS: "Ganó O",
            GameState.DRAW: "Empate",
        }[self]


# ============================================================
# Clase principal: GameModel
# ============================================================

class GameModel:
    """
    Representa el tablero del Tres en Raya con una matriz 3x3
    y gestiona todas las reglas de la partida.

    Atributos:
        board (list[list[str]]): Matriz 3x3 con "", "X" o "O".
        current_player (str): Jugador con el turno actual.
        game_over (bool): True si la partida terminó.
        state (GameState): Estado actual de la partida.
        move_history (list[tuple[int, int, str]]): Historial de movimientos
                                                    (fila, col, símbolo) para Backtracking.
    """

    def __init__(self, starting_player: str = PLAYER_X) -> None:
        self.starting_player: str = starting_player
        self.board: list[list[str]] = []
        self.current_player: str = starting_player
        self.game_over: bool = False
        self.state: GameState = GameState.IN_PROGRESS
        self.move_history: list[tuple[int, int, str]] = []
        self.reset()

    # --------------------------------------------------------
    # Reinicio
    # --------------------------------------------------------

    def reset(self) -> None:
        """Reinicia la matriz 3x3 y todos los estados lógicos."""
        self.board = [[EMPTY for _ in range(3)] for _ in range(3)]
        self.current_player = self.starting_player
        self.game_over = False
        self.state = GameState.IN_PROGRESS
        self.move_history.clear()

    # --------------------------------------------------------
    # Consultas de estado
    # --------------------------------------------------------

    def get_cell(self, row: int, col: int) -> str:
        """Devuelve el valor de una celda ("" , "X" o "O")."""
        return self.board[row][col]

    def get_flat_board(self) -> list[str]:
        """
        Convierte la matriz 3x3 en una lista lineal de 9 elementos.
        Útil para que Tkinter pinte los botones fácilmente.
        ["", "X", "O", "", "", "", "X", "", "O"]
        """
        return [cell for row in self.board for cell in row]

        #     0    1    2
        #     3    4    5
        #     6    7    8

    def get_available_moves(self) -> list[int]:
        """
        Devuelve los índices lineales (0-8) de las celdas vacías.
        Esencial para Minimax. 
        """
        return [
            r * 3 + c
            for r in range(3)
            for c in range(3)
            if self.board[r][c] == EMPTY
        ]

    def is_full(self) -> bool:
        """True si no quedan celdas vacías."""
        return all(cell != EMPTY for row in self.board for cell in row)

    def is_terminal(self) -> bool:
        """True si la partida terminó (victoria o empate)."""
        return self.game_over

    def is_valid_move(self, index: int) -> bool:
        """Valida que el índice esté en rango y la celda esté vacía."""
        if not 0 <= index <= 8:
            return False
        row, col = index // 3, index % 3
        return self.board[row][col] == EMPTY

    # --------------------------------------------------------
    # Modificación del estado (con Backtracking)
    # --------------------------------------------------------

    def make_move(self, index: int) -> bool:
        """
        Realiza un movimiento en el juego real:
        coloca la ficha, evalúa el estado y alterna el turno.

        Args:
            index: Posición lineal (0-8).

        Returns:
            True si el movimiento fue válido.
        """
        if self.game_over:
            return False
        if not 0 <= index <= 8:
            return False

        row = index // 3
        col = index % 3
        if self.board[row][col] != EMPTY:
            return False

        # Colocar ficha
        self.board[row][col] = self.current_player
        self.move_history.append((row, col, self.current_player))

        # Evaluar estado
        if self.check_winner(self.current_player):
            self.game_over = True
            self.state = (
                GameState.X_WINS if self.current_player == PLAYER_X
                else GameState.O_WINS
            )
        elif self.is_full():
            self.game_over = True
            self.state = GameState.DRAW
        else: # Alternar turno
            if self.current_player == PLAYER_X:
                self.current_player = PLAYER_O
            else:
                self.current_player = PLAYER_X

        return True

    def undo_move(self, index: int | None = None) -> bool:
        """
        Deshace un movimiento (BACKTRACKING).

        Args:
            index: Si se especifica, deshace ese movimiento (0-8).
                   Si es None, deshace el último.

        Returns:
            True si se pudo deshacer.
        """
        if not self.move_history:
            return False

        if index is None:
            row, col, player = self.move_history.pop()
        else:
            if not 0 <= index <= 8:
                return False
            target_row = index // 3
            target_col = index % 3
            # Buscar el último movimiento en esa celda
            for i in range(len(self.move_history) - 1, -1, -1):
                r, c, p = self.move_history[i] # Deshacer el movimiento
                if r == target_row and c == target_col:
                    row, col, player = self.move_history.pop(i) # Eliminar del historial
                    break
            else:
                return False

        # Deshacer
        self.board[row][col] = EMPTY
        self.game_over = False
        self.state = GameState.IN_PROGRESS
        self.current_player = player  # Vuelve a ser el turno de quien movió

        return True

    # --------------------------------------------------------
    # Detección de victoria / empate
    # --------------------------------------------------------

    def check_winner(self, player: str) -> bool:
        """
        Comprueba si `player` ha ganado (filas, columnas o diagonales).
        """
        b = self.board

        # Filas
        for r in range(3):
            if b[r][0] == b[r][1] == b[r][2] == player:
                return True

        # Columnas
        for c in range(3):
            if b[0][c] == b[1][c] == b[2][c] == player:
                return True

        # Diagonal principal
        if b[0][0] == b[1][1] == b[2][2] == player:
            return True

        # Diagonal secundaria
        if b[0][2] == b[1][1] == b[2][0] == player:
            return True

        return False

    def get_winner(self) -> str | None:
        """Devuelve 'X', 'O' o None si nadie ganó."""
        if self.check_winner(PLAYER_X):
            return PLAYER_X
        if self.check_winner(PLAYER_O):
            return PLAYER_O
        return None

    def get_winning_line(self) -> tuple[int, int, int] | None:
        """
        Devuelve la combinación ganadora como índices lineales (0-8),
        o None si no hay ganador. Útil para resaltar en la UI.
        """
        for combo in WINNING_LINES:
            r1, c1 = divmod(combo[0], 3)
            r2, c2 = divmod(combo[1], 3)
            r3, c3 = divmod(combo[2], 3)
            v1 = self.board[r1][c1]
            v2 = self.board[r2][c2]
            v3 = self.board[r3][c3]
            if v1 != EMPTY and v1 == v2 == v3:
                return combo
        return None

    def get_state(self) -> GameState:
        """Devuelve el estado actual de la partida."""
        return self.state

    # --------------------------------------------------------
    # Utilidades para Minimax y ML
    # --------------------------------------------------------

    def copy(self) -> "GameModel":
        """
        Copia profunda del tablero.
        (Alternativa a undo_move() si se prefiere no mutar el original.)
        """
        new = GameModel(starting_player=self.starting_player)
        new.board = [row.copy() for row in self.board]
        new.current_player = self.current_player
        new.game_over = self.game_over
        new.state = self.state
        new.move_history = self.move_history.copy()
        return new

    def to_tuple(self, ai_symbol: str = PLAYER_O, human_symbol: str = PLAYER_X) -> tuple[int, ...]:
        """
        Convierte el tablero en una tupla de 9 enteros para ML.
        Codificación: 1 = IA, -1 = Humano, 0 = vacío.

        Args:
            ai_symbol: Símbolo que representa a la IA (por defecto "O").
            human_symbol: Símbolo que representa al humano (por defecto "X").
        """
        mapping = {ai_symbol: 1, human_symbol: -1, EMPTY: 0}
        return tuple(mapping[cell] for row in self.board for cell in row)

    def from_flat(self, flat: list[str], current_player: str | None = None) -> None:
        """
        Carga un tablero desde una lista plana de 9 elementos.
        Útil para restaurar estados o para el dataset de ML.
        """
        assert len(flat) == 9, "La lista debe tener 9 elementos"
        self.board = [[flat[r * 3 + c] for c in range(3)] for r in range(3)]
        if current_player is not None:
            self.current_player = current_player
        self._recompute_state()

    def _recompute_state(self) -> None:
        """Recalcula game_over, state y winner a partir del tablero actual."""
        if self.check_winner(PLAYER_X):
            self.game_over = True
            self.state = GameState.X_WINS
        elif self.check_winner(PLAYER_O):
            self.game_over = True
            self.state = GameState.O_WINS
        elif self.is_full():
            self.game_over = True
            self.state = GameState.DRAW
        else:
            self.game_over = False
            self.state = GameState.IN_PROGRESS

    # --------------------------------------------------------
    # Representación para debug
    # --------------------------------------------------------

    def __str__(self) -> str:
        symbols = {EMPTY: "."}
        rows = []
        for r in range(3):
            rows.append(" | ".join(symbols.get(self.board[r][c], self.board[r][c])
                                   for c in range(3)))
        return "\n---------\n".join(rows)

    def __repr__(self) -> str:
        return f"Board(board={self.board}, current={self.current_player}, state={self.state})"