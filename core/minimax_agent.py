"""
Idea central (Backtracking):
    1. HACER la jugada       -> board.make_move(move)
    2. EXPLORAR recursivamente lo que puede pasar después
    3. DESHACER la jugada    -> board.undo_move()
Así se prueban todas las jugadas sobre el MISMO tablero, sin copiarlo.

El nivel de dificultad se controla con `max_depth`:
    - None  -> explora todo el árbol (Difícil, imbatible)
    - n     -> mira solo n jugadas hacia adelante (Medio)
"""

import math
import random

from core.game_model import GameModel, PLAYER_O, PLAYER_X

WIN_SCORE: int = 10  # Valor base de una victoria (se ajusta según la profundidad)


class MinimaxAgent:
    """Elige la mejor jugada para `ai_player` usando Minimax."""

    def __init__(
        self,
        ai_player: str = PLAYER_O,
        human_player: str = PLAYER_X,
        difficulty: str = "dificil",
    ) -> None:
        """
        Args:
            ai_player: Símbolo que juega la IA ("O" por defecto).
            human_player: Símbolo del oponente ("X" por defecto).
            max_depth: Jugadas máximas a explorar. None = sin límite.
        """
        self.ai_player: str = ai_player
        self.human_player: str = human_player
        self.difficulty: str = difficulty
        self.max_depth: int | None = None
        self.nodes_evaluated: int = 0  # Métrica educativa: nodos visitados

    def set_difficulty(self, difficulty: str) -> None:
        """Cambia el nivel de dificultad y ajusta la profundidad máxima."""
        self.difficulty = difficulty
        if difficulty == "facil":
            self.max_depth = 0  # No explorará el árbol (movimientos aleatorios)
        elif difficulty == "medio":
            self.max_depth = 2  # Limita la profundidad a 2 jugadas
        else:
            self.max_depth = None  # Difícil: sin límite, árbol completo

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def choose_move(self,board: GameModel) -> int:
        """Devuelve el índice (0-8) de la mejor jugada para la IA.

        Si varias jugadas valen lo mismo, elige una al azar (partidas más variadas).
        El tablero queda EXACTAMENTE igual que antes de llamar al método.
        """
        if board.current_player != self.ai_player:
            raise ValueError(f"No es el turno de la IA ({self.ai_player}).")
        
        moves = board.get_available_moves()
        if not moves:
            raise ValueError("No hay movimientos disponibles.")

        self.nodes_evaluated = 0

        # NIVEL FÁCIL: Movimiento totalmente aleatorio
        if self.difficulty == "facil":
            self.nodes_evaluated = 0
            return random.choice(moves)
        # NIVEL DICIFIL Y MEDIO
        return random.choice(self.get_best_moves(board))
    
    def get_best_moves(self, board: GameModel) -> list[int]:
        """Devuelve el índice (0-8) de la mejor jugada para la IA.

        El tablero queda EXACTAMENTE igual que antes de llamar al método.

        Raises:
            ValueError: si no es el turno de la IA o no hay casillas libres.
        """
        if board.current_player != self.ai_player:
            raise ValueError(f"No es el turno de la IA ({self.ai_player}).")
        moves = board.get_available_moves()
        if not moves:
            raise ValueError("No hay movimientos disponibles.")

        self.nodes_evaluated = 0
        best_score: float = -math.inf
        best_moves: list[int] = []

        for move in moves:
            board.make_move(move)                                   # 1. HACER
            score = self._minimax(board, depth=1, is_maximizing=False)  # 2. EXPLORAR
            board.undo_move()                                       # 3. DESHACER

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        # Si varias jugadas valen lo mismo, elegimos una al azar (partidas más variadas)
        return best_moves

    # ------------------------------------------------------------------
    # Algoritmo recursivo
    # ------------------------------------------------------------------

    def _minimax(self, board: GameModel, depth: int, is_maximizing: bool) -> float:
        """Evalúa un nodo del árbol de juego.

        Args:
            board: Tablero (se modifica y se restaura: backtracking).
            depth: Jugadas hechas desde la posición original.
            is_maximizing: True si le toca a la IA (busca el score MÁS alto);
                           False si le toca al humano (busca el MÁS bajo).
        """
        self.nodes_evaluated += 1

        # ---- Casos base (nodos hoja) ----
        winner = board.get_winner()
        if winner == self.ai_player:
            return WIN_SCORE - depth      # Gana la IA: mejor si gana rápido
        if winner == self.human_player:
            return depth - WIN_SCORE      # Gana el humano: mejor si tarda más
        moves = board.get_available_moves()
        if not moves:
            return 0                      # Empate
        if self.max_depth is not None and depth >= self.max_depth:
            return 0                      # Límite de profundidad (nivel Medio)

        # ---- Caso recursivo ----
        if is_maximizing:
            best: float = -math.inf
            for move in moves:
                board.make_move(move)                                   # HACER
                best = max(best, self._minimax(board, depth + 1, False))
                board.undo_move()                                       # DESHACER
            return best

        best = math.inf
        for move in moves:
            board.make_move(move)                                       # HACER
            best = min(best, self._minimax(board, depth + 1, True))
            board.undo_move()                                           # DESHACER
        return best