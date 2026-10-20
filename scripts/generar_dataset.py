"""
Machine Learning: generar el dataset.

Idea:
    1. Se simulan muchas partidas donde el "humano" (X) juega al azar.
    2. En cada turno de la IA (O), Minimax calcula cuáles son las mejores jugadas.
    3. Se guarda una fila por posición:  9 casillas  ->  mejor jugada.

Codificación de las casillas (la misma de GameModel.to_tuple):
     1 = ficha de la IA (O),   -1 = ficha del humano (X),   0 = vacía
"""

import os
import random

import pandas as pd

from core.game_model import GameModel, PLAYER_O, PLAYER_X
from core.minimax_agent import MinimaxAgent

DATASET_FILE: str = "data/dataset.csv"

FEATURE_COLUMNS: list[str] = [f"c{i}" for i in range(9)]  # Entradas: las 9 casillas
LABEL_COLUMN: str = "best_move"                           # Salida: la jugada (0-8)
OPTIMAL_COLUMN: str = "optimal_moves"                     # Todas las jugadas óptimas, ej. "0|4"


def generate_dataset(n_games: int = 5000, explore_prob: float = 0.5) -> tuple[pd.DataFrame, int]:
    """Simula partidas y devuelve (dataset, cantidad de turnos de la IA simulados).

    Args:
        n_games: Número de partidas a simular.
        explore_prob: Probabilidad de que la IA juegue al azar durante la simulación.
            Sirve para visitar más posiciones distintas; la etiqueta SIEMPRE la calcula Minimax.
    """
    teacher = MinimaxAgent(PLAYER_O, PLAYER_X)       # Minimax completo = el "maestro"
    labels: dict[tuple[int, ...], list[int]] = {}    # posición -> mejores jugadas
    ai_turns = 0

    for _ in range(n_games):
        game = GameModel()
        while not game.is_terminal():
            # Turno del humano simulado: jugada aleatoria
            if game.current_player == PLAYER_X:
                game.make_move(random.choice(game.get_available_moves()))
                continue

            # Turno de la IA: Minimax calcula las mejores jugadas de ESTA posición
            position = game.to_tuple(PLAYER_O, PLAYER_X)
            if position not in labels:               # Si ya la vimos, no se recalcula
                labels[position] = teacher.get_best_moves(game)
            ai_turns += 1

            # Avanzar la partida: a veces al azar (explorar), a veces con una jugada óptima
            if random.random() < explore_prob:
                game.make_move(random.choice(game.get_available_moves()))
            else:
                game.make_move(random.choice(labels[position]))

    # Un diccionario guarda cada posición una sola vez: así el dataset queda sin repetidas
    rows = []
    for position, best_moves in labels.items():
        row = dict(zip(FEATURE_COLUMNS, position))
        row[LABEL_COLUMN] = best_moves[0]                              # Regla fija: la de menor índice
        row[OPTIMAL_COLUMN] = "|".join(str(move) for move in best_moves)
        rows.append(row)
    return pd.DataFrame(rows), ai_turns


def main() -> None:
    random.seed(42)  # Para obtener siempre el mismo dataset
    print("Simulando partidas (unos segundos)...")
    df, ai_turns = generate_dataset()

    os.makedirs(os.path.dirname(DATASET_FILE), exist_ok=True)
    df.to_csv(DATASET_FILE, index=False)

    print(f"Turnos de la IA simulados:  {ai_turns}")
    print(f"Posiciones únicas (filas):  {len(df)}")
    print(f"Dataset guardado en:        {DATASET_FILE}\n")
    print(df.head(5).to_string())


if __name__ == "__main__":
    main()