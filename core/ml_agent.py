"""
de Machine Learning: entrenar un Árbol de Decisión y usarlo para jugar.

En lugar de explorar miles de jugadas futuras (Minimax), el árbol mira el
tablero y responde con una cadena corta de preguntas como
"¿la casilla 4 está vacía?" hasta llegar a una jugada.

Contiene:
    - MLAgent:  elige jugadas con el árbol ya entrenado (inferencia)
    - train():  entrena el árbol y mide su precisión
    - main():   lee el CSV, entrena, guarda el modelo y muestra sus reglas
"""

import os
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from scripts.generar_dataset import DATASET_FILE, FEATURE_COLUMNS, LABEL_COLUMN, OPTIMAL_COLUMN
from core.game_model import GameModel, PLAYER_O, PLAYER_X

MODEL_FILE: str = "models/tree_model.joblib"
RULES_FILE: str = "reports/tree_rules.txt"


# ============================================================
# Agente (inferencia)
# ============================================================

class MLAgent:
    """Elige jugadas con un DecisionTreeClassifier ya entrenado.

    Tiene la misma interfaz que MinimaxAgent: `choose_move()` y `nodes_evaluated`.
    """

    def __init__(
        self,
        tree: DecisionTreeClassifier,
        accuracy: float | None = None,
        ai_player: str = PLAYER_O,
        human_player: str = PLAYER_X,
    ) -> None:
        """
        Args:
            tree: Árbol de decisión entrenado.
            accuracy: Precisión medida con posiciones que el árbol no vio (0 a 1).
        """
        self.tree = tree
        self.accuracy = accuracy
        self.ai_player = ai_player
        self.human_player = human_player
        # Nodos del ÁRBOL DE DECISIÓN recorridos en la última jugada
        # (compáralo con los miles de nodos que explora Minimax)
        self.nodes_evaluated: int = 0

    def choose_move(self, board: GameModel) -> int:
        """Devuelve el índice (0-8) de la jugada elegida para el tablero actual."""
        if board.current_player != self.ai_player:
            raise ValueError(f"No es el turno de la IA ({self.ai_player}).")
        if not board.get_available_moves():
            raise ValueError("No hay movimientos disponibles.")
        return self.predict(board.to_tuple(self.ai_player, self.human_player))

    def predict(self, features: tuple[int, ...]) -> int:
        """Predice una jugada a partir de las 9 casillas codificadas (1 / -1 / 0).

        El árbol NO sabe qué casillas están ocupadas, así que podría sugerir una
        ilegal. Por eso se elige la casilla VACÍA con mayor probabilidad.
        """
        x = np.array([features])
        probabilities = self.tree.predict_proba(x)[0]
        self.nodes_evaluated = int(self.tree.decision_path(x).indices.size)

        empty_cells = {i for i, value in enumerate(features) if value == 0}
        best_move, best_prob = -1, 0.0
        for move, prob in zip(self.tree.classes_, probabilities):
            if int(move) in empty_cells and prob > best_prob:
                best_move, best_prob = int(move), float(prob)

        if best_move == -1:  # El árbol solo sugería casillas ocupadas: elegimos una libre al azar
            return random.choice(sorted(empty_cells))
        return best_move

    @classmethod
    def load(cls, file: str = MODEL_FILE) -> "MLAgent":
        """Carga un modelo guardado. Lanza FileNotFoundError si todavía no se entrenó."""
        data = joblib.load(file)
        return cls(data["tree"], data["accuracy"])

    def save(self, file: str = MODEL_FILE) -> None:
        """Guarda el árbol y su precisión en un archivo."""
        os.makedirs(os.path.dirname(file), exist_ok=True)
        joblib.dump({"tree": self.tree, "accuracy": self.accuracy}, file)


# ============================================================
# Entrenamiento
# ============================================================

def _fit(data: pd.DataFrame) -> DecisionTreeClassifier:
    """Entrena un árbol: 9 casillas (entrada) -> jugada (salida)."""
    tree = DecisionTreeClassifier(random_state=42)
    # Ensena a hasociar el tablero de 9 c con la jugada optima
    tree.fit(data[FEATURE_COLUMNS].to_numpy(), data[LABEL_COLUMN].to_numpy())
    return tree     


def measure_accuracy(agent: MLAgent, test_df: pd.DataFrame) -> float:
    """% de jugadas del agente que son óptimas según Minimax.

    Si hay varias jugadas igual de buenas, cualquiera de ellas cuenta como acierto.
    """
    hits = 0
    for row in test_df.itertuples(index=False):
        features = tuple(int(getattr(row, col)) for col in FEATURE_COLUMNS)
        optimal_moves = {int(m) for m in str(getattr(row, OPTIMAL_COLUMN)).split("|")}
        hits += int(agent.predict(features) in optimal_moves)
    return hits / len(test_df)


def train(df: pd.DataFrame) -> MLAgent:
    """Entrena en dos pasos: primero MIDE la precisión, luego entrena el árbol definitivo.

    Paso 1 (medir): se entrena con el 80 % de las filas y se evalúa con el 20 % restante,
        posiciones que ese árbol nunca vio. Esa es la precisión honesta.
    Paso 2 (definitivo): se vuelve a entrenar con TODAS las filas para que el árbol
        conozca la mayor cantidad posible de posiciones.
    """
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    accuracy = measure_accuracy(MLAgent(_fit(train_df)), test_df)   # Paso 1
    final_tree = _fit(df)                                           # Paso 2
    return MLAgent(final_tree, accuracy)


def main() -> None:
    try:
        df = pd.read_csv(DATASET_FILE, dtype={OPTIMAL_COLUMN: str})
    except FileNotFoundError:
        print(f"No existe {DATASET_FILE}.")
        return

    print(f"Dataset: {len(df)} posiciones. Entrenando el arbol...")
    agent = train(df)
    agent.save()

    tree = agent.tree
    print(f"\nPrecisión en posiciones que no vio: {agent.accuracy:.1%}")
    print(f"Árbol: profundidad {tree.get_depth()}, {tree.tree_.node_count} nodos, "
          f"{tree.get_n_leaves()} hojas")
    print(f"Modelo guardado en: {MODEL_FILE}")

    # Reglas en texto (export_text): así se "ve" lo que aprendió el árbol
    header = (
        "Casillas c0..c8 (fila por fila).  1 = IA (O),  -1 = humano (X),  0 = vacía\n"
        "  'cK <= -0.5' -> la casilla K tiene X      'cK > 0.5' -> la casilla K tiene O\n"
        "  'class: N'   -> la jugada elegida es la casilla N\n\n"
    )
    rules = export_text(tree, feature_names=FEATURE_COLUMNS)
    os.makedirs(os.path.dirname(RULES_FILE), exist_ok=True)
    with open(RULES_FILE, "w", encoding="utf-8") as file:
        file.write(header + rules)
    print(f"Reglas completas en: {RULES_FILE}\n")
    print("Primeras reglas del árbol:")
    print(export_text(tree, feature_names=FEATURE_COLUMNS, max_depth=3))


if __name__ == "__main__":
    main()