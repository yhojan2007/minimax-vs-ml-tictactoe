"""
Punto de entrada del proyecto.
"""

from interfaces.tkinter.view_tk import GUIView
from interfaces.tkinter.controller_tk import GameController


def main() -> None:
    view = GUIView()
    controller = GameController(view)
    controller.start()


if __name__ == "__main__":
    main()