"""Interfaz experimental para emitir disparos causales a partir de bloques."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from operator import index

import numpy as np


@dataclass(frozen=True)
class Trigger:
    """Disparo en el índice de la última muestra disponible, con el valor del indicador."""

    available_sample: int
    score: float


class CausalWindowDetector:
    """Evalúa una ventana retrospectiva en cada muestra nueva de un canal.

    score debe ser una función pura de la ventana real, en orden cronológico.
    threshold se expresa en las unidades que devuelva score. Un disparo corresponde
    al primer valor mayor o igual al umbral al pasar desde abajo. En el arranque y
    después de un hueco se espera a reunir una ventana completa.
    """

    def __init__(
        self,
        window_samples: int,
        threshold: float,
        score: Callable[[np.ndarray], float],
    ):
        """Configura la ventana causal y el umbral en unidades del indicador."""
        self.window_samples = index(window_samples)
        self.threshold = float(threshold)
        if self.window_samples < 1 or not math.isfinite(self.threshold) or not callable(score):
            raise ValueError(
                "Se requiere ventana positiva, umbral finito y una función de puntuación."
            )
        self.score = score
        self._tail = np.empty(0)
        self._next_sample = None
        self._above = False

    def process(self, samples, *, start_sample: int) -> list[Trigger]:
        """Consume un bloque real finito con índice absoluto de inicio.

        Un hueco reinicia el historial y exige una ventana completa. Un bloque
        solapado o fuera de orden se rechaza. Cada disparo queda fechado en
        la última muestra usada. El estado se actualiza tras procesar el bloque.
        """
        start = index(start_sample)
        if start < 0 or (self._next_sample is not None and start < self._next_sample):
            raise ValueError("Índice de inicio negativo o bloque solapado/fuera de orden.")
        block = np.asarray(samples)
        if (
            block.ndim != 1
            or not block.size
            or block.dtype.kind not in "iuf"
            or np.ma.is_masked(samples)
        ):
            raise ValueError("Se requiere un bloque unidimensional real sin muestras enmascaradas.")
        block = block.astype(float, copy=False)
        if not np.isfinite(block).all():
            raise ValueError("El bloque contiene muestras no finitas.")

        contiguous = start == self._next_sample
        tail = self._tail if contiguous else self._tail[:0]
        above = self._above if contiguous else False
        combined = np.concatenate((tail, block))
        triggers = []
        width = self.window_samples
        if combined.size >= width:
            windows = np.lib.stride_tricks.sliding_window_view(combined, width)
            # Cada ventana termina en una muestra que pertenece al bloque nuevo.
            first_available = start - tail.size + width - 1
            for available, window in enumerate(windows, first_available):
                value = float(self.score(window))
                if not math.isfinite(value):
                    raise ValueError("La función de puntuación devolvió un valor no finito.")
                current = value >= self.threshold
                if current and not above:
                    triggers.append(Trigger(available, value))
                above = current

        # Confirmar el avance solo tras validar y procesar todas las ventanas del bloque.
        keep = min(width - 1, combined.size)
        self._tail = combined[combined.size - keep :].copy()
        self._above = above
        self._next_sample = start + block.size
        return triggers
