"""Controles del piloto con referencia conocida y ruido reproducible."""

import numpy as np


def tone_packet(samples, fs, start, width, frequency):
    """Genera un coseno modulado con envolvente seno al cuadrado y soporte en muestras."""
    offsets = np.arange(width) / fs
    envelope = np.sin(np.pi * np.arange(width) / width) ** 2
    carrier = np.cos(2 * np.pi * frequency * offsets)
    packet = np.zeros(samples)
    packet[start : start + width] = envelope * carrier
    return packet, envelope


def inject_at_snr(noise, packet, start, width, snr_db):
    """Ajusta la amplitud usando los RMS dentro del soporte nominal de la inyección."""
    support = slice(start, start + width)
    scale = 10 ** (snr_db / 20) * np.linalg.norm(noise[support])
    scale /= np.linalg.norm(packet[support])
    return noise + scale * packet
