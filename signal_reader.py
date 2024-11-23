from abc import ABC, abstractmethod

import numpy as np  # wersja numpy 1.24.1


class ISignalReader(ABC):
    '''
    Funkcja zwraca wczytany sygnal ekg.
    :return: ndarray postaci: n na c, gdzie n to liczba probek, m to ilosc kanalow
    '''
    def read_signal(self) -> np.ndarray:
        raise NotImplementedError

    '''
    Funkcja zwraca czestotliwosc probkowania czytanego sygnalu
    :return: czestotliwosc probkowania
    '''
    def read_fs(self) -> float:
        raise NotImplementedError

    '''
    Funkcja zwraca specjalny kod pod nazwa ktore nalezy zapisac wynik ewaluacji w klasie RecordEvaluator
    :return: kod identyfikujacy ewaluacje/nazwa pliku do zapisu ewaluacji
    '''
    def get_code(self) -> str:
        raise NotImplementedError