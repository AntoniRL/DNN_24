from abc import ABC, abstractmethod
import wfdb

import numpy as np  # wersja numpy 1.24.1


class SignalReader(ABC):
    def __init__(self, record_path) -> None:
        self.record_path = record_path
        self.record = wfdb.rdrecord(self.record_path)
        self.record_name = self.record.record_name
        self.annotations = wfdb.rdann(self.record_path, 'atr')
    
    def read_signal(self) -> np.ndarray:
        '''
        Funkcja zwraca wczytany sygnal ekg.
        :return: ndarray postaci: n na c, gdzie n to liczba probek, m to ilosc kanalow
        ''' 
        return self.record.p_signal

    def read_fs(self) -> float:
        '''
        Funkcja zwraca czestotliwosc probkowania czytanego sygnalu
        :return: czestotliwosc probkowania
        '''
        return self.record.fs

    
    def get_code(self) -> str:
        '''
        Funkcja zwraca specjalny kod pod nazwa ktore nalezy zapisac wynik ewaluacji w klasie RecordEvaluator
        :return: kod identyfikujacy ewaluacje/nazwa pliku do zapisu ewaluacji
        '''
        return "".join([self.record_name, "_results"])
    



    # -----------MY METODS----------------

    def read_afib_ref(self) -> np.ndarray:
        '''
        Funkcja buduje i zwraca tablicę z informacją czy dana próbka należy do przedziału oznaczonego jako migotanie.
        :return: ndarray długości n, gdzie n to liczba probek. 0 - brak migotania; 1 - migotanie; non - brak danych;
        '''
        sample = self.annotations.sample
        aux_note = self.annotations.aux_note
        afib_ref = np.zeros(self.record.sig_len, dtype=np.uint8)
        # afib_ref[:sample[0]] = None
        for i in range(len(sample)):
            if i == len(sample)-1 and aux_note[i] == "(AFIB":
                afib_ref[sample[i]:] = 1.0
            elif aux_note[i] == "(AFIB":
                afib_ref[sample[i]:sample[i+1]] = 1.0
        return afib_ref