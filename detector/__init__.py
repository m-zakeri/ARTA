import inspect
import sys


# from .smells_detector import *


def run_detection(content):
    smells = {}
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) and obj.__name__[-5:] == 'Smell':
            detector = obj()
            detector.find(content, smells)
    return smells
