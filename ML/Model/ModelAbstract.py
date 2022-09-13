
import logging
logger = logging.getLogger(__name__)
from abc import ABC, ABCMeta, abstractmethod


class ModelAbstract(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def preparedata(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def inference(self):
        pass

    def inference_data_prepare(self, data):
        result = data.split(',')
        result = [float(x) for x in result]
        return [result]