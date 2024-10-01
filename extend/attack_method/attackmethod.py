from abc import ABCMeta, abstractmethod


class AttackMethod(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, **kwargs):
        pass


    @abstractmethod
    def get_mask(self, ):
        pass


    @abstractmethod
    def get_texture(self, ):
        pass

    @abstractmethod
    def get_adv_image(self, ):
        pass


    @abstractmethod
    def optimizable_params(self, ):
        pass

