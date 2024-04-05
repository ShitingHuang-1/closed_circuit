import numpy as np
import scipy as sp
import sympy
import math
import matplotlib.pyplot as plt
class Cirluation:
    def __init__(self,compliance):
        self.C=compliance
    def dp_c(self,qin,qout):
        delta_q=qin-qout
        c=self.C
        return delta_q/c
    def q_c(self,dpdt):
        return dpdt*self.compliance
    
        