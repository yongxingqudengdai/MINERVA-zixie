# 04/14
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf

# 定义基本基线类baseline
class baseline(object):
    def get_baseline_value(self):
        pass
    def update(self, target):
        pass

#定义基于反应的基线类ReactiveBaseline
class ReactiveBaseline(baseline):
    def __init__(self, l): 
        # l用于控制基线更新的速度与平滑度，在0到1之间
        self.l = l
        # b用于存储当前基线值。trainable=False指定该参数不会在训练过程被更新
        self.b = tf.Variable( 0.0, trainable=False)
    def get_baseline_value(self):
        return self.b
    # ***基于目标值和上一次的基线值更新基线
    def update(self, target):
        self.b = tf.add((1-self.l)*self.b, self.l*target)
