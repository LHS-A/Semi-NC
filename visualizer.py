# -- coding: utf-8 --
from visdom import Visdom
import numpy as np

class Visualizer():
    def __init__(self, env="default", **kwargs): 
        self.vis = Visdom(env=env, **kwargs)  
        self.index = {}  


    def plot(self, win, y, con_point,x=None, **kwargs):
        if x is not None:
            x = x
        else:
            x = self.index.get(win, con_point) 
        self.vis.line(Y=np.array([y]), X=np.array([x]), win=str(win), update=None if x == 0 else "append", **kwargs)
        self.index[win] = x + 1  


    def plot_line(self, win, y, **kwargs):
        self.vis.line(win=win, X=np.linspace(1, len(y), len(y)), Y=y, **kwargs)

    def img(self, name, img_, **kwargs):
        self.vis.images(img_, win=str(name),  
                        opts=dict(title=name),
                        **kwargs)
        
    def plot_pred_contrast(self,pred,label,image):
        self.img(name="image", img_= image)
        self.img(name="pred", img_= pred)
        self.img(name="label", img_= label)
    
    def plot_entropy(self,H):
        self.img(name="pred_entropy", img_= H)

    def plot_metrics_total(self, metrics_dict):

        for metric, values in metrics_dict.items():
            if values['total']: 
                latest_value = values['total'][-1]  
                self.plot(win=metric, y=latest_value, opts=dict(title=metric, xlabel="Epoch", ylabel=metric), con_point=len(values['total']))
            else:
                # print(f"No data available for metric: {metric}")
                pass

    def plot_metrics_single(self, metrics_dict):

        for metric, values in metrics_dict.items():
            for task, task_values in values["total"][-1].items():
                self.plot(win=f"{metric}_{task}", y=task_values, opts=dict(title=f"{metric} - {task}", xlabel="Epoch", ylabel=metric), con_point=len(values["total"]))

    def plot_metrics_Test(self, metrics_dict, vis_name, number):
        for metric, values in metrics_dict.items():
            self.plot(win=metric, y=values['total'][-1], opts=dict(title=metric, xlabel="Epoch", ylabel=metric),con_point = number,name = vis_name)

   
        