from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import itertools

class RecordManager(object) :
    
    def __init__(self, keys) :
        self.records = {}
        self._keys = keys
        
        # Mode 0 : No Epoch, No Iter
        # Mode 1 : Epoch Only, No Iter
        # Mode 2 : Epoch, Iter 
        
        if keys[0] != 'Epoch' :
            mode = 0
        else :
            if keys[1] != 'Iter' :
                mode = 1
            else : 
                mode = 2
                
        self._mode = mode
                        
        self._record_len = 0
        self._text_len = 0
        
        self._start_time = datetime.now()
        self._progress_time = datetime.now()
        
        for key in self._keys :
            self.records[key] = []
            
        self._spinner = itertools.cycle(['-', '/', '|', '\\'])
    
    def progress(self) :
        t = datetime.now() - self._progress_time
        print("Progress: "+ next(self._spinner) + " [" + str(t)+"/it]" + " "*20, end='\r')
        self._progress_time = datetime.now()
    
    def __repr__(self):
        return "RecordManager(keys=[%s])"%(", ".join(self._keys))
    
    def _head(self, values) :
              
        lengths = []
        slack = 3
        for i, value in enumerate(values) :
            length = max(len(str(value)), len(self._keys[i])) + slack
            if isinstance(value, float) :
                length = max(len("%.4f"%(value)), len(self._keys[i])) + slack
            lengths.append(length)
        
        self._form = "".join(['{:<'+str(length)+'.'+str(length)+'}' for length in lengths])
#         if self._keys[0] == "Epoch" :
#             self._mode = 1
#             self._form = ('{:<10.10}'*1 + '{:<15.15}'*(len(self._keys)-1))
#             if self._keys[1] == "Iter" :
#                 self._form = ('{:<10.10}'*2 + '{:<15.15}'*(len(self._keys)-2))
#                 self._mode = 2
                
        text = self._form.format(*self._keys)
        self._text_len = len(text)
        print("-"*self._text_len)
        print(text)
        print("="*self._text_len)
    
    def add(self, values) : 
        
        if len(values) != len(self._keys):
            raise ValueError('Values are NOT matched with Keys.')
            
        print(" "*50, end='\r')
        if self._record_len == 0 :
            self._head(values)
            
        self._record_len += 1
        text_arr = []
        
        for i, value in enumerate(values) :
            self.records[self._keys[i]].append(value)
            
            if isinstance(value, str) :
                text_arr.append("%s"%(value))
            elif isinstance(value, int) :
                text_arr.append("%d"%(value))
            elif isinstance(value, float) :
                text_arr.append("%.4f"%(value))
            else :
                text_arr.append(value)                
            
        print(self._form.format(*text_arr))
        print("-"*self._text_len)
       
    def summary(self) : 
        print("="*self._text_len)
        
        if self._mode > 0 :
            print("Total Epoch:", max(np.array(self.records["Epoch"])))
        else :
            print("Total Records:", self._record_len)
            
        print("Time Elapsed:", datetime.now() - self._start_time)
        
        if self._mode > 0 :
            print("Min(epoch)/Max(epoch): ")
        else :
            print("Min(th)/Max(th): ")
            
        for i, key in enumerate(self._keys) :
            history = np.array(self.records[key])
            
            if i < self._mode :
                continue
            
            if isinstance(self.records[key][0], (float, int)) :
                argmin = history.argmin()
                argmax = history.argmax()
                
                if self._mode > 0 :
                    pos_min = self.records["Epoch"][argmin]
                    pos_max = self.records["Epoch"][argmax]
                else :
                    pos_min = argmin+1
                    pos_max = argmax+1
                
                print("-"+key+": %.4f(%d)"%(history[argmin], pos_min)+
                      "/%.4f(%d)"%(history[argmax], pos_max))
                
        print("-"*self._text_len)
        
    def plot(self, x_key, y_keys, title="",
             xlabel="", ylabel="", ylabel_second="",
             xlim=None, ylim=None, ylim_second=None,
             colors = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
 (1.0, 0.4980392156862745, 0.054901960784313725),
 (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
 (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
 (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
 (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
 (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
 (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
 (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
 (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)],
             legend=True, loc='best') :
        
#         tableau20 = [[ 31, 119, 180],
#              [255, 127,  14],
#              [ 44, 160,  44],
#              [214,  39,  40],
#              [148, 103, 189],
#              [140,  86,  75],
#              [227, 119, 194],
#              [127, 127, 127],
#              [188, 189,  34],
#              [ 23, 190, 207]]

#         for i in range(len(tableau20)):  
#             r, g, b = tableau20[i]  
#             tableau20[i] = (r / 255., g / 255., b / 255.)  

        
        colors = itertools.cycle(colors)
        
        if not isinstance(y_keys, list) :
            y_keys = [y_keys]
                
        if self._mode > 0 and x_key == 'Epoch' :
            data = self.to_dataframe().groupby('Epoch').tail(1)
        elif self._mode > 1 and x_key == 'Iter' :
#             print("Warnings : This graph is an estimated graph based on Epoch/Iter.") 
            data = self.to_dataframe()
            data['Iter'] += (data['Epoch']-min(data['Epoch']))*max(data['Iter'])
        else : 
            data = self.to_dataframe()
            
        if len(y_keys) == 1 :
            if isinstance(y_keys[0], list) :
                raise ValueError("Please check 'y_keys' shape. List of lists is ONLY for two axises.")
            plt.plot(data[x_key], data[y_keys[0]], color=next(colors)) 
            plt.xlabel(xlabel)
            plt.ylabel(ylabel) 
            plt.xlim(xlim)
            plt.ylim(ylim)
            
            if legend :
                plt.legend(y_keys, loc=loc)
            
        elif len(y_keys) == 2 :
            if not isinstance(y_keys[0], list) :
                for y_key in y_keys :
                    plt.plot(data[x_key], data[y_key], color=next(colors)) 
                    
                plt.xlabel(xlabel)
                plt.ylabel(ylabel) 
                plt.xlim(xlim)
                plt.ylim(ylim)
                    
                if legend :
                    plt.legend(y_keys, loc=loc)
                
            else :
                if len(y_keys) > 2:
                    raise ValueError('The maximum length of y_axis is two.')
                    
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax2 = ax1.twinx()

                lines = None
                
                for y_key in y_keys[0] :
                    line = ax1.plot(data[x_key], data[y_key], color=next(colors))
                    if lines is None :
                        lines = line
                    else :
                        lines += line
                for y_key in y_keys[1] :
                    line = ax2.plot(data[x_key], data[y_key], color=next(colors))
                    if lines is None :
                        lines = line
                    else :
                        lines += line

                ax1.set_xlabel(xlabel)
                ax1.set_ylabel(ylabel)
                ax2.set_ylabel(ylabel_second)
                ax1.set_xlim(xlim)
                ax1.set_ylim(ylim)
                ax2.set_ylim(ylim_second)

                if legend :
                    labels = [line.get_label() for line in lines]
                    ax1.legend(lines, labels, loc=loc)

#         if self._mode > 0 and x_key == 'Epoch' :
#             plt.xticks(data[x_key])
#         if self._mode > 1 and x_key == 'Iter' :
#             plt.xticks(data[x_key])
                
        plt.title(title)
            
        plt.show()
        
    def to_dataframe(self, keys=None) :
        if keys == None :
            keys = self._keys
            
        data = pd.DataFrame(columns=[*keys])
        
        for key in keys :
            data[key] = np.array(self.records[key])
        
        return data
        
    def to_csv(self, path) :
        data = self.to_dataframe()            
        data.to_csv(path, mode="w", index=False)
        print("...Saved as csv to", path, "!")
        
    def save(self, path) :
        with open(path, "wb") as fp:   #Pickling
            pickle.dump(self.records, fp)
        print("...Saved as pickle to", path, "!")
        
    def load(self, path) :
        with open(path, "rb") as fp:   # Unpickling
            return pickle.load(fp)