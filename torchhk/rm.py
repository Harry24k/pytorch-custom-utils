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
        
    def plot(self, x_key, y_keys, title="", legend=True, loc='best') :
        if not isinstance(y_keys, list) :
            y_keys = [y_keys]
        
        if len(y_keys) > 2:
            raise ValueError('The maximum length of y_keys is two.')
        
        if self._mode > 0 and x_key == 'Epoch' :
            data = self.to_dataframe(['Epoch'] + y_keys).groupby('Epoch').tail(1)
        elif self._mode > 1 and x_key == 'Iter' :
#             print("Warnings : This graph is an estimated graph based on Epoch/Iter.") 
            data = self.to_dataframe(['Epoch', 'Iter'] + y_keys)
            data['Iter'] += (data['Epoch']-min(data['Epoch']))*max(data['Iter'])
        else : 
            data = self.to_dataframe([x_key] + y_keys)
            
        if len(y_keys) == 1 :
            plt.plot(data[x_key], data[y_keys[0]]) 
            plt.xlabel(x_key)
            plt.ylabel(y_keys[0]) 

            if legend :
                plt.legend(y_keys, loc=loc)
            
        elif len(y_keys) == 2 :
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()
            
            line1 = ax1.plot(data[x_key], data[y_keys[0]], 'r-')
            line2 = ax2.plot(data[x_key], data[y_keys[1]], 'b-')

            ax1.set_xlabel(x_key)
            ax1.set_ylabel(y_keys[0])
            ax2.set_ylabel(y_keys[1])
            
            if legend :
                lines = line1 + line2
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