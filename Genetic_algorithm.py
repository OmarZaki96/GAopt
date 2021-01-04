# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:20:51 2020

@author: OmarZaki
"""

import numpy as np
import sys
import time
from func_timeout import func_timeout, FunctionTimedOut
import matplotlib.pyplot as plt
import multiprocessing
import os

class GA():

    def __init__(self,function, dimension, variable_type='bool', variable_type_mixed=None,
                 variable_boundaries=None, function_timeout=5000, 
                 algorithm_parameters=None):
        #############################################################
        # input function
        assert (callable(function)),"function must be callable"     
        
        self.f = function
        #############################################################
        # dimension
        
        self.dim=int(dimension)
                
        #############################################################
        # input variable type
        
        assert(variable_type=='bool' or variable_type=='int' or\
               variable_type=='real'), \
               "\n variable_type must be 'bool', 'int', or 'real'"
        #############################################################
        # input variables' type (MIXED)     
        
        if variable_type_mixed is None:
             
            if variable_type=='real': 
                self.var_type=np.array([['real']]*self.dim)
            else:
                self.var_type=np.array([['int']]*self.dim)            
        
         
        else:
            assert (type(variable_type_mixed).__module__=='numpy'),\
            "\n variable_type must be numpy array"  
            assert (len(variable_type_mixed) == self.dim), \
            "\n variable_type must have a length equal dimension."       
        
            for i in variable_type_mixed:
                assert (i=='real' or i=='int'),\
                "\n variable_type_mixed is either 'int' or 'real' "+\
                "ex:['int','real','real']"+\
                "\n for 'boolean' use 'int' and specify boundary as [0,1]"
                        
        
            self.var_type=variable_type_mixed
        #############################################################
        # input variables' boundaries 
        
        assert(not isinstance(variable_boundaries,type(None))),"Please define variable boundaries"
        
        if variable_type!='bool' or type(variable_type_mixed).__module__=='numpy':
                       
            assert (type(variable_boundaries).__module__=='numpy'),\
            "\n variable_boundaries must be numpy array"
        
            assert (len(variable_boundaries)==self.dim),\
            "\n variable_boundaries must have a length equal dimension"        
        
        
            for i in variable_boundaries:
                assert (len(i) == 2), \
                "\n boundary for each variable must be a tuple of length two." 
                assert(i[0]<=i[1]),\
                "\n lower_boundaries must be smaller than upper_boundaries [lower,upper]"
            self.var_bound=variable_boundaries
        else:
            self.var_bound=np.array([[0,1]]*self.dim)
         
        ############################################################# 
        #Timeout
        self.funtimeout=float(function_timeout)
        
        ############################################################# 
        # input algorithm's parameters
        
        assert(algorithm_parameters!=None),"Please define algorithm parameters."
        
        param=algorithm_parameters
        
        self.pop_s=int(param['population_size'])
        
        assert (param['parents_portion']<=1\
                and param['parents_portion']>=0),\
        "parents_portion must be in range [0,1]" 
        self.par_portion = param['parents_portion']
        self.par_s=int(self.par_portion*self.pop_s)
        trl=self.pop_s - self.par_s
        if trl % 2 != 0:
            self.par_s+=1
               
        self.prob_mut=param['mutation_probability']
        
        assert (self.prob_mut<=1 and self.prob_mut>=0), \
        "mutation_probability must be in range [0,1]"
        
        
        self.prob_cross=param['crossover_probability']
        assert (self.prob_cross<=1 and self.prob_cross>=0), \
        "mutation_probability must be in range [0,1]"
        
        assert (param['elit_ratio']<=1 and param['elit_ratio']>=0),\
        "elit_ratio must be in range [0,1]"                
        self.elite_ratio = param['elit_ratio']
        trl=self.pop_s*self.elite_ratio
        
        if trl<1 and self.elite_ratio>0:
            self.num_elit=1
        else:
            self.num_elit=int(trl)
            
        assert(self.par_s>=self.num_elit), \
        "\n number of parents must be greater than number of elits"
        
        if param['max_num_iteration']==None:
            self.iterate=0
            for i in range (0,self.dim):
                if self.var_type[i]=='int':
                    self.iterate+=(self.var_bound[i][1]-self.var_bound[i][0])*self.dim*(100/self.pop_s)
                else:
                    self.iterate+=(self.var_bound[i][1]-self.var_bound[i][0])*50*(100/self.pop_s)
            self.iterate=int(self.iterate)
            if (self.iterate*self.pop_s)>10000000:
                self.iterate=10000000/self.pop_s
        else:
            self.iterate=int(param['max_num_iteration'])
        
        self.c_type=param['crossover_type']
        assert (self.c_type=='uniform' or self.c_type=='one_point' or\
                self.c_type=='two_point'),\
        "\n crossover_type must 'uniform', 'one_point', or 'two_point' Enter string" 
        
        
        self.stop_mniwi=False
        if param['max_iteration_without_improv']==None:
            self.mniwi=self.iterate+1
        else: 
            self.mniwi=int(param['max_iteration_without_improv'])

        #############################################################
        # Number of processes
        
        if param['Number_of_processes'] == 'max':
            self.N_processes = multiprocessing.cpu_count()
        else:
            self.N_processes = param['Number_of_processes']
        
        
        ############################################################# 
        # path of population file
        self.pop_file = param['Population_file_path']
        
        #############################################################
    def run(self):
    
        ############################################################# 
        # Initial Population
        self.integers=np.where(self.var_type=='int')
        self.reals=np.where(self.var_type=='real')
        
        
        self.pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
        self.solo=np.zeros(self.dim+1)
        self.var=np.zeros(self.dim)       
        
        for p in range(0,self.pop_s):
         
            for i in self.integers[0]:
                self.var[i]=np.random.randint(self.var_bound[i][0],\
                        self.var_bound[i][1]+1)  
                self.solo[i]=self.var[i].copy()
            for i in self.reals[0]:
                self.var[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0])    
                self.solo[i]=self.var[i].copy()
            self.pop[p] = self.solo.copy()
        
        total_runs = np.ones(len(self.pop))*len(self.pop)
        pool = multiprocessing.Pool(self.N_processes)
        result = pool.map(self.sim,zip(self.pop[:,:self.dim],total_runs,range(len(total_runs))))
        pool.close()
        pool.join()
        for p in range(0,self.pop_s):
            self.pop[p,self.dim] = result[p]
            obj = result[p]
        print("------------------------------------------")
        print("generation:",0,flush=True)
        print("Average fitness:", np.average(self.pop[:,self.dim]),flush=True)
        print("Variance:",np.var(self.pop[:,self.dim]),flush=True)
        print("Max fitness:",np.amin(self.pop[:,self.dim]),flush=True)
        print("best var:",self.pop[self.pop[:,self.dim].argsort()[0],:self.dim],flush=True)
        print("------------------------------------------")
    
        #############################################################
        
        #############################################################
        # Report
        self.report=[]
        test_obj=obj
        best_variable=self.var.copy()
        best_function=obj
        ##############################################################   
                        
        t=1
        counter=0
        while t<=self.iterate:
            
            
            self.progress(t,self.iterate,status="GA is running...")
            #############################################################
            #Sort
            self.pop = self.pop[self.pop[:,self.dim].argsort()]
        
                
            
            if self.pop[0,self.dim]<best_function:
                counter=0
                best_function=self.pop[0,self.dim].copy()
                best_variable=self.pop[0,: self.dim].copy()
            else:
                counter+=1
            #############################################################
            # Report
        
            self.report.append(float(self.pop[0,self.dim]))
            self.counter_num = counter
            self.iter_num = t
            
            self.Write2CSV(self.pop, self.pop_file, append=False)
            
            ##############################################################         
            # Normalizing objective function 
            
            normobj=np.zeros(self.pop_s)
            
            minobj=self.pop[0,self.dim]
            if minobj<0:
                normobj=self.pop[:,self.dim]+abs(minobj)
                
            else:
                normobj=self.pop[:,self.dim].copy()
        
            maxnorm=np.amax(normobj)
            normobj=maxnorm-normobj+1
        
            #############################################################        
            # Calculate probability
            
            sum_normobj=np.sum(normobj)
            prob=np.zeros(self.pop_s)
            prob=normobj/sum_normobj
            cumprob=np.cumsum(prob)
          
            #############################################################        
            # Select parents
            par=np.array([np.zeros(self.dim+1)]*self.par_s)
            
            for k in range(0,self.num_elit):
                par[k]=self.pop[k].copy()
            for k in range(self.num_elit,self.par_s):
                index=np.searchsorted(cumprob,np.random.random())
                par[k]=self.pop[index].copy()
                
            ef_par_list=np.array([False]*self.par_s)
            par_count=0
            while par_count==0:
                for k in range(0,self.par_s):
                    if np.random.random()<=self.prob_cross:
                        ef_par_list[k]=True
                        par_count+=1
                 
            ef_par=par[ef_par_list].copy()
        
            #############################################################  
            #New generation
            self.pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
            
            for k in range(0,self.par_s):
                self.pop[k]=par[k].copy()
            
            list_of_runs = []
            
            for k in range(self.par_s, self.pop_s, 2):
                r1=np.random.randint(0,par_count)
                r2=np.random.randint(0,par_count)
                pvar1=ef_par[r1,: self.dim].copy()
                pvar2=ef_par[r2,: self.dim].copy()
                
                ch=self.cross(pvar1,pvar2,self.c_type)
                ch1=ch[0].copy()
                ch2=ch[1].copy()
                
                ch1=self.mut(ch1)
                ch2=self.mutmidle(ch2,pvar1,pvar2)               
                
                self.pop[k,:self.dim]=ch1.copy()
                self.pop[k+1,:self.dim]=ch2.copy()   
                
                list_of_runs.append(k)
                list_of_runs.append(k+1)
            
            total_runs = np.ones(len(list_of_runs))*len(list_of_runs)
            p = multiprocessing.Pool(self.N_processes)
            result = p.map(self.sim,zip(self.pop[list_of_runs,:self.dim],total_runs,range(len(total_runs))))
            p.close()
            p.join()
                
            for i,result_i in zip(list_of_runs,result):
                self.pop[i,self.dim]= result_i
            
            print("------------------------------------------")
            print("generation:",t, flush=True)
            print("Average fitness:", np.average(self.pop[:,self.dim]),flush=True)
            print("Variance:",np.var(self.pop[:,self.dim]),flush=True)
            print("Max fitness:",np.amin(self.pop[:,self.dim]),flush=True)
            print("best var:",self.pop[self.pop[:,self.dim].argsort()[0],:self.dim],flush=True)
            print("------------------------------------------")
        #############################################################       
            t+=1
            if counter > self.mniwi:
                self.pop = self.pop[self.pop[:,self.dim].argsort()]
                if self.pop[0,self.dim]>=best_function:
                    t=iterate
                    self.progress(t,iterate,status="GA is running...")
                    # time.sleep(2)
                    t+=1
                    self.stop_mniwi=True
                
        #############################################################
        #Sort
        self.pop = self.pop[self.pop[:,self.dim].argsort()]
        
        if self.pop[0,self.dim]<best_function:
                
            best_function=self.pop[0,self.dim].copy()
            best_variable=self.pop[0,: self.dim].copy()
            #############################################################
            # Report
    
            self.report.append(float(self.pop[0,self.dim]))
     
            
            output_dict={'variable': best_variable, 'function':\
                              best_function}
            show=' '*100
            sys.stdout.write('\r%s' % (show))
            sys.stdout.write('\r The best solution found:\n %s' % (best_variable))
            sys.stdout.write('\n\n Objective function:\n %s\n' % (best_function))
            sys.stdout.flush() 
            re=np.array(report)
            plt.plot(re)
            plt.xlabel('Iteration')
            plt.ylabel('Objective function')
            plt.title('Genetic Algorithm')
            plt.show()
            if self.stop_mniwi==True:
                sys.stdout.write('\nWarning: GA is terminated due to the'+\
                                 ' maximum number of iterations without improvement was met!')
        ##############################################################################         

    def resume(self,Pop_file_path):
        # make sure the file exist
        pop = []
        assert(os.path.exists(Pop_file_path)),"Population file doesn't exist"
        with open(Pop_file_path,'r') as f:
            lines = f.readlines()
        for i,line in enumerate(lines):
            if i == 12:
                current_generation = line
                current_generation = current_generation.replace("Current generation number: ",'')
                current_generation = current_generation.replace("\n",'')
                current_generation = int(current_generation)
                
            elif i == 16:
                counter_value = line
                counter_value = counter_value.replace("Counter: ",'')
                counter_value = counter_value.replace("\n",'')
                counter_value = int(counter_value)
            
            elif i == 17:
                report = line
                report = report.replace("Report values: ",'')
                report = report.replace("\n",'')
                report = report.split(",")
                report = [float(i) for i in report]
                
            elif i > 17:
                pop_element = line[:-2]
                if not line == "\n":
                    pop_element = pop_element.split(",")
                    pop_element = [float(i) for i in pop_element]
                    pop.append(pop_element)
        pop = np.array(pop)
        ############################################################# 
        # Initial Population
        self.integers=np.where(self.var_type=='int')
        self.reals=np.where(self.var_type=='real')
        
        self.pop=pop
        self.pop = self.pop[self.pop[:,self.dim].argsort()]
        best_function=self.pop[0,self.dim].copy()
        best_variable=self.pop[0,: self.dim].copy()
        
        assert(len(self.pop) == self.pop_s), "Selected population size in algorithm parameters is different from population size in the file"
        #############################################################
        # Report
        self.report=report
        ##############################################################   
                        
        t=current_generation
        counter=counter_value
        while t<=self.iterate:
            
            
            self.progress(t,self.iterate,status="GA is running...")
            #############################################################
            #Sort
            self.pop = self.pop[self.pop[:,self.dim].argsort()]
            
            if self.pop[0,self.dim]<best_function:
                counter=0
                best_function=self.pop[0,self.dim].copy()
                best_variable=self.pop[0,: self.dim].copy()
            else:
                counter+=1
            #############################################################
            # Report
        
            self.report.append(float(self.pop[0,self.dim]))
            self.counter_num = counter
            self.iter_num = t
            
            self.Write2CSV(self.pop, self.pop_file, append=False)
            ##############################################################         
            # Normalizing objective function 
            
            normobj=np.zeros(self.pop_s)
            
            minobj=self.pop[0,self.dim]
            if minobj<0:
                normobj=self.pop[:,self.dim]+abs(minobj)
                
            else:
                normobj=self.pop[:,self.dim].copy()
        
            maxnorm=np.amax(normobj)
            normobj=maxnorm-normobj+1
        
            #############################################################        
            # Calculate probability
            
            sum_normobj=np.sum(normobj)
            prob=np.zeros(self.pop_s)
            prob=normobj/sum_normobj
            cumprob=np.cumsum(prob)
          
            #############################################################        
            # Select parents
            par=np.array([np.zeros(self.dim+1)]*self.par_s)
            
            for k in range(0,self.num_elit):
                par[k]=self.pop[k].copy()
            for k in range(self.num_elit,self.par_s):
                index=np.searchsorted(cumprob,np.random.random())
                par[k]=self.pop[index].copy()
                
            ef_par_list=np.array([False]*self.par_s)
            par_count=0
            while par_count==0:
                for k in range(0,self.par_s):
                    if np.random.random()<=self.prob_cross:
                        ef_par_list[k]=True
                        par_count+=1
                 
            ef_par=par[ef_par_list].copy()
        
            #############################################################  
            #New generation
            self.pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
            
            for k in range(0,self.par_s):
                self.pop[k]=par[k].copy()
            
            list_of_runs = []
            
            for k in range(self.par_s, self.pop_s, 2):
                r1=np.random.randint(0,par_count)
                r2=np.random.randint(0,par_count)
                pvar1=ef_par[r1,: self.dim].copy()
                pvar2=ef_par[r2,: self.dim].copy()
                
                ch=self.cross(pvar1,pvar2,self.c_type)
                ch1=ch[0].copy()
                ch2=ch[1].copy()
                
                ch1=self.mut(ch1)
                ch2=self.mutmidle(ch2,pvar1,pvar2)               
                
                self.pop[k,:self.dim]=ch1.copy()
                self.pop[k+1,:self.dim]=ch2.copy()   
                
                list_of_runs.append(k)
                list_of_runs.append(k+1)
            
            total_runs = np.ones(len(list_of_runs))*len(list_of_runs)
            p = multiprocessing.Pool(self.N_processes)
            result = p.map(self.sim,zip(self.pop[list_of_runs,:self.dim],total_runs,range(len(total_runs))))
            p.close()
            p.join()
                
            for i,result_i in zip(list_of_runs,result):
                self.pop[i,self.dim]= result_i
            
            print("------------------------------------------")
            print("generation:",t, flush=True)
            print("Average fitness:", np.average(self.pop[:,self.dim]),flush=True)
            print("Variance:",np.var(self.pop[:,self.dim]),flush=True)
            print("Max fitness:",np.amin(self.pop[:,self.dim]),flush=True)
            print("best var:",self.pop[self.pop[:,self.dim].argsort()[0],:self.dim],flush=True)
            print("------------------------------------------")
        #############################################################       
            t+=1
            if counter > self.mniwi:
                self.pop = self.pop[self.pop[:,self.dim].argsort()]
                if self.pop[0,self.dim]>=best_function:
                    t=iterate
                    self.progress(t,iterate,status="GA is running...")
                    # time.sleep(2)
                    t+=1
                    self.stop_mniwi=True
                
        #############################################################
        #Sort
        self.pop = self.pop[self.pop[:,self.dim].argsort()]
        
        if self.pop[0,self.dim]<best_function:
                
            best_function=self.pop[0,self.dim].copy()
            best_variable=self.pop[0,: self.dim].copy()
            #############################################################
            # Report
    
            self.report.append(float(self.pop[0,self.dim]))
     
            
            output_dict={'variable': best_variable, 'function':\
                              best_function}
            show=' '*100
            sys.stdout.write('\r%s' % (show))
            sys.stdout.write('\r The best solution found:\n %s' % (best_variable))
            sys.stdout.write('\n\n Objective function:\n %s\n' % (best_function))
            sys.stdout.flush() 
            re=np.array(report)
            plt.plot(re)
            plt.xlabel('Iteration')
            plt.ylabel('Objective function')
            plt.title('Genetic Algorithm')
            plt.show()
            if self.stop_mniwi==True:
                sys.stdout.write('\nWarning: GA is terminated due to the'+\
                                 ' maximum number of iterations without improvement was met!')
        ##############################################################################         
    ##############################################################################         
    def cross(self,x,y,c_type):
         
        ofs1=x.copy()
        ofs2=y.copy()
        
    
        if c_type=='one_point':
            ran=np.random.randint(0,self.dim)
            for i in range(0,ran):
                ofs1[i]=y[i].copy()
                ofs2[i]=x[i].copy()
      
        if c_type=='two_point':
                
            ran1=np.random.randint(0,self.dim)
            ran2=np.random.randint(ran1,self.dim)
                
            for i in range(ran1,ran2):
                ofs1[i]=y[i].copy()
                ofs2[i]=x[i].copy()
            
        if c_type=='uniform':
                
            for i in range(0, self.dim):
                ran=np.random.random()
                if ran <0.5:
                    ofs1[i]=y[i].copy()
                    ofs2[i]=x[i].copy() 
                   
        return np.array([ofs1,ofs2])
    ###############################################################################  
    
    def mut(self,x):
        
        for i in self.integers[0]:
            ran=np.random.random()
            if ran < self.prob_mut:
                
                x[i]=np.random.randint(self.var_bound[i][0],\
                 self.var_bound[i][1]+1) 
                    
        
    
        for i in self.reals[0]:                
            ran=np.random.random()
            if ran < self.prob_mut:   
    
               x[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0])    
            
        return x
    ###############################################################################
    def mutmidle(self,x, p1, p2):
        for i in self.integers[0]:
            ran=np.random.random()
            if ran < self.prob_mut:
                if p1[i]<p2[i]:
                    x[i]=np.random.randint(p1[i],p2[i])
                elif p1[i]>p2[i]:
                    x[i]=np.random.randint(p2[i],p1[i])
                else:
                    x[i]=np.random.randint(self.var_bound[i][0],\
                 self.var_bound[i][1]+1)
                        
        for i in self.reals[0]:                
            ran=np.random.random()
            if ran < self.prob_mut:   
                if p1[i]<p2[i]:
                    x[i]=p1[i]+np.random.random()*(p2[i]-p1[i])  
                elif p1[i]>p2[i]:
                    x[i]=p2[i]+np.random.random()*(p1[i]-p2[i])
                else:
                    x[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0]) 
        return x
    ###############################################################################     
    def evaluate(self):
        return self.f(self.temp)
    ###############################################################################    
    def sim(self,inputs):
        X = inputs[0]
        total = int(inputs[1])
        current = inputs[2]+1
        # print("running process",current,"of",total,flush=True)
        self.temp=X.copy()
        obj=None
        try:
            obj=func_timeout(self.funtimeout,self.evaluate)
        except FunctionTimedOut:
            print("given function is not applicable")
        assert (obj!=None), "After "+str(self.funtimeout)+" seconds delay "+\
                "func_timeout: the given function does not provide any output"
        return obj
    
    ###############################################################################
    def progress(self,count, total, status=''):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))
    
        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '_' * (bar_len - filled_len)
        sys.stdout.write('\r%s %s%s %s \n' % (bar, percents, '%', status))
        sys.stdout.flush()     
        
    ###############################################################################
    def Write2CSV(self,Values, file, append=False):
        string = ""
        string += "Population Size: " + str(self.pop_s) + "\n"
        string += "Parents Size: " + str(self.par_s) + "\n"
        string += "Parents portion: " + str(self.par_portion) + "\n"
        string += "Number of variables: " + str(self.dim) + "\n"
        var_type = [i[0] for i in self.var_type]
        string += "Variable types: " + ','.join(var_type) + "\n"
        var_bound = [str(i[0])+":"+str(i[1]) for i in self.var_bound]
        string += "Variable boundaries: " + ','.join(var_bound) + "\n"
        string += "Function Timeout: " + str(self.funtimeout) + "\n"
        string += "Mutation Probability: " + str(self.prob_mut) + "\n"
        string += "Crossover Probability: " + str(self.prob_cross) + "\n"
        string += "Number of elite elements: " + str(self.num_elit) + "\n"
        string += "Elite ratio: " + str(self.elite_ratio) + "\n"
        string += "Total number of generations: " + str(self.iterate) + "\n"
        string += "Current generation number: " + str(self.iter_num) + "\n"
        string += "Crossover type: " + str(self.c_type) + "\n"
        string += "Maximum number of iterations without improvement: " + str(self.mniwi) + "\n"
        string += "Number of processes: " + str(self.N_processes) + "\n"
        string += "Counter: " + str(self.counter_num) + "\n"
        report = [str(i) for i in self.report]
        string += "Report values: " + ','.join(report) + "\n"
        for row in Values:
            row = [str(i) for i in row]
            string += ','.join(row) + "\n"
        if type(file)!=type('some string'):
            #A file object was passed in, use it
            fP=file
            firstRow=False
        else:
            if os.path.exists(file):
                firstRow=False
            else:
                firstRow=True
            if append==True:
                fP=open(file,'a')
            else:
                fP=open(file,'w')
                
        if append==True and firstRow==False:
            fP.write(string)
        else:           
            fP.write(string)
        fP.close()
