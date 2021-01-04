# GAopt
A python package for genetic algorithm with parallel processing implemented

The package has the ability for parallel processing and resuming.


# example of implementation
```python
import numpy as np
from GAopt import GA
def objective(X):
    X1 = X[0]
    X2 = X[1]
    X3 = X[2]
    X4 = X[3]
    return (X1+X2)/(X3+X4+0.5)

varbound=np.array([[1,3],[1,4],[0.5,1.5],[2,20],])
vartype=np.array([['int'],['int'],['real'],['int'],])

parameters = {'max_num_iteration': None,
              'population_size': 400,
              'mutation_probability':0.1,
              'elit_ratio': 0.1,
              'crossover_probability': 0.5,
              'parents_portion': 0.3,
              'crossover_type':'uniform',
              'max_iteration_without_improv':None,
              'Number_of_processes':'max',
              'Population_file_path': "pop.csv"}

Genetic = GA(objective,4,
             variable_type_mixed=vartype, 
             variable_boundaries=varbound, 
             function_timeout=5000, 
             algorithm_parameters=parameters)

# this line is to run the code for the first time
Genetic.run()

# this line is to resume an already existing run
Genetic.resume("old_pop.csv")
```
