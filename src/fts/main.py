import pandas as pd
from matplotlib import pyplot as plt

from .helpers import make_simulation_params, random_IC
from .multistrain import run

def main():

    ## If you want to use Cobey's parameters, let what='cobey':
    params = make_simulation_params(what='cobey')
    #print("Cobey's parameters:")
    #print(pd.DataFrame(params).head())
    
    ## If you want to use parameters from our paper, let what='measles':
    params = make_simulation_params(what='measles', pna=0, ona=0, run_years=2, burn_in_years=20)
    #print("Measles' parameters:")    
    #print(pd.DataFrame(params).head())
   
    ## Run simulation (index 3 is arbitrary)
    df = run(**params[3])
    df[['C1', 'C2']].plot()
    plt.show()
    
    ## Check reproducibility
    S_init, I_init = random_IC()
    df1 = run(**params[3], seed=3, S_init=S_init, I_init=I_init)
    df2 = run(**params[3], seed=3, S_init=S_init, I_init=I_init)
    assert df1.equals(df2)

    
if __name__ == '__main__':
    try:
        main()
    except:
        import traceback, sys, pdb
        traceback.print_exc()
        tb = sys.exc_info()[2]        
        pdb.post_mortem(tb)
