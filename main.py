from matplotlib import pyplot as plt
import pdb

from src.fts import measles ## You need to (e.g. pip) install the package
                        ## for this to work.
                        

def main():
    df = measles(run_years=7,
                 burn_in_years=2000,
                 pna=0, ona=0)[['C1', 'C2', 'F1']]
    df = (df - df.mean()) /df.std()
    deltas = df.index.diff()
    assert deltas.nunique() == 1, deltas
    df.plot(title='C1 and C2 are cumulative weekly new cases (incidence)')
    plt.savefig('res.png')
    
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback, pdb, sys
        _,_, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
