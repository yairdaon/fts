from matplotlib import pyplot as plt

from fts import measles ## You need to (e.g. pip) install the package
                        ## for this to work.
                        

def main():
    df = measles(run_years=20)
    df.plot(title='C1 and C2 are cumulative weekly new cases (incidence)')
    plt.show()
    
main()
