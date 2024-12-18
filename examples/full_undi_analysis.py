from undi.undi_analysis import execute_undi_analysis
from ase.io import read 

structure = read('Cu6H.cif')

# to dump into results.json: res = execute_undi_analysis(structure,max_hdim=1e3, dump=True)
# else
res = execute_undi_analysis(structure,max_hdim=1e4)

# to do the same but from cmd line: python -m undi Cu6H.cif --dump True
# this will use the __main__.py
