#!/usr/bin/env python
"""
Thomas:

"""

########################################
# Imports
########################################

from math import factorial
from math import sqrt

import numpy


########################################
# Main
########################################

np_column = lambda py_list: numpy.array( [ [i] for i in py_list ] )

def main():

    bfin_fn = 'bfin_cards/extr-NNLO-CT14nnlo.bfin'

    y, sigma, E = Read_bfin_card( bfin_fn )

    print 'Found y = '
    print '  ' + ', '.join( map(str,y) )
    print 'Found sigma = '
    print '  ' + ', '.join( map(str,sigma) )
    print 'Found E = '
    for line in E:
        print '  ' + ', '.join(map(str,line))
    print '------------------------------------\n'



    # Convert to numpy arrays
    y = np_column(y)
    sigma = np_column(sigma)

    # Initial weight vector
    n_meas = len(y)
    alpha = np_column([ float(i)/n_meas for i in range(n_meas) ])

    
    






def fn_sigma_squared( alpha, E ):
    alpha_T = alpha.reshape( (1,len(alpha)) )
    return numpy.dot( alpha_T, numpy.dot( E, alpha ) )[0][0]



def Read_bfin_card( bfin_fn ):

    with open( bfin_fn, 'rb' ) as bfin_fp:
        lines = []
        for line in bfin_fp:
            line = line.strip()
            if len(line) == 0:
                continue
            elif line[0] == '#':
                continue
            else:
                lines.append( line )

    TITLE = lines.pop(0).split()[1]
    NOBS  = int(lines.pop(0).split()[1])
    NMEA  = int(lines.pop(0).split()[1])
    NERR  = int(lines.pop(0).split()[1])
    
    EXPS = lines.pop(0).split()[1:]
    OBSNAME = lines.pop(0)

    # Read the center values
    y = map( float, lines.pop(0).split()[1:] )

    # Read the sigmas
    sigma = {}
    for i in range(NERR):
        err_line = lines.pop(0).split()
        sigma[ err_line[0] ] = map( float, err_line[1:] )

    # Read the correlations
    ERRS = lines.pop(0).split()[2:]

    corr_dict = {}
    for exp1 in EXPS:
        corr_dict[exp1] = {}
        corr_dict[exp1][exp1] = [ 1.0 for i in range(NMEA-1) ]
        
    for i in range(factorial(NMEA-1)):
        line = lines.pop(0).split()
        exp1 = line[0]
        exp2 = line[1]
        corr_dict[exp1][exp2] = line[2:]
        corr_dict[exp2][exp1] = line[2:]

    # Make error table per source
    err_tables = []
    for i_err, err in enumerate(ERRS):
        table = []
        for i_exp1, exp1 in enumerate(EXPS):
            line = []
            for i_exp2, exp2 in enumerate(EXPS):
                sigma_exp1 = sigma[err][i_exp1]
                sigma_exp2 = sigma[err][i_exp2]
                rho_exp12 = float(corr_dict[exp1][exp2][i_err])
                line.append( rho_exp12 * sigma_exp1 * sigma_exp2 )
            table.append(line)
        err_tables.append(table)


    # ======================================
    # Add errors in quadrature
                
    # Add errors in quadrature for 1 E-matrix
    # This can be made more sophisticated (information is thrown away here)
    E = []
    for i_exp1 in range(NMEA):
        line = []
        for i_exp2 in range(NMEA):
            e = 0.0
            for i_err in range(NERR):
                e += err_tables[i_err][i_exp1][i_exp2]**2
            line.append( sqrt(e) )
        E.append(line)

    # Quad summed sigmas
    qsummed_sigmas = []
    for i_exp in range(NMEA):
        e = 0.0
        for err in ERRS:
            e += sigma[err][i_exp]**2
        qsummed_sigmas.append( sqrt(e) )

    return y, qsummed_sigmas, E


########################################
# End of Main
########################################
if __name__ == "__main__":
    main()
