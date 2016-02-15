#!/usr/bin/env python
"""
Thomas

Inspiration from:
http://docs.scipy.org/doc/scipy-0.16.0/reference/tutorial/optimize.html

"""

########################################
# Imports
########################################

from math import sqrt
from copy import deepcopy

import numpy

from scipy.optimize import minimize


########################################
# Main
########################################

np_column = lambda py_list: numpy.array( [ [i] for i in py_list ] )

def main():

    ########################################
    # Read bfin file and print found values
    ########################################

    bfin_fn = 'bfin_cards/extr-NNLO-CT14nnlo.bfin'
    #bfin_fn = 'bfin_cards/extr-NNLO-NNLL-NNPDF23-nnlo-FFN-NF5.bfin'

    y, sigma, E, exps = Read_bfin_card( bfin_fn )

    print '------------------------------------'
    print 'Using bfin card {0}'.format( bfin_fn )
    print 'Found y = '
    print '  ' + ', '.join( map(str,y) )
    print 'Found sigma = '
    print '  ' + ', '.join( map(str,sigma) )
    print 'Found E = '
    for line in E:
        print '  ' + ', '.join(map(str,line))


    ########################################
    # Minimization procedure
    ########################################

    # Convert to numpy arrays
    y = np_column(y)
    sigma = np_column(sigma)

    n_meas = len(y)

    # Initial weight vector - simply 1/n
    alpha_0 = [ float(i)/n_meas for i in range(n_meas) ]
    
    # Build the constraint
    cons = ({   'type': 'eq',
                'fun' : lambda x: numpy.array( [sum(x) - 1.0 ]  ),
                'jac' : lambda x: numpy.array( [1.0 for i in x] )
                })

    # Build the function so that it takes 1 np array
    func = lambda alpha: fn_sigma_squared( alpha, E )
    func_deriv = lambda alpha: der_fn_sigma_squared( alpha, E )


    print '\n------------------------------------'
    print 'Starting minimize'
    res = minimize(
        func,
        alpha_0,
        #args=(-1.0,),
        jac=func_deriv,
        constraints=cons,
        method='SLSQP',
        options={'disp': True}
        )

    print '\n------------------------------------'
    print res

    print '\n------------------------------------'
    
    blue_c = 0.0
    for i_exp, exp in enumerate(exps):
        blue_c += float(res.x[i_exp]) * y[i_exp]
    blue_e = sqrt(float(res.fun))
    blue_c = float( blue_c )

    for i_exp, exp in enumerate(exps):
        print '{0:10s}: {1:7.5f} +- {2:7.5f} ({3:7.5f} %)'.format(
            exp, float(y[i_exp]), float(sigma[i_exp]), 100*float(res.x[i_exp]) )
    print '{0:10s}: {1:7.5f} +- {2:7.5f}'.format( 'BLUE', blue_c, blue_e )







# ======================================
# Function to be minimized

def fn_sigma_squared( alpha, E ):
    alpha = numpy.array(alpha)
    alpha_T = alpha.reshape( (1,len(alpha)) )
    res = numpy.dot( alpha_T, numpy.dot( E, alpha ) )

    """
    print '-----------------------------'
    print 'Inside fn_sigma_squared():'
    print 'alpha ='
    print alpha
    print 'E = '
    print E
    print 'alpha^T * E * alpha = '
    print res
    """

    return float(numpy.dot( alpha_T, numpy.dot( E, alpha ) ))


def der_fn_sigma_squared( alpha, E ):
    ret = []
    da = 0.01
    for i_a in range(len(alpha)):

        alpha_plus_da = deepcopy(alpha)
        alpha_plus_da[i_a] += da

        ret.append( (
            fn_sigma_squared( alpha_plus_da, E ) - fn_sigma_squared( alpha, E )
            ) / da )

    return numpy.array(ret)


# ======================================
# Reads in a bfin card
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
        corr_dict[exp1][exp1] = [ 1.0 for i in range(NERR) ]
        
    for i in range( sum([ i for i in range(NMEA) ]) ):
        line = lines.pop(0).split()
        exp1 = line[0]
        exp2 = line[1]
        corr_dict[exp1][exp2] = line[2:]
        corr_dict[exp2][exp1] = line[2:]


    print EXPS
    print ERRS

    # Make error table per source
    err_tables = []
    for i_err, err in enumerate(ERRS):
        table = []
        for i_exp1, exp1 in enumerate(EXPS):
            line = []
            for i_exp2, exp2 in enumerate(EXPS):

                print 'Trying err = {0}, exp1 = {1}, exp2 = {2}'.format( err, exp1, exp2 )

                sigma_exp1 = sigma[err][i_exp1]
                sigma_exp2 = sigma[err][i_exp2]

                print corr_dict[exp1][exp2]

                rho_exp12 = float(corr_dict[exp1][exp2][i_err])
                line.append( rho_exp12 * sigma_exp1 * sigma_exp2 )
            table.append(line)
        err_tables.append(table)


    # ======================================
    # Add errors in quadrature

    # Quad summed sigmas
    qsummed_sigmas = []
    for i_exp in range(NMEA):
        e = 0.0
        for err in ERRS:
            e += sigma[err][i_exp]**2
        qsummed_sigmas.append( sqrt(e) )

                
    # Add errors in quadrature for 1 E-matrix
    # This can be made more sophisticated (information is thrown away here)
    E = []
    for i_exp1 in range(NMEA):
        line = []
        for i_exp2 in range(NMEA):

            if i_exp1 == i_exp2:
                line.append( qsummed_sigmas[i_exp1]**2 )
                continue

            e = 0.0
            for i_err in range(NERR):
                e += err_tables[i_err][i_exp1][i_exp2]**2
            line.append( sqrt(e) )

        E.append(line)

    return y, qsummed_sigmas, E, EXPS


########################################
# End of Main
########################################
if __name__ == "__main__":
    main()
