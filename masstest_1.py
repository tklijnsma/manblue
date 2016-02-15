#!/usr/bin/env python
"""
Thomas:

"""

########################################
# Imports
########################################

import ROOT
from math import *
from array import array


########################################
# Main
########################################

ROOT.gROOT.SetBatch(True)
ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = kError;")
ROOT.gStyle.SetOptFit(1011)
ROOT.gStyle.SetOptStat(0)

c1 = ROOT.TCanvas("c1","c1",1200,800)

def main():


    f_Rayleigh = lambda x, s: (x/s**2) * exp( -x**2/(2*s**2) )
    f_Gauss = lambda x, mu, s: 1/(abs(s)*sqrt(2*pi)) * exp( -0.5 * ((x-mu)/s)**2 )

    mass_true = 20.0
    scale = 500
    b_fn = lambda x: scale*f_Rayleigh( x, 10.0 )
    s_fn = lambda x: scale*0.025*f_Gauss( x, mass_true, 1.5 )

    x_min = 0.0
    x_max = 50.0
    n_bins = 100

    x_axis = Make_center_axis( x_min, x_max, n_bins )


    b_hist = ROOT.TH1F( 'bhist', 'bhist', n_bins, x_min, x_max )
    s_hist = ROOT.TH1F( 'shist', 'shist', n_bins, x_min, x_max )
    sb_hist = ROOT.TH1F( 'sbhist', 'sbhist', n_bins, x_min, x_max )

    for i in range(n_bins):
        b_hist.SetBinContent( i+1, int( b_fn(x_axis[i])) )
        s_hist.SetBinContent( i+1, int( s_fn(x_axis[i])) )
        sb_hist.SetBinContent( i+1, int( s_fn(x_axis[i]) + b_fn(x_axis[i]) ) )

    b_hist.SetLineColor(1)
    s_hist.SetLineColor(2)
    sb_hist.SetLineColor(3)

    b_hist.Draw()
    s_hist.Draw('SAME')
    sb_hist.Draw('SAME')
    Print_c1( 'count_hist' )

        
    ########################################
    # Minimize likelihood functions
    ########################################

    b_list = [ int( b_fn(x_axis[i]) ) for i in range(n_bins) ]
    s_list = [ int( s_fn(x_axis[i]) ) for i in range(n_bins) ]
    sb_list = [ int(b_fn(x_axis[i]))+int(s_fn(x_axis[i])) for i in range(n_bins) ]

    def L( mu, theta ):
        val = 1.0
        for i in range(n_bins):
            s = s_list[i]
            b = b_list[i]
            n = sb_list[i]
            val *= (mu*s + theta*b)**n / factorial(n) * exp(-(mu*s+theta*b))
        return val


    # Hypothesis functions
    s_mass_fn = lambda x, mass: scale*0.025*f_Gauss( x, mass, 1.5 )
    sb_mass_fn = lambda x, mass: s_mass_fn( x, mass ) + b_fn(x)

    # Likelihood under hypothesis of some signal mass
    def L_mass( mu, theta, mass ):
        s_list = [ s_mass_fn( x_axis[i], mass ) for i in range(n_bins) ]
        #sb_list = [ sb_mass_fn( x_axis[i], mass ) for i in range(n_bins) ]
        val = 1.0
        for i in range(n_bins):
            s = s_list[i]
            b = b_list[i]
            n = sb_list[i]
            val *= (mu*s + theta*b)**n / factorial(n) * exp(-(mu*s+theta*b))
        print '    L_mass({0},{1},{2}) = {3}'.format( mu, theta, mass, val )
        return val


    # Loop over masses
    mass_list = Make_center_axis( 10.0, 40.0, 12 )
    for mass in mass_list:

        print '\n' + '-'*35
        print '    Likelihood at mass = ' + str(mass)

        # -----------------
        # Hat

        # Function to minimize
        #minL_hat = lambda mu, theta: -1.0*L(mu,theta)
        minL_hat =  lambda mu, theta: -1.0*L_mass( mu, theta, mass )

        min_result = Minimize_2D( minL_hat )

        mu_hat = min_result[0]
        theta_hat = min_result[1]
        
        print '\nmu_hat = ' + str(mu_hat)
        print 'theta_hat = ' + str(theta_hat)

        # -----------------
        # Hat-hat

        # Hypothesized mu
        mu = 1.0

        # Function to minimize
        #minL_hathat = lambda theta: -1.0*L(mu,theta)
        minL_hathat = lambda theta: -1.0*L_mass( mu, theta, mass )

        theta_hathat = Minimize_1D( minL_hathat )

        print '\nmu = ' + str(mu)
        print 'theta_hathat = ' + str(theta_hathat)

        





    """
    parabola_2D = lambda x,y: (x-3.2)*(x-3.2) + (y-4.1)*(y-4.1)
    min_result = Minimize_2D( parabola_2D )

    print '2D parabola:'
    print min_result[0]
    print min_result[1]

    print '\n1D parabola:'

    parabola_1D = lambda x: (x-3.6)*(x-3.6)
    min_result = Minimize_1D( parabola_1D )

    print min_result
    """


########################################
# Functions
########################################

# ======================================
# 2D minimizing

class PyMyFCN2( ROOT.TPyMultiGenFunction ):
    def __init__( self, py_function ):
        self.py_function = py_function
        ROOT.TPyMultiGenFunction.__init__( self, self )

    def NDim( self ):
        #print 'PYTHON NDim called: 2'
        return 2

    def DoEval( self, args ):
        # args is type <Double_t buffer, size 2147483647>
        x = args[0]
        y = args[1]
        #ret = f_parabola(x,y)
        ret = self.py_function(x,y)
        #print 'PYTHON MyFCN2::DoEval val=', ret
        return ret

def Minimize_2D( py_function ):
    
    minimizer = ROOT.Minuit2.Minuit2Minimizer()

    minimizer.SetMaxFunctionCalls(1000000)
    minimizer.SetMaxIterations(100000)
    minimizer.SetTolerance(0.001)

    step = array('d', (0.01,0.01) )
    variable = array('d', (0.5,0.5) )

    # Turn python function into root readable function
    f = PyMyFCN2( py_function )
    minimizer.SetFunction(f)

    # Set the free variables to be minimized
    minimizer.SetVariable( 0, "x", variable[0], step[0])
    minimizer.SetVariable( 1, "y", variable[1], step[1])

    minimizer.Minimize()
    min_result = minimizer.X()

    # Types have some issues
    result = [ float(min_result[0]), float(min_result[1]) ] 
    return result


# ======================================
# 1D minimizing

class PyMyFCN1( ROOT.TPyMultiGenFunction ):
    def __init__( self, py_function ):
        self.py_function = py_function
        ROOT.TPyMultiGenFunction.__init__( self, self )

    def NDim( self ):
        #print 'PYTHON NDim called: 1'
        return 1

    def DoEval( self, args ):
        # args is type <Double_t buffer, size 2147483647>
        x = args[0]
        #y = args[1]
        ret = self.py_function(x)
        #print 'PYTHON MyFCN2::DoEval val=', ret
        return ret

def Minimize_1D( py_function ):
    
    minimizer = ROOT.Minuit2.Minuit2Minimizer()

    minimizer.SetMaxFunctionCalls(1000000)
    minimizer.SetMaxIterations(100000)
    minimizer.SetTolerance(0.001)

    step = array('d', (0.01,0.01) )
    variable = array('d', (0.5,0.5) )

    # Turn python function into root readable function
    f = PyMyFCN1( py_function )
    minimizer.SetFunction(f)

    # Set the free variables to be minimized
    minimizer.SetVariable( 0, "x", variable[0], step[0])
    #minimizer.SetVariable( 1, "y", variable[1], step[1])

    minimizer.Minimize()
    min_result = minimizer.X()

    # Types have some issues
    result = float(min_result[0])
    return result




def Make_center_axis( x_min, x_max, N ):
    binw = (x_max-x_min)/N
    return [ x_min + 0.5*binw + i*binw for i in range(N) ]

    


def Print_c1( out_name = 'plot1', plotdir = '', png = False ):

    #c1.SetTicks( 1, 0 )
    #c1.RedrawAxis()
    c1.Print( plotdir + out_name + '.pdf' , 'pdf')

    if png:
        # Png
        img = ROOT.TImage.Create()
        img.FromPad( c1 )
        img.WriteImage( plotdir + out_name + '.png' )


########################################
# End of Main
########################################
if __name__ == "__main__":
    main()
