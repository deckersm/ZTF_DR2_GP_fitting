from scipy.interpolate import interp1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, RBF, ConstantKernel
from scipy.interpolate import UnivariateSpline
import scipy.integrate as integrate

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

import glob
import scipy
import os
from scipy.interpolate import UnivariateSpline
import scipy.integrate as integrate
from matplotlib import rcParams
from math import log
import extinction

from scipy.optimize import minimize

#from snpy import kcorr, fset
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=73, Om0=0.3)

import warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning)


yeff_ztfg = 4753.2  # phot wl from http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=Palomar/ZTF.g&&mode=browse&gname=Palomar&gname2=ZTF#filter
yeff_ztfr = 6370.98 # phot wl from http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=Palomar/ZTF.g&&mode=browse&gname=Palomar&gname2=ZTF#filter
yeff_ztfi = 7912.99 # phot wl from http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=Palomar/ZTF.g&&mode=browse&gname=Palomar&gname2=ZTF#filter

salt_dir="/Users/gdimit/python_workplace/ztf_dr2_project/kcorrections/salt_2_4"
wave_array_for_kcor=np.linspace(3000, 9200, 6201)

def collect_lc(sn_name):
    file_sn_lc='/home/users/deckersm/second_max/lightcurves/'+sn_name+'_LC.csv'
    lc_all=pd.read_csv(file_sn_lc,comment='#',delim_whitespace=True)
    lc_clean = lc_all[(lc_all.flag&1==0) & (lc_all.flag&2==0) & (lc_all.flag&4==0) & (lc_all.flag&8==0) & (lc_all.flag&16==0)]
    lc_clean2 = lc_clean[ (lc_clean.x_pos >= 0) & (lc_clean.y_pos >= 0)]
    new_df = lc_clean2[ ['mjd', 'filter','flux', 'flux_err','ZP'] ].reset_index(drop=True)
#     print('collect gives ')
#     print(new_df)
    return new_df

def stack_lc(df_lc,t0,redshift,days_start=-30.0,days_end=70.0):
    
    new_df = pd.DataFrame(columns = ['mjd','filter','filter_wl', 'flux', 'flux_err','ZP'])

    df_lc_days=df_lc[ (df_lc['mjd']>t0+days_start) & (df_lc['mjd']<t0+days_end)]
    min_mjd=df_lc_days['mjd'].min()
    max_mjd=df_lc_days['mjd'].max()
    
    unique_filters = np.unique(df_lc_days['filter'].values)
    
    mjd_av=[]
    filter_av=[]
    filter_wl_av=[]
    flux_av=[]
    flux_err_av=[]
    ZP_av=[]
        
    # applying filter specific error floors
    for unique_filter in unique_filters:
        if unique_filter=='ztfg':
            error_floor = 0.025
        elif unique_filter=='ztfr':
            error_floor = 0.035
        elif unique_filter=='ztfi':
            error_floor = 0.06
        df_lc_days.loc[(df_lc_days['filter']==unique_filter) & (df_lc_days['flux_err']< error_floor * df_lc_days['flux']), 'flux_err'] = np.abs(error_floor * df_lc_days.loc[(df_lc_days['filter']==unique_filter) & (df_lc_days['flux_err']< error_floor * df_lc_days['flux']), 'flux'])
    
    for unique_filter in unique_filters:
                
        df_lc_days_filter=df_lc_days[df_lc_days['filter']==unique_filter]    
        
        for time in range( int(min_mjd),int(max_mjd)):
            df_lc_day_filter=df_lc_days_filter[ (df_lc_days_filter['mjd'] >= time-0.5) & (df_lc_days_filter['mjd'] < time+0.5) ]
            
            if len(df_lc_day_filter) > 0 :
                
                if unique_filter == 'ztfg':
                    filt_wl = yeff_ztfg
                elif unique_filter == 'ztfr':
                    filt_wl = yeff_ztfr
                elif unique_filter == 'ztfi':
                    filt_wl = yeff_ztfi
                else:
                    print('not valid filter')
                
                mjd_av.append(np.mean(df_lc_day_filter['mjd']))
                avg,avgerr=np.average(df_lc_day_filter['flux'].values,weights=1.0/(df_lc_day_filter['flux_err'].values**2),returned=True)
                flux_av.append(avg)
                flux_err_av.append(np.sqrt(1.0/avgerr))
                filter_av.append(unique_filter)
                filter_wl_av.append(filt_wl)
                ZP_av.append(30.0)
         
    new_df['mjd']=mjd_av
    new_df['filter']=filter_av
    new_df['filter_wl']=filter_wl_av
    new_df['flux']=flux_av
    new_df['flux_err']=flux_err_av
    new_df['ZP']=ZP_av
    
    
    """
    wdetect= new_df['flux']/new_df['flux_err'] >= 5.0

    new_df.loc[wdetect, 'mag'] = -2.5 * np.log10(new_df['flux'])+30.0
    new_df.loc[wdetect, 'mag_err'] = 1.0857362 * new_df['flux_err']/new_df['flux']
    
    #new_df.loc[~wdetect, 'mag'] = -2.5 * np.log10(3.0*new_df['flux_err'])+30.0
    new_df.loc[~wdetect, 'mag'] = -2.5 * np.log10(5.0*new_df['flux_err'])+30.0
    
    new_df.loc[~wdetect, 'mag_err'] = np.nan
    
    new_df['phase'] = (new_df['mjd'] - t0)/(1.+redshift)
    """
#     print('stack gives ')
#     print(new_df)
    
    return new_df


def ext_mw_correct_lc(df_lc,ebv_mw,R_V=3.1):
    
    AV = R_V*ebv_mw
    
    unique_filters = np.unique(df_lc['filter'].values)
    
    for unique_filter in unique_filters:
    
        if unique_filter == 'ztfg':
            correction = extinction.fitzpatrick99(np.array([yeff_ztfg]), AV,  r_v=R_V)[0]
        elif unique_filter == 'ztfr':
            correction = extinction.fitzpatrick99(np.array([yeff_ztfr]), AV,  r_v=R_V)[0]
        elif unique_filter == 'ztfi':
            correction = extinction.fitzpatrick99(np.array([yeff_ztfi]), AV,  r_v=R_V)[0]
        else:
            print('not valid filter')

        df_lc.loc[df_lc['filter'] == unique_filter, 'flux'] = extinction.remove(correction, df_lc.loc[df_lc['filter'] == unique_filter, 'flux'])
        df_lc.loc[df_lc['filter'] == unique_filter, 'flux_err'] = extinction.remove(correction, df_lc.loc[df_lc['filter'] == unique_filter, 'flux_err'])
        
#     print('ext_mw_correct_lc gives ')
#     print(df_lc)
    
    return df_lc
    
def kcor_lc(df_lc,t0,x1,c,redshift,days_start=-10.0,days_end=40.0,nokcor=False): 
        
    unique_filters = np.unique(df_lc['filter'].values)
    """
    for unique_filter in unique_filters:
        
        filt_input = fset['fp_'+unique_filter]
        
        phase_data=(df_lc.loc[df_lc['filter'] == unique_filter, 'mjd']- t0)/(1.+redshift)
        phase_data_for_k=phase_data[ (phase_data>days_start) & (phase_data<days_end)]
        flux_data_for_k=df_lc.loc[df_lc['filter'] == unique_filter, 'flux'][ (phase_data>days_start) & (phase_data<days_end)]
        flux_err_data_for_k=df_lc.loc[df_lc['filter'] == unique_filter, 'flux_err'][ (phase_data>days_start) & (phase_data<days_end)]
        
        if nokcor == False:
            phase_array=np.arange(days_start, days_end+1, 1)
            model_base = sncosmo.SALT2Source(modeldir=salt_dir)
            model_base.set(x1=x1,c=c)
            k_array=[]
            for phase in phase_array:
                foo=model_base.flux([phase],wave_array_for_kcor)
                flux_array=foo[0]
                k,flag = kcorr.K(wave_array_for_kcor, flux_array, filt_input, filt_input, z=redshift)
                k_array.append(k)
            interp_k = interp1d(phase_array, k_array)
            k_interpol = interp_k(phase_data_for_k)
            k_interpol_flux_factor=10**((np.array(k_interpol))/(-2.5))
            flux_kcor=(flux_data_for_k/k_interpol_flux_factor)
            fluxerr_kcor=(flux_err_data_for_k/k_interpol_flux_factor)
        elif nokcor == True:
            flux_kcor=(flux_data_for_k)
            fluxerr_kcor=(flux_err_data_for_k)
        
        df_lc.loc[df_lc['filter'] == unique_filter, 'flux'] = flux_kcor
        df_lc.loc[df_lc['filter'] == unique_filter, 'flux_err'] = fluxerr_kcor
    """    
    df_lc.dropna(inplace=True)
    
    df_lc['phase'] = (df_lc['mjd'] - t0)/(1.+redshift)
    
    #wdetect= df_lc['flux']/df_lc['flux_err'] >= 3.0
    wdetect= df_lc['flux']/df_lc['flux_err'] >= 5.0

    df_lc.loc[wdetect, 'mag'] = -2.5 * np.log10(df_lc['flux'])+30.0
    df_lc.loc[wdetect, 'mag_err'] = 1.0857362 * df_lc['flux_err']/df_lc['flux']
    
    #df_lc.loc[~wdetect, 'mag'] = -2.5 * np.log10(3.0*df_lc['flux_err'])+30.0
    df_lc.loc[~wdetect, 'mag'] = -2.5 * np.log10(5.0*df_lc['flux_err'])+30.0
    
    df_lc.loc[~wdetect, 'mag_err'] = np.nan
        
    #df_lc['mag_err']=1.0857362 * df_lc['flux_err']/df_lc['flux']
    #df_lc['mag'][~wdetect]=-2.5 * np.log10(3.0*df_lc['flux_err'][~wdetect])+30.0
    #df_lc['mag_err'][~wdetect]=np.nan
    
#     print('kcor gives ')
#     print(df_lc.reset_index(drop=True))
    
    foo=df_lc.reset_index(drop=True)
    
    return foo

def quality_cut_lc(df_lc,quality_factor=8.0):
        
    factor_for_cut=np.nanmedian(df_lc['flux'])
        
    wqual=abs(df_lc['flux']/factor_for_cut)<quality_factor
        
    new_df=df_lc.loc[wqual]
    
#     print('quality_cut_lc gives ')
#     print(new_df.reset_index(drop=True))
    
    foo=new_df.reset_index(drop=True)
    
    return foo
    
def gp_2d_fit(df_lc,redshift,t0_salt,dt_peak=0.01,min_time_for_plot=-15.0, max_time_for_plot=+50.0,
              min_time_for_peak=-8.0, max_time_for_peak=+10.0,dt=0.05,n_optimizer=5, bounds = False):
    
    dismod = cosmo.distmod(redshift).value
    dismod_err = 5.0/np.log(10.0)*300/299792.458/redshift 
#     dismod_err2 = 0.00297321
#     dismod_err = np.sqrt(dismod_err1**2 + dismod_err2**2)
    
    unique_filters = np.unique(df_lc['filter'].values)
    
    stacked_data_x_for_fit = np.vstack([df_lc['mjd'].values, df_lc['filter_wl'].values]).T
    
    data_y_for_fit = df_lc['flux'].values
    data_yerr_for_fit = df_lc['flux_err'].values

    factor_gp=np.nanmedian(data_y_for_fit)
    data_y_scaled_fit=data_y_for_fit/factor_gp
    data_yerr_scaled_fit=data_yerr_for_fit/factor_gp
    
    mean_value  = np.nanmean(data_y_scaled_fit)
    min_value = 0.1*np.nanmean(data_y_scaled_fit)
    max_value = 10.0*np.nanmean(data_y_scaled_fit)
    
    lengthscales = [15.0,2000.0]

    wn=np.var(data_y_scaled_fit)
    
    #kernel = WhiteKernel(wn, (1e-10*wn,1e10*wn)) + RBF(length_scale=lengthscales, length_scale_bounds=(5, 30))  * ConstantKernel(constant_value=mean_value, constant_value_bounds=(np.absolute(min_value), np.absolute(max_value)))
    kernel = WhiteKernel(wn) + RBF(length_scale=lengthscales)  * ConstantKernel(constant_value=mean_value)
    
    if bounds == True:
        kernel = WhiteKernel(1e-5, (1e-8,1e0)) + RBF(10, (5, 30))  * ConstantKernel(mean_value, (min_value, max_value))
    
    gp_regressor = GaussianProcessRegressor(kernel=kernel, alpha=data_yerr_scaled_fit**2, random_state=0,n_restarts_optimizer=n_optimizer).fit(stacked_data_x_for_fit, data_y_scaled_fit)
    
    x_gp=[]
    wl_gp=[]
    filter_gp=[]
    
    for unique_filter in unique_filters:
        foo=np.arange(t0_salt+min_time_for_plot,t0_salt+max_time_for_plot,dt)
        x_gp.append(foo)
        
        if unique_filter == 'ztfg':
            wl_gp.append([yeff_ztfg] * len(foo))
            filter_gp.append(['ztfg']* len(foo))
        elif unique_filter == 'ztfr':
            wl_gp.append([yeff_ztfr] * len(foo))
            filter_gp.append(['ztfr']* len(foo))
        elif unique_filter == 'ztfi':
            wl_gp.append([yeff_ztfi] * len(foo))
            filter_gp.append(['ztfi']* len(foo))
        else:
            print('not valid filter')
                
    stacked_data = np.vstack([np.concatenate(x_gp), np.concatenate(wl_gp)]).T
        
    gp, gp_err = gp_regressor.predict(stacked_data, return_std=True)

    gp_res = factor_gp * gp.T
    gp_err_res = factor_gp * gp_err.T
    
    df_lc_fit = pd.DataFrame(columns = ['mjd','phase','filter','filter_wl', 'flux', 'flux_err_up','flux_err_down','ZP','mag','mag_err_up','mag_err_down'])

    df_lc_fit['mjd']=np.concatenate(x_gp)
    df_lc_fit['phase']=(np.concatenate(x_gp) - t0_salt)/(1.+redshift)

    df_lc_fit['filter']=np.concatenate(filter_gp)
    df_lc_fit['filter_wl']=np.concatenate(wl_gp)
    
    df_lc_fit['flux']=gp_res
    df_lc_fit['flux_err_up']=gp_res + 1 * gp_err_res
    df_lc_fit['flux_err_down']=gp_res - 1 * gp_err_res

    df_lc_fit['ZP']=30.0
    
    df_lc_fit['mag']= -2.5 * np.log10(gp_res) + 30.0
    df_lc_fit['mag_err_up']= -2.5 * np.log10(gp_res + 1 * gp_err_res) + 30.0
    df_lc_fit['mag_err_down']= -2.5 * np.log10(gp_res - 1 * gp_err_res) + 30.0
    
    filter_name_array=[]
    t0_array=[]
    
    mag0_array=[]
    mag0_err_array=[]
    magp15_array=[]
    magp15_err_array=[]
    magm5_array=[]
    magm5_err_array=[]
    magm10_array=[]
    magm10_err_array=[]
    
    absmag0_array=[]
    absmag0_err_array=[]
    absmagp15_array=[]
    absmagp15_err_array=[]
    absmagm5_array=[]
    absmagm5_err_array=[]
    absmagm10_array=[]
    absmagm10_err_array=[]
    
    dm_p15_array=[]
    dm_p15_err_array=[]
    
    dm_m5_array=[]
    dm_m5_err_array=[]
    
    dm_m10_array=[]
    dm_m10_err_array=[]
    
    for unique_filter in unique_filters:
        
        if unique_filter == 'ztfg':
            unique_filter_wl=yeff_ztfg
        elif unique_filter == 'ztfr':
            unique_filter_wl=yeff_ztfr
        elif unique_filter == 'ztfi':
            unique_filter_wl=yeff_ztfi
        else:
            print('not valid filter')
        
        filter_name_array.append(unique_filter)
        
        df_lc_fit_filter=df_lc_fit[df_lc_fit['filter']==unique_filter]
        
        factor=np.nanmedian(df_lc_fit_filter['flux'])
        flux_scaled_fit=df_lc_fit_filter['flux']/factor
        
        #find peak
        y_spl = UnivariateSpline(df_lc_fit_filter['mjd'], flux_scaled_fit,s=0,k=4)
        y_spl_1d = y_spl.derivative(n=1)
        x_peak = np.arange(t0_salt+min_time_for_peak, t0_salt+max_time_for_peak, dt_peak)
        rounded_y = np.round(y_spl_1d(x_peak), 4)
        minima = x_peak[np.where(np.abs(rounded_y) == np.nanmin(np.abs(rounded_y)))[0]]
        t0_peak = np.mean(minima)
        t0_array.append(t0_peak)
        
        #t0_array.append(t0_salt)  # hack for t0 salt as peak
        
#         phase_plot=np.arange(t0_salt+min_time_for_plot,t0_salt+max_time_for_plot,dt)

#         df_lc_fit.loc[df_lc_fit['filter'] == unique_filter, 'phase'] = (phase_plot - t0_peak)/(1.+redshift)
        
        flux_peak,fluxerr_peak=gp_regressor.predict(np.vstack([t0_peak,unique_filter_wl]).T, return_std=True)
                
        flux_p15, fluxerr_p15 = gp_regressor.predict(np.vstack([t0_peak + 15.0*(1.+redshift),unique_filter_wl]).T, return_std=True)
        
        flux_m5, fluxerr_m5 = gp_regressor.predict(np.vstack([t0_peak - 5.0*(1.+redshift),unique_filter_wl]).T, return_std=True)
        
        flux_m10, fluxerr_m10 = gp_regressor.predict(np.vstack([t0_peak - 10.0*(1.+redshift),unique_filter_wl]).T, return_std=True)

        #if flux_peak[0]/fluxerr_peak[0] >= 3.0:
        if flux_peak[0]/fluxerr_peak[0] >= 5.0:
            mag_0 = -2.5 * np.log10(flux_peak[0]*factor_gp) + 30.0
            mag_0_err = 1.0857362 * fluxerr_peak[0]/flux_peak[0]
            mag_0_abs = mag_0 - dismod
            mag_0_abs_err=np.sqrt( dismod_err**2 + mag_0_err**2 )
        else:
            #mag_0 = -2.5 * np.log10(3.0*fluxerr_peak[0]*factor_gp) + 30.0
            mag_0 = -2.5 * np.log10(5.0*fluxerr_peak[0]*factor_gp) + 30.0
            mag_0_err = np.nan
            mag_0_abs = mag_0 - dismod
            mag_0_abs_err= np.nan
                
        mag0_array.append(mag_0)
        mag0_err_array.append(mag_0_err)
        absmag0_array.append(mag_0_abs)
        absmag0_err_array.append(mag_0_abs_err)
        
        #if flux_p15[0]/fluxerr_p15[0] >= 3.0:
        if flux_p15[0]/fluxerr_p15[0] >= 5.0:
            mag_p15 = -2.5 * np.log10(flux_p15[0]*factor_gp) + 30.0
            mag_p15_err = 1.0857362 * fluxerr_p15[0]/flux_p15[0]
            mag_p15_abs = mag_p15 - dismod
            mag_p15_abs_err=np.sqrt( dismod_err**2 + mag_p15_err**2 )
        else:
            #mag_p15 = -2.5 * np.log10(3.0*fluxerr_p15[0]*factor_gp) + 30.0
            mag_p15 = -2.5 * np.log10(5.0*fluxerr_p15[0]*factor_gp) + 30.0
            mag_p15_err = np.nan
            mag_p15_abs = mag_p15 - dismod
            mag_p15_abs_err= np.nan
        
        magp15_array.append(mag_p15)
        magp15_err_array.append(mag_p15_err)
        absmagp15_array.append(mag_p15_abs)
        absmagp15_err_array.append(mag_p15_abs_err)
        
        #if flux_m5[0]/fluxerr_m5[0] >= 3.0:
        if flux_m5[0]/fluxerr_m5[0] >= 5.0:
            mag_m5 = -2.5 * np.log10(flux_m5[0]*factor_gp) + 30.0
            mag_m5_err = 1.0857362 * fluxerr_m5[0]/flux_m5[0]
            mag_m5_abs = mag_m5 - dismod
            mag_m5_abs_err=np.sqrt( dismod_err**2 + mag_m5_err**2 )
        else:
            #mag_m5 = -2.5 * np.log10(3.0*fluxerr_m5[0]*factor_gp) + 30.0
            mag_m5 = -2.5 * np.log10(5.0*fluxerr_m5[0]*factor_gp) + 30.0
            mag_m5_err = np.nan
            mag_m5_abs = mag_m5 - dismod
            mag_m5_abs_err= np.nan
        
        magm5_array.append(mag_m5)
        magm5_err_array.append(mag_m5_err)
        absmagm5_array.append(mag_m5_abs)
        absmagm5_err_array.append(mag_m5_abs_err)
        
        #if flux_m10[0]/fluxerr_m10[0] >= 3.0:
        if flux_m10[0]/fluxerr_m10[0] >= 5.0:
            mag_m10 = -2.5 * np.log10(flux_m10[0]*factor_gp) + 30.0
            mag_m10_err = 1.0857362 * fluxerr_m10[0]/flux_m10[0]
            mag_m10_abs = mag_m10 - dismod
            mag_m10_abs_err=np.sqrt( dismod_err**2 + mag_m10_err**2 )
        else:
            #mag_m10 = -2.5 * np.log10(3.0*fluxerr_m10[0]*factor_gp) + 30.0
            mag_m10 = -2.5 * np.log10(5.0*fluxerr_m10[0]*factor_gp) + 30.0
            mag_m10_err = np.nan
            mag_m10_abs = mag_m10 - dismod
            mag_m10_abs_err= np.nan
        
        magm10_array.append(mag_m10)
        magm10_err_array.append(mag_m10_err)
        absmagm10_array.append(mag_m10_abs)
        absmagm10_err_array.append(mag_m10_abs_err)
        
#         if ( (flux_peak[0]/fluxerr_peak[0] >= 5.0 ) & (flux_p15[0]/fluxerr_p15[0] >= 5.0 ) ):
#             dm_p15 = mag_p15 - mag_0
#             dm_p15_err = np.sqrt( mag_p15_err**2 + mag_0_err**2 )
#         else:
#             dm_p15 = np.sqrt( mag_p15_err**2 + mag_0_err**2 )
#             dm_p15_err = np.nan
        
        #dm_p15 = mag_p15 - mag_0
        # new
        for_spline = df_lc_fit_filter.dropna(subset = ['mag'])
        y_spl = UnivariateSpline(for_spline['phase'], for_spline['mag'],s=0,k=4)
        dm_p15 = y_spl(15) - y_spl(0)
        dm_p15_err = np.sqrt( mag_p15_err**2 + mag_0_err**2 )
        
        dm_m5 = mag_m5 - mag_0
        dm_m5_err = np.sqrt( mag_m5_err**2 + mag_0_err**2 )
        
        dm_m10 = mag_m10 - mag_0
        dm_m10_err = np.sqrt( mag_m10_err**2 + mag_0_err**2 )
        
        dm_p15_array.append(dm_p15)
        dm_p15_err_array.append(dm_p15_err)
        
        dm_m5_array.append(dm_m5)
        dm_m5_err_array.append(dm_m5_err)
        
        dm_m10_array.append(dm_m10)
        dm_m10_err_array.append(dm_m10_err)

    
    
    df_lc_fit.dropna(inplace=True)
    
    df_lc_fit2=df_lc_fit.reset_index(drop=True)
    
    df_lc_fit_params = pd.DataFrame(columns = ['filter','t0','mag_0','mag_0_err','mag_p15',
                                                    'mag_p15_err','mag_m5','mag_m5_err',
                                                    'mag_m10','mag_m10_err','absmag_0','absmag_0_err',
                                                    'absmag_p15','absmag_p15_err','absmag_m5','absmag_m5_err',
                                                    'absmag_m10','absmag_m10_err','dm_p15','dm_p15_err',
                                                    'dm_m5','dm_m5_err','dm_m10','dm_m10_err'])
    df_lc_fit_params['filter']=filter_name_array
    df_lc_fit_params['t0']=t0_array
    df_lc_fit_params['mag_0']=mag0_array
    df_lc_fit_params['mag_0_err']=mag0_err_array
    df_lc_fit_params['mag_p15']=magp15_array
    df_lc_fit_params['mag_p15_err']=magp15_err_array
    df_lc_fit_params['mag_m5']=magm5_array
    df_lc_fit_params['mag_m5_err']=magm5_err_array
    df_lc_fit_params['mag_m10']=magm10_array
    df_lc_fit_params['magm10_err']=magm10_err_array
    df_lc_fit_params['absmag_0']=absmag0_array
    df_lc_fit_params['absmag_0_err']=absmag0_err_array
    df_lc_fit_params['absmag_p15']=absmagp15_array
    df_lc_fit_params['absmag_p15_err']=absmagp15_err_array
    df_lc_fit_params['absmag_m5']=absmagm5_array
    df_lc_fit_params['absmag_m5_err']=absmagm5_err_array
    df_lc_fit_params['absmag_m10']=absmagm10_array
    df_lc_fit_params['absmagm10_err']=absmagm10_err_array
    df_lc_fit_params['dm_p15']=dm_p15_array
    df_lc_fit_params['dm_p15_err']=dm_p15_err_array
    df_lc_fit_params['dm_m5']=dm_m5_array
    df_lc_fit_params['dm_m5_err']=dm_m5_err_array
    df_lc_fit_params['dm_m10']=dm_m10_array
    df_lc_fit_params['dm_m10_err']=dm_m10_err_array
    
#     print('fit gives ')
#     print(df_lc_fit)
#     print(df_lc_fit_params)

    return df_lc_fit2,df_lc_fit_params,gp_regressor, factor_gp






def calculate_time_peak_error(df_lc_fit,gp_regressor,band,time_peak,min_time_for_peak,max_time_for_peak,number_samples=100):
    
    if band == 'ztfg':
        unique_filter_wl=yeff_ztfg
    elif band == 'ztfr':
        unique_filter_wl=yeff_ztfr
    elif band == 'ztfi':
        unique_filter_wl=yeff_ztfi
    else:
        print('not valid filter')
    
    stacked_data_x = np.vstack([df_lc_fit['mjd'].values, df_lc_fit['filter_wl'].values]).T
    
    samples=gp_regressor.sample_y(stacked_data_x,n_samples=number_samples, random_state=0)
    
    times_array=[]
    for x in range(number_samples):
        #new_df = pd.DataFrame(columns = ['mjd', 'flux'])
        #new_df['mjd'] = df_lc_fit['mjd'][ df_lc_fit['filter_wl'] == unique_filter_wl]
        #new_df['flux'] = samples[:,x][ df_lc_fit['filter_wl'] == unique_filter_wl ]
        mjd = df_lc_fit['mjd'][ df_lc_fit['filter_wl'] == unique_filter_wl].values
        fl = samples[:,x][ df_lc_fit['filter_wl'] == unique_filter_wl ]
        
        #find peak
        y_spl = UnivariateSpline(mjd, fl/np.nanmedian(fl),s=0,k=4)
        y_spl_1d = y_spl.derivative(n=1)
        x_peak = np.arange(time_peak+min_time_for_peak, time_peak+max_time_for_peak, 0.05)
        rounded_y = np.round(y_spl_1d(x_peak), 4)
        minima = x_peak[np.where(np.abs(rounded_y) == np.nanmin(np.abs(rounded_y)))[0]]
        t0_peak = np.mean(minima)
        times_array.append(t0_peak)
    
    #print(times_array)
    
    return times_array  #np.mean(times_array),np.std(times_array)

def color_sn(gp_regressor,c1,c2,c1_mjd):  #factor_gp
    
    if c1 == 'ztfg':
        unique_filter_wl_c1=yeff_ztfg
    elif c1 == 'ztfr':
        unique_filter_wl_c1=yeff_ztfr
    elif c1 == 'ztfi':
        unique_filter_wl_c1=yeff_ztfi
    else:
        print('not valid filter')
    
    if c2 == 'ztfg':
        unique_filter_wl_c2=yeff_ztfg
    elif c2 == 'ztfr':
        unique_filter_wl_c2=yeff_ztfr
    elif c2 == 'ztfi':
        unique_filter_wl_c2=yeff_ztfi
    else:
        print('not valid filter')

    flux_c1,fluxerr_c1=gp_regressor.predict(np.vstack([c1_mjd,unique_filter_wl_c1]).T, return_std=True)
    flux_c2,fluxerr_c2=gp_regressor.predict(np.vstack([c1_mjd,unique_filter_wl_c2]).T, return_std=True)

    #if flux_c1[0]/fluxerr_c1[0] >= 3.0:
    if flux_c1[0]/fluxerr_c1[0] >= 5.0:
        mag_c1 = -2.5 * np.log10(flux_c1[0] ) + 30.0
        mag_err_c1 = 1.0857362 * fluxerr_c1[0]/flux_c1[0]
    else:
        #mag_c1 = -2.5 * np.log10(3.0 * fluxerr_c1[0] ) + 30.0
        mag_c1 = -2.5 * np.log10(5.0 * fluxerr_c1[0] ) + 30.0
        mag_err_c1 = np.nan
    
    #if flux_c2[0]/fluxerr_c2[0] >= 3.0:
    if flux_c2[0]/fluxerr_c2[0] >= 5.0:
        mag_c2 = -2.5 * np.log10(flux_c2[0] ) + 30.0
        mag_err_c2 = 1.0857362 * fluxerr_c2[0]/flux_c2[0]
    else:
       #mag_c2 = -2.5 * np.log10(3.0 * fluxerr_c2[0] ) + 30.0
        mag_c2 = -2.5 * np.log10(5.0 * fluxerr_c2[0] ) + 30.0
        mag_err_c2 = np.nan
    
    c1_c2=mag_c1-mag_c2
    c1_c2_err=np.sqrt( mag_err_c1**2 + mag_err_c2**2 )
    
    return c1_c2,c1_c2_err

def coverage_lc(df_lc,t0,redshift,p_start,p_finish):
    df_lc_detect=df_lc[df_lc['flux']/df_lc['flux_err'] >= 5.0]
    df_lc_phase = (df_lc_detect['mjd'] - t0)/(1.+redshift)
    n_points=int(len(df_lc_phase[(df_lc_phase.values > p_start) & (df_lc_phase.values <= p_finish )]))
    return n_points


def analyse_lc(gp_fit, lc, z, t0, band = 'ztfr', min_time = 13, max_time = 35, plot = False):
        
    lc = lc.loc[lc['filter']==band]
    lc = lc.reset_index(drop=True)

    gp = gp_fit.loc[gp_fit['filter']==band]
    gp = gp.reset_index(drop=True)
    
    gp_r = gp_fit.loc[gp_fit['filter']=='ztfr']
    gp_r = gp_r.reset_index(drop=True)
    df_dt_from_zero, time_shoulder, flux_int_norm, flux_int_10_20_norm, flux_int_20_30_norm, flux_int_30_40_norm= np.nan,np.nan,np.nan,np.nan,np.nan, np.nan
    flux_int_0_5_norm,flux_int_5_10_norm,flux_int_10_15_norm,flux_int_15_20_norm,flux_int_20_25_norm,flux_int_25_30_norm, flux_int_30_35_norm, time_bump =np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if len(gp)>1 and len(gp_r)>1:
        
        y_spl = UnivariateSpline(gp_r['mjd'], gp_r['flux'],s=0,k=4)

        x_fit = np.arange(np.nanmin(gp_r['mjd']), max_time, 0.01)

        y_spl_1d = y_spl.derivative(n=1)
        y_spl_2d = y_spl.derivative(n=2)

        x_first_bump = np.arange(-10 + t0, 10 + t0, 0.01)
        rounded_y = np.round(y_spl_1d(x_first_bump), 4)
        minima = x_first_bump[np.where(np.abs(rounded_y) == np.nanmin(np.abs(rounded_y)))[0][0]]
        t0_gp = np.nanmean(minima)
        peak_flux = y_spl(np.mean(t0_gp))        
        flux_15d = y_spl(np.mean(t0_gp + 15))
    
        
        gp['phase_r'] = (gp['mjd'] - t0_gp) / (1. + z)
        lc['phase_r'] = (lc['mjd'] - t0_gp) / (1. + z)
        
        y_spl = UnivariateSpline(gp['phase_r'], gp['flux'] / peak_flux, s=0,k=4)

        x_fit = np.arange(np.nanmin(gp['phase_r']), max_time, 0.01)

        y_spl_1d = y_spl.derivative(n=1)
        y_spl_2d = y_spl.derivative(n=2)

        x_int = np.arange(10, 20, 0.01)
        integral = integrate.quad(lambda x_int: y_spl(x_int), 10, 20)
        flux_int = integral[0] / 10
        flux_int_10_20_norm = flux_int #/ peak_flux
        
        x_int = np.arange(20, 30, 0.01)
        integral = integrate.quad(lambda x_int: y_spl(x_int), 20, 30)
        flux_int = integral[0] / 10
        flux_int_20_30_norm = flux_int #/ peak_flux
        
        x_int = np.arange(30, 40, 0.01)
        integral = integrate.quad(lambda x_int: y_spl(x_int), 30, 40)
        flux_int = integral[0] / 10
        flux_int_30_40_norm = flux_int #/ peak_flux
        
        x_int = np.arange(0, 5, 0.01)
        integral = integrate.quad(lambda x_int: y_spl(x_int), 0, 5)
        flux_int = integral[0] / 5
        flux_int_0_5_norm = flux_int #/ peak_flux
        
        x_int = np.arange(5, 10, 0.01)
        integral = integrate.quad(lambda x_int: y_spl(x_int), 5, 10)
        flux_int = integral[0] / 5
        flux_int_5_10_norm = flux_int #/ peak_flux
        
        x_int = np.arange(10, 15, 0.01)
        integral = integrate.quad(lambda x_int: y_spl(x_int), 10, 15)
        flux_int = integral[0] / 5
        flux_int_10_15_norm = flux_int #/ peak_flux

        x_int = np.arange(15, 20, 0.01)
        integral = integrate.quad(lambda x_int: y_spl(x_int), 15, 20)
        flux_int = integral[0] / 5
        flux_int_15_20_norm = flux_int #/ peak_flux

        x_int = np.arange(20, 25, 0.01)
        integral = integrate.quad(lambda x_int: y_spl(x_int), 20, 25)
        flux_int = integral[0] / 5
        flux_int_20_25_norm = flux_int #/ peak_flux
        
        x_int = np.arange(25, 30, 0.01)
        integral = integrate.quad(lambda x_int: y_spl(x_int), 25, 30)
        flux_int = integral[0] / 5
        flux_int_25_30_norm = flux_int #/ peak_flux

        x_int = np.arange(30, 35, 0.01)
        integral = integrate.quad(lambda x_int: y_spl(x_int), 30, 35)
        flux_int = integral[0] / 5
        flux_int_30_35_norm = flux_int #/ peak_flux
        
        x_int = np.arange(15, 40, 0.01)
        #integral = integrate.quad(lambda x_int: y_spl(x_int), 15, 40)
        integral = y_spl.integral(15, 40)
        flux_int = integral / 25
        mean_flux = np.nanmean(y_spl(x_int))
        #flux_int_norm = flux_int / peak_flux
        flux_int_norm = flux_int #/ mean_flux

    
        
        x_second_bump = np.arange(min_time, max_time, 0.01)
        time_shoulder = np.nan
        minima = x_second_bump[np.where(np.abs(y_spl_1d(x_second_bump)) == np.nanmin(np.abs(y_spl_1d(x_second_bump))))[0][0]]
        df_dt_from_zero = np.nan
        time_bump = np.nan
        days = np.arange(7, max_time, 0.001)
        # changed range from 10-maxtime to 12-maxtime
        
        crosses_1d = []
        for k in range(len(days)-1):
            sign = y_spl_1d(days[k]) * y_spl_1d(days[k+1])
            if sign < 0:
                crosses_1d.append(days[k])
                
        if np.round(y_spl_1d(minima), 3) == 0 and y_spl_2d(minima) <= 0:
            time_bump = minima
        
        days = np.arange(0, max_time, 0.001)
        crosses_2d = []
        for k in range(len(days)-1):
            sign = y_spl_2d(days[k]) * y_spl_2d(days[k+1])
            if sign < 0:
                crosses_2d.append(days[k])
 
        if len(crosses_2d) >= 2:
            new_minima = crosses_2d[1]
            #new_second_bump_range = np.arange(crosses_2d[1] -0.5, crosses_2d[1] +0.5, 0.001)
            #new_minima = new_second_bump_range[np.where(np.abs(y_spl_2d(new_second_bump_range)) == np.nanmin(np.abs(y_spl_2d(new_second_bump_range))))[0][0]]
            #print(new_minima,y_spl_2d(new_minima) )
            if np.round(new_minima, 0) <= max_time and np.round(new_minima, 0) > min_time:# and np.round(y_spl_2d(new_minima), 3) == 0:
                print("INFLECTION POINT")
                print(new_minima,y_spl_2d(new_minima) )

                time_shoulder = new_minima
                df_dt_from_zero = y_spl_1d(time_shoulder)

    return df_dt_from_zero,time_shoulder, flux_int_norm, flux_int_10_20_norm, flux_int_20_30_norm, flux_int_30_40_norm,flux_int_0_5_norm,flux_int_5_10_norm,flux_int_10_15_norm,flux_int_15_20_norm,flux_int_20_25_norm,flux_int_25_30_norm, flux_int_30_35_norm, time_bump


def deltam15(gp, t0_salt, z):
    df_lc_fit_filter = gp.loc[gp['filter']=='ztfg']

    y_spl = UnivariateSpline(df_lc_fit_filter['mjd'], df_lc_fit_filter['flux'],s=0,k=4)
    y_spl_1d = y_spl.derivative(n=1)
    x_peak = np.arange(t0_salt-7, t0_salt+7, 0.01)
    rounded_y = np.round(y_spl_1d(x_peak), 4)
    minima = x_peak[np.where(np.abs(rounded_y) == np.nanmin(np.abs(rounded_y)))[0]]
    t0_peak = np.mean(minima)

    df_lc_fit_filter.loc[:, 'phase'] = (df_lc_fit_filter['mjd'] - t0_peak) / (1.+z)
    for_spline = df_lc_fit_filter.dropna(subset = ['mag'])
    y_spl = UnivariateSpline(for_spline['phase'], for_spline['mag'],s=0,k=4)
    deltam15 = y_spl(15) - y_spl(0)

    return deltam15



def perturb_lc(lc,realisations):
    
    rand=np.random.normal(lc['flux'], lc['flux_err'], size=(realisations,len(lc)) )
    
    realisation_array=[]
    mjd_array=[]
    filter_array=[]
    filter_wl_array=[]
    flux_array=[]
    flux_err_array=[]
    ZP_array=[]

    for i in range(realisations):
        foo=([int(i)]* len(lc))
        realisation_array.append(foo)
        mjd_array.append(lc['mjd'])
        filter_array.append(lc['filter'])
        filter_wl_array.append(lc['filter_wl'])
        flux_array.append(rand[i])
        flux_err_array.append(lc['flux_err'])
        ZP_array.append(lc['ZP'])
    
    new_df = pd.DataFrame(columns = ['realisation','mjd','filter','filter_wl', 'flux', 'flux_err','ZP'])
    
    new_df.loc[:, 'realisation']=np.concatenate(realisation_array)
    new_df.loc[:, 'mjd']=np.concatenate(mjd_array)
    new_df.loc[:, 'filter']=np.concatenate(filter_array)
    new_df.loc[:, 'filter_wl']=np.concatenate(filter_wl_array)
    new_df.loc[:, 'flux']=np.concatenate(flux_array)
    new_df.loc[:, 'flux_err']=np.concatenate(flux_err_array)
    new_df.loc[:, 'ZP']=np.concatenate(ZP_array)
    
    return new_df
