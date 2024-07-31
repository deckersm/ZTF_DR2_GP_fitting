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
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=73, Om0=0.3)
import warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning)


yeff_ztfg = 4753.2  # phot wl from http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=Palomar/ZTF.g&&mode=browse&gname=Palomar&gname2=ZTF#filter
yeff_ztfr = 6370.98 # phot wl from http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=Palomar/ZTF.g&&mode=browse&gname=Palomar&gname2=ZTF#filter
yeff_ztfi = 7912.99 # phot wl from http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=Palomar/ZTF.g&&mode=browse&gname=Palomar&gname2=ZTF#filter

salt_dir="path_to_salt_directory"
wave_array_for_kcor=np.linspace(3000, 9200, 6201)

def collect_lc(sn_name):
    file_sn_lc='path_to_light_curves/'+sn_name+'_LC.csv'
    lc_all=pd.read_csv(file_sn_lc,comment='#',delim_whitespace=True)
    lc_clean = lc_all[(lc_all.flag&1==0) & (lc_all.flag&2==0) & (lc_all.flag&4==0) & (lc_all.flag&8==0) & (lc_all.flag&16==0)]
    lc_clean2 = lc_clean[ (lc_clean.x_pos >= 0) & (lc_clean.y_pos >= 0)]
    new_df = lc_clean2[ ['mjd', 'filter','flux', 'flux_err','ZP'] ].reset_index(drop=True)
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
        
    return df_lc
    
def kcor_lc(df_lc,t0,x1,c,redshift,days_start=-10.0,days_end=40.0,nokcor=False): 
        
    unique_filters = np.unique(df_lc['filter'].values)
    # uncomment below if SNooPy is installed and you want to apply K-corrections
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

    # detection limit = 5 sigma
    wdetect= df_lc['flux']/df_lc['flux_err'] >= 5.0

    df_lc.loc[wdetect, 'mag'] = -2.5 * np.log10(df_lc['flux'])+30.0
    df_lc.loc[wdetect, 'mag_err'] = 1.0857362 * df_lc['flux_err']/df_lc['flux']
    
    df_lc.loc[~wdetect, 'mag'] = -2.5 * np.log10(5.0*df_lc['flux_err'])+30.0
    
    df_lc.loc[~wdetect, 'mag_err'] = np.nan
    
    foo=df_lc.reset_index(drop=True)
    
    return foo

# 2D GP fitting function    
# Also outputs key light curve parameters
def gp_2d_fit(df_lc,redshift,t0_salt,dt_peak=0.01,min_time_for_plot=-15.0, max_time_for_plot=+50.0,
              min_time_for_peak=-8.0, max_time_for_peak=+10.0,dt=0.05,n_optimizer=5, bounds = False):
    
    dismod = cosmo.distmod(redshift).value
    dismod_err = 5.0/np.log(10.0)*300/299792.458/redshift 
  
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

    return df_lc_fit2,df_lc_fit_params,gp_regressor, factor_gp

