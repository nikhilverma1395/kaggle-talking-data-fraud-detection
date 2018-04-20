import pandas as pd
import gc
import dask as dd
import dask.dataframe as ddf


def pp(path,type_d,dtype,parse_dates,lm=False):
    
    print('Reading ',path)
    
    # Doing the merge operations with Dask, even on a 64G system - 16 cores, threw me a Memory Error, so back to plain old pandas for train
    if type_d != 'test':
        ddf = pd
       
    df = ddf.read_csv(path,low_memory=lm,dtype=dtype,parse_dates=parse_dates)
    
    if type_d == 'test':
        df=df.set_index('click_id')
        meta = ('click_time', 'datetime64[ns]')
        click_time_form = df.click_time.map_partitions(pd.to_datetime, meta=meta)
    else:
        click_time_form = pd.to_datetime(df.click_time)
    
    df['hr'] =  click_time_form.dt.hour.astype('uint8')
    df['day'] =  click_time_form.dt.day.astype('uint8')
    df['dow'] =  click_time_form.dt.dayofweek.astype('uint8')
    df['min'] =  click_time_form.dt.minute.astype('uint8')
    
    print(df.columns.values)
    
    
    del click_time_form;
    gc.collect()

    ## ip,app
    ## ip,app,os
    ## ip,device,app 
    ## ip,device
    ## ip,device,os  
    ## ip,os
    ## ip,channel
    ## ip,channel,hr 
    ## ip,hr
    ## ...
    

    # do merge operations with dask
    ip_app = df[['ip','app','channel']].groupby(by=['ip','app'])[['channel']].count().rename(columns={'channel':'ip_app_count'}).reset_index().astype('uint32')
    df = df.merge(ip_app,on=['ip','app'], how='left')
    print('M1')
    
    del ip_app;
    gc.collect()
    
    ip_app_os = df[['ip','os','app','channel']].groupby(by=['ip','app','os'])[['channel']].count().rename(columns={'channel':'ip_app_os_count'}).reset_index().astype('uint32')
    df = df.merge(ip_app_os,on=['ip','app','os'], how='left')
    print('M2')
    del ip_app_os;
    gc.collect()
    
    ip_app_device = df[['ip','device','app','channel']].groupby(by=['ip','app','device'])[['channel']].count().rename(columns={'channel':'ip_app_device_count'}).reset_index().astype('uint32')
    df = df.merge(ip_app_device,on=['ip','app','device'], how='left')
    print('M3')
    del ip_app_device;
    gc.collect()
    
    ip_dev = df[['ip','device','channel']].groupby(by=['ip','device'])[['channel']].count().rename(columns= {'channel':'ip_dev_count'}).reset_index().astype('uint32')
    df = df.merge(ip_dev,on=['ip','device'], how='left');
    print('M4')
    del ip_dev;
    gc.collect()
    
    ip_dev_os = df[['ip','device','os','channel']].groupby(by=['ip','device','os'])[['channel']].count().rename(columns= {'channel':'ip_dev_os_count'}).reset_index().astype('uint32')
    df = df.merge(ip_dev_os,on=['ip','device','os'], how='left');
    print('M5')
    del ip_dev_os;
    gc.collect()

    ip_os = df[['ip','os','channel']].groupby(by=['ip','os'])[['channel']].count().rename(columns={'channel':'ip_os_count'}).reset_index().astype('uint32')
    df = df.merge(ip_os,on=['ip','os'], how='left');
    print('M6')

    del ip_os;
    gc.collect()

    ip_ch = df[['ip','os','channel']].groupby(by=['ip','channel'])[['os']].count().rename(columns={'os':'ip_ch_count'}).reset_index().astype('uint32')
    df = df.merge(ip_ch,on=['ip','channel'], how='left');
    print('M7')

    del ip_ch;
    gc.collect()
    
    ip_ch_hr = df[['ip','os','channel','hr']].groupby(by=['ip','channel','hr'])[['os']].count().rename(columns={'os':'ip_ch_hr_count'}).reset_index().astype('uint32')
    df = df.merge(ip_ch_hr,on=['ip','channel','hr'], how='left');
    print('M8')

    del ip_ch_hr;
    gc.collect()
   
    
    ip_hr_level_data = df[['hr','ip','os']].groupby(['ip','hr'])[['os']].count().rename(columns={'os':'ip_hr_count'}).reset_index().astype('uint32')
    df = df.merge(ip_hr_level_data,on=['ip','hr'], how='left')
    print('M9')
    
    del ip_hr_level_data;
    gc.collect()
    
    #
    gp = df[['ip', 'day', 'hr', 'channel']].groupby(by=['ip', 'day', 'hr'])[['channel']].count().reset_index().rename(
             columns={'channel': 'ip_day_hh_count'})
    df = df.merge(gp, on=['ip','day','hr'], how='left')
    print('M10')

    del gp;
    gc.collect()

    gp = df[['ip', 'day', 'os', 'hr', 'channel']].groupby(by=['ip', 'os', 'day','hr'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_hh_os_count'}).astype('uint32')
    df = df.merge(gp, on=['ip','os','hr','day'], how='left')
    
    print('M11')
    del gp
    gc.collect()

    gp = df[['ip', 'app', 'hr', 'day', 'channel']].groupby(by=['ip', 'app', 'day','hr'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_hh_day_count'}).astype('uint32')

    df = df.merge(gp, on=['ip','app','hr','day'], how='left')
    
    print('M12')
    del gp
    gc.collect()

    gp = df[['ip', 'device', 'hr', 'day', 'channel']].groupby(by=['ip', 'device', 'day','hr'])[['channel']].count().reset_index().rename( columns={'channel': 'ip_dev_day_hr_count'}).astype('uint32')
    df = df.merge(gp, on=['ip','device','day','hr'], how='left')
    
    print('M13')
    del gp
    gc.collect()
    
    gp = df[['ip', 'app', 'min', 'channel']].groupby(by=['ip', 'app', 'min'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_min_count'}).astype('uint32')
    df = df.merge(gp, on=['ip', 'app', 'min'], how='left')
    
    print('M14')
    del gp
    gc.collect()
    
    
    gp = df[['ip', 'day', 'min', 'channel']].groupby(by=['ip', 'channel', 'min'])[['day']].count().reset_index().rename(columns={'day': 'ip_ch_min_count'}).astype('uint32')
    df = df.merge(gp, on=['ip','min', 'channel'], how='left')
    
    print('M15')
    del gp
    gc.collect()
    
    gp = df[['ip', 'day', 'dow', 'channel']].groupby(by=['ip', 'channel', 'dow'])[['day']].count().reset_index().rename(columns={'day': 'ip_ch_dow_count'}).astype('uint32')
    df = df.merge(gp, on=['ip','dow', 'channel'], how='left')
    
    print('M16')
    del gp
    gc.collect()
    
    gp = df[['ip', 'app', 'dow', 'channel']].groupby(by=['ip', 'app', 'dow'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_dow_count'}).astype('uint32')
    df = df.merge(gp, on=['ip','dow', 'app'], how='left')
    
    print('M17')
    del gp
    gc.collect()
    
    return df

def dropDicKeys(tdict,col):
    for c in col:
        tdict.pop(c,None)
    return tdict
    
   
    
    

    