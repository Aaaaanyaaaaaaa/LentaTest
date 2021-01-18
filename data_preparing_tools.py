#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pyarrow
import datetime as dt
import numpy as np
import scipy
import scipy.stats


# In[4]:


global WEEKS_ENCODED
WEEKS_ENCODED ={'2016-10-03/2016-10-09' : '01', '2016-10-10/2016-10-16'  : '02',
             '2016-10-17/2016-10-23' : '03', '2016-10-24/2016-10-30' : '04',
             '2016-10-31/2016-11-06' : '05', '2016-11-07/2016-11-13' : '06',
             '2016-11-14/2016-11-20' : '07', '2016-11-21/2016-11-27' : '08',
             '2016-11-28/2016-12-04' : '09', '2016-12-05/2016-12-11' : '10',
             '2016-12-12/2016-12-18' : '11', '2016-12-19/2016-12-25' : '12',
             '2016-12-26/2017-01-01' : '13', '2017-01-02/2017-01-08' : '14',
             '2017-01-09/2017-01-15' : '15', '2017-01-16/2017-01-22' : '16',
             '2017-01-23/2017-01-29' : '17', '2017-01-30/2017-02-05' : '18',
             '2017-02-06/2017-02-12' : '19', '2017-02-13/2017-02-19' : '20',
             '2017-02-20/2017-02-26' : '21', '2017-02-27/2017-03-05' : '22',
             '2017-03-06/2017-03-12' : '23', '2017-03-13/2017-03-19' : '24',
             '2017-03-20/2017-03-26' : '25', '2017-03-27/2017-04-02' : '26',
             '2017-04-03/2017-04-09' : '27', '2017-04-10/2017-04-16' : '28',
             '2017-04-17/2017-04-23' : '29', '2017-04-24/2017-04-30' : '30',
             '2017-05-01/2017-05-07' : '31', '2017-05-08/2017-05-14' : '32',
             '2017-05-15/2017-05-21' : '33', '2017-05-22/2017-05-28' : '34',
             '2017-05-29/2017-06-04' : '35', '2017-06-05/2017-06-11' : '36',
             '2017-06-12/2017-06-18' : '37', '2017-06-19/2017-06-25' : '38',
             '2017-06-26/2017-07-02' : '39', '2017-07-03/2017-07-09' : '40',
             '2017-07-10/2017-07-16' : '41', '2017-07-17/2017-07-23' : '42',
             '2017-07-24/2017-07-30' : '43', '2017-07-31/2017-08-06' : '44',
             '2017-08-07/2017-08-13' : '45', '2017-08-14/2017-08-20' : '46',
             '2017-08-21/2017-08-27' : '47', '2017-08-28/2017-09-03' : '48',
             '2017-09-04/2017-09-10' : '49', '2017-09-11/2017-09-17' : '50',
             '2017-09-18/2017-09-24' : '51', '2017-09-25/2017-10-01' : '52',
             '2017-10-02/2017-10-08' : '53'}


# In[6]:


def weeks_get_dummies(df_trans_cut, by_variable):
    if by_variable == 'sales_sum':
        week_sums= df_trans_cut.groupby(['client_id', 'week'])[by_variable].sum().to_frame()
        
    elif by_variable == 'chq_date':
        week_sums= df_trans_cut.groupby(['client_id', 'week'])[by_variable].count().to_frame()
        
        
    week_sums.reset_index(drop=False, inplace=True)
    week_sums.week= week_sums.week.astype(str)
    weeks = pd.get_dummies(week_sums.week)
    weeks = weeks.rename(columns= WEEKS_ENCODED)

    weeks =weeks.astype(int)

    week_sums_dummies = pd.merge(week_sums, weeks, on= weeks.index)
    week_sums_dummies =  week_sums_dummies.drop(['week'], axis=1)
    week_sums_dummies =  week_sums_dummies.drop(['key_0'], axis=1)
    
    
    for idx in weeks.columns.values:
        week_sums_dummies[idx] =  week_sums_dummies[idx] *  week_sums_dummies[by_variable]

    week_clients_sums =  week_sums_dummies.groupby('client_id')['01', 
                '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
               '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
               '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33',
               '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44',
               '45', '46', '47', '48', '49', '50', '51', '52', '53'].sum()
    week_clients_sums.reset_index(drop=False, inplace=True)
    return  weeks, week_clients_sums


# In[7]:


def kruskal_sums(weeks, week_clients_sums2):
    pvalues_dict_eight2 = {1: [],
                   2 : [],
                   3 : [],
                   4 : [],
                   5 : [],
                   6 : [],
                   }
    for ind in range(98455):
        ref_data = week_clients_sums2.loc[ind][weeks.columns.values[0*8 : (0+1)*8]]
        ref_zers = ref_data.eq(0,axis =0 ).sum()
        ref_sum_ = ref_data.sum()
        ref_mean_ = ref_data.mean()
        ref_median_ = ref_data.median()
        ref_std_ = ref_data.std()
        ref_var_ = ref_data.var()

        for i in range(1,7):
            com_data = week_clients_sums2.loc[ind][weeks.columns.values[i*8 : (i+1)*8]]

            coeff = -1
            if ref_median_ ==0 :
                if(com_data.mean() >=ref_mean_) | ( ref_mean_ / com_data.mean()<=5):
                        coeff = 1
            else:
                if com_data.median() == 0:
                    if(com_data.mean() >=ref_mean_) | ( ref_mean_ / com_data.mean()<=5):
                        coeff = 1 
                    coeff = -1
                else:
                    if(com_data.median() >=ref_median_) | ( ref_median_ / com_data.median()<=3):
                        coeff = 1
            sum_ = com_data.sum()
            if sum_ == 0:
                pvalues_dict_eight2[i].append(-1)
            else:
                pvalues_dict_eight2[i].append(scipy.stats.kruskal(ref_data,com_data).pvalue * coeff)
    
    pvalues_dataset_eight_sign_3 = pd.DataFrame.from_dict(pvalues_dict_eight2)
    
    pvalues_clients_3 = pd.merge(week_clients_sums2[['client_id']], pvalues_dataset_eight_sign_3, 
                             on =week_clients_sums2.index )
    pvalues_clients_3_cut = pvalues_clients_3[pvalues_dataset_eight_sign_3.median(axis=1) >-0.9]
    pvalues_clients_3_cut = pvalues_clients_3_cut.drop(columns = ['key_0'])
    
    return pvalues_clients_3_cut
    


# In[8]:


def kruskal_chqs(weeks, week_clients_sums):
    pvalues_dict_eight = {1: [],
               2 : [],
               3 : [],
               4 : [],
               5 : [],
               6 : [],
               }
    for ind in range(98455):
        ref_data = week_clients_sums.loc[ind][weeks.columns.values[0*8 : (0+1)*8]]
        ref_zers = ref_data.eq(0,axis =0 ).sum()
        ref_sum_ = ref_data.sum()

        for i in range(1,7):
            com_data = week_clients_sums.loc[ind][weeks.columns.values[i*8 : (i+1)*8]]
            com_zers = com_data.eq(0,axis =0 ).sum()
            sum_ = com_data.sum()
            if sum_ == 0:
                pvalues_dict_eight[i].append(-1)
            else:
                pvalues_dict_eight[i].append(scipy.stats.kruskal(ref_data,com_data).pvalue * 
                                         ( 1 if ref_zers >= com_zers else (-1)))

    pvalues_dataset_eight_sign_2 = pd.DataFrame.from_dict(pvalues_dict_eight)
    pvalues_clients_2 = pd.merge(week_clients_sums[['client_id']], pvalues_dataset_eight_sign_2, 
                             on =week_clients_sums.index )
    pvalues_clients_2  = pvalues_clients_2.drop(columns='key_0')
    pvalues_clients_2_cut  = pvalues_clients_2[pvalues_dataset_eight_sign_2.median(axis=1) > -0.9]
    
    return pvalues_clients_2_cut


# In[9]:


def define_churn(pvalues_clients_3_cut):
    
    pvalues_clients_3_cut['gone_11_13'] =(( pvalues_clients_3_cut[6]<0) & 
                                        ( pvalues_clients_3_cut[5]<0)).astype(int)
    
    pvalues_clients_3_cut['gone_9_10'] =((pvalues_clients_3_cut[4]<0) & 
                                         ( pvalues_clients_3_cut[6]<0) & 
                                         ( pvalues_clients_3_cut[5]<0)).astype(int)
    
    pvalues_clients_3_cut['gone_7_8'] =(( pvalues_clients_3_cut[3] < 0)
                                    & ( pvalues_clients_3_cut[4]<0)
                                    & ( pvalues_clients_3_cut[6]<0)
                                    & ( pvalues_clients_3_cut[5]<0)).astype(int)
    
    pvalues_clients_3_cut['gone_5_6'] =(  ( pvalues_clients_3_cut[2]<0)
                                    & ( pvalues_clients_3_cut[3]<0)
                                    & ( pvalues_clients_3_cut[4]<0)
                                    & ( pvalues_clients_3_cut[6]<0)
                                    & ( pvalues_clients_3_cut[5]<0)).astype(int)
    
    pvalues_clients_3_cut['gone'] =( pvalues_clients_3_cut['gone_9_10'] |
                                pvalues_clients_3_cut['gone_11_13'] |
                                pvalues_clients_3_cut['gone_7_8'] |
                                pvalues_clients_3_cut['gone_5_6'] )
    
    pvalues_clients_3_cut = pvalues_clients_3_cut[[ 'client_id',     
                                               1,   2,   3,  4,  5,  6, 
                                               'gone_5_6', 'gone_7_8', 'gone_9_10', 'gone_11_13', 
                                               'gone']]
    return pvalues_clients_3_cut


# In[1]:


def gone56(pvalues_clients_2_cut, df_trans_cut):
    
    clients_g56 = pvalues_clients_2_cut[pvalues_clients_2_cut.gone_5_6== 1].client_id.values
    df_trans_cut_g56 = df_trans_cut[df_trans_cut.client_id.isin(clients_g56)]
    df_trans_cut_g56  = df_trans_cut_g56[(df_trans_cut_g56.month_encoded == 1) |
                                    (df_trans_cut_g56.month_encoded == 2)|
                                    (df_trans_cut_g56.month_encoded == 3)|
                                    (df_trans_cut_g56.month_encoded == 4)]
    
    df_g56_chq = df_trans_cut_g56.groupby(['client_id'])['chq_id'].nunique().to_frame()
    df_g56_chq.reset_index(drop=False, inplace=True)
    df_g56_chq = df_g56_chq.rename(columns={'chq_id': 'total_num_chqs'})



    df_g56_days = df_trans_cut_g56.groupby(['client_id'])['chq_date'].nunique().to_frame()
    df_g56_days.reset_index(drop=False, inplace=True)
    df_g56_days = df_g56_days.rename(columns={'chq_date': 'total_num_days'})


    df_g56_sum = df_trans_cut_g56.groupby(['client_id'])['sales_sum'].sum().to_frame()
    df_g56_sum.reset_index(drop=False, inplace=True)
    df_g56_sum =df_g56_sum.rename(columns= {'sales_sum': "total_money_spent"})


    df_g56_promo_material = df_trans_cut_g56.groupby(['client_id'])['material'].nunique().to_frame()
    df_g56_promo_material.reset_index(drop=False, inplace=True)
    df_g56_promo_material  = df_g56_promo_material.rename(columns = {'material': 'num_unique_goods'})



    df_trans_cut_g56['total_is_promo'] = df_trans_cut_g56['sales_count']  * df_trans_cut_g56['is_promo'] 
    df_trans_cut_g56['total_is_promo_sum']  =  df_trans_cut_g56['sales_sum'] * (df_trans_cut_g56['total_is_promo'] >0)


    df_g56_promo_sum = df_trans_cut_g56.groupby(['client_id'])['total_is_promo_sum'].sum().to_frame()
    df_g56_promo_sum.reset_index(drop=False, inplace=True)


    df_week_sum = df_trans_cut_g56.groupby(['client_id', 'week_encoded'])['sales_sum'].sum().to_frame()
    df_week_sum.reset_index(drop=False, inplace=True)


    wks = pd.get_dummies(df_week_sum.week_encoded)
    wks = wks.astype(int)
    wks = pd.merge(df_week_sum['client_id'], wks, on=df_week_sum.index)
    wks = wks.drop(columns='key_0')


    for col in wks.columns[1:]:
        wks[col] = wks[col] * df_week_sum['sales_sum']  


    df_month_sum = df_trans_cut_g56.groupby(['client_id', 'month_encoded'])['sales_sum'].sum().to_frame()
    df_month_sum.reset_index(drop=False, inplace=True)
    mnths = pd.get_dummies(df_month_sum.month_encoded)
    mnths = mnths.astype(int)
    mnths = pd.merge(df_month_sum['client_id'], mnths, on=df_month_sum.index)
    mnths = mnths.drop(columns='key_0')



    for col in mnths.columns[1:]:
        mnths[col] = mnths[col] * df_month_sum['sales_sum']
        
    wks = wks.groupby(['client_id'])[ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].sum()
    wks.reset_index(drop=False, inplace=True)

    mnths = mnths.groupby(['client_id'])[1, 2, 3, 4].sum()
    mnths.reset_index(drop=False, inplace=True)

    mnths['trend_coef'] = ((mnths[4] - mnths[1])/mnths[[1, 2, 3, 4]].mean(axis=1))
    mnths['sparsity'] = wks[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]].eq(0, axis=0).sum(axis=1).values/16

    df_g56 = pd.merge(df_g56_chq, df_g56_days, on=['client_id'])
    df_g56 = pd.merge(df_g56, df_g56_promo_material, on=['client_id'])
    df_g56 = pd.merge(df_g56, df_g56_sum, on=['client_id'])
    df_g56 = pd.merge(df_g56, df_g56_promo_sum, on=['client_id'])
    df_g56 = pd.merge(df_g56, mnths[['client_id', 'trend_coef', 'sparsity']], on=['client_id'])
    
    return df_g56


# In[2]:


def gone78(pvalues_clients_2_cut, df_trans_cut):
    clients_g78 = pvalues_clients_2_cut[ (pvalues_clients_2_cut.gone_5_6== 0)
            & (pvalues_clients_2_cut.gone_7_8== 1)].client_id.values
    df_trans_cut_g78 = df_trans_cut[df_trans_cut.client_id.isin(clients_g78)]
    df_trans_cut_g78  = df_trans_cut_g78[(df_trans_cut_g78.month_encoded == 3) |
                                    (df_trans_cut_g78.month_encoded == 4)|
                                    (df_trans_cut_g78.month_encoded == 5)|
                                    (df_trans_cut_g78.month_encoded == 6)]
    
    df_g78_chq = df_trans_cut_g78.groupby(['client_id'])['chq_id'].nunique().to_frame()
    df_g78_chq.reset_index(drop=False, inplace=True)
    df_g78_chq = df_g78_chq.rename(columns={'chq_id': 'total_num_chqs'})



    df_g78_days = df_trans_cut_g78.groupby(['client_id'])['chq_date'].nunique().to_frame()
    df_g78_days.reset_index(drop=False, inplace=True)
    df_g78_days = df_g78_days.rename(columns={'chq_date': 'total_num_days'})


    df_g78_sum = df_trans_cut_g78.groupby(['client_id'])['sales_sum'].sum().to_frame()
    df_g78_sum.reset_index(drop=False, inplace=True)
    df_g78_sum =df_g78_sum.rename(columns= {'sales_sum': "total_money_spent"})


    df_g78_promo_material = df_trans_cut_g78.groupby(['client_id'])['material'].nunique().to_frame()
    df_g78_promo_material.reset_index(drop=False, inplace=True)
    df_g78_promo_material  = df_g78_promo_material.rename(columns = {'material': 'num_unique_goods'})



    df_trans_cut_g78['total_is_promo'] = df_trans_cut_g78['sales_count']  * df_trans_cut_g78['is_promo'] 
    df_trans_cut_g78['total_is_promo_sum']  =  df_trans_cut_g78['sales_sum'] * (df_trans_cut_g78['total_is_promo'] >0)


    df_g78_promo_sum = df_trans_cut_g78.groupby(['client_id'])['total_is_promo_sum'].sum().to_frame()
    df_g78_promo_sum.reset_index(drop=False, inplace=True)


    df_week_sum = df_trans_cut_g78.groupby(['client_id', 'week_encoded'])['sales_sum'].sum().to_frame()
    df_week_sum.reset_index(drop=False, inplace=True)


    wks = pd.get_dummies(df_week_sum.week_encoded)
    wks = wks.astype(int)
    wks = pd.merge(df_week_sum['client_id'], wks, on=df_week_sum.index)
    wks = wks.drop(columns='key_0')


    for col in wks.columns[1:]:
        wks[col] = wks[col] * df_week_sum['sales_sum']  


    df_month_sum = df_trans_cut_g78.groupby(['client_id', 'month_encoded'])['sales_sum'].sum().to_frame()
    df_month_sum.reset_index(drop=False, inplace=True)
    mnths = pd.get_dummies(df_month_sum.month_encoded)
    mnths = mnths.astype(int)
    mnths = pd.merge(df_month_sum['client_id'], mnths, on=df_month_sum.index)
    mnths = mnths.drop(columns='key_0')



    for col in mnths.columns[1:]:
        mnths[col] = mnths[col] * df_month_sum['sales_sum']
        
    wks = wks.groupby(['client_id'])[ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24].sum()
    wks.reset_index(drop=False, inplace=True)


    mnths = mnths.groupby(['client_id'])[ 3, 4, 5,6].sum()
    mnths.reset_index(drop=False, inplace=True)



    mnths['trend_coef'] = ((mnths[5] - mnths[3])/mnths[[3, 4, 5,6]].mean(axis=1))
    mnths['sparsity'] = wks[[ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]].eq(0, axis=0).sum(axis=1).values/16

    df_g78 = pd.merge(df_g78_chq, df_g78_days, on=['client_id'])
    df_g78 = pd.merge(df_g78, df_g78_promo_material, on=['client_id'])
    df_g78 = pd.merge(df_g78, df_g78_sum, on=['client_id'])
    df_g78 = pd.merge(df_g78, df_g78_promo_sum, on=['client_id'])
    df_g78 = pd.merge(df_g78, mnths[['client_id', 'trend_coef', 'sparsity']], on=['client_id'])

    return df_g78


# In[3]:


def gone910(pvalues_clients_2_cut, df_trans_cut):
    clients_g910 = pvalues_clients_2_cut[ (pvalues_clients_2_cut.gone_5_6== 0)
                                     & (pvalues_clients_2_cut.gone_7_8== 0)
                                     & (pvalues_clients_2_cut.gone_9_10== 1)].client_id.values
    
    df_trans_cut_g910 = df_trans_cut[df_trans_cut.client_id.isin(clients_g910)]
    df_trans_cut_g910  = df_trans_cut_g910[(df_trans_cut_g910.month_encoded == 5) |
                                    (df_trans_cut_g910.month_encoded == 6)|
                                    (df_trans_cut_g910.month_encoded == 7)|
                                    (df_trans_cut_g910.month_encoded == 8)]
    
    df_g910_chq = df_trans_cut_g910.groupby(['client_id'])['chq_id'].nunique().to_frame()
    df_g910_chq.reset_index(drop=False, inplace=True)
    df_g910_chq = df_g910_chq.rename(columns={'chq_id': 'total_num_chqs'})



    df_g910_days = df_trans_cut_g910.groupby(['client_id'])['chq_date'].nunique().to_frame()
    df_g910_days.reset_index(drop=False, inplace=True)
    df_g910_days = df_g910_days.rename(columns={'chq_date': 'total_num_days'})


    df_g910_sum = df_trans_cut_g910.groupby(['client_id'])['sales_sum'].sum().to_frame()
    df_g910_sum.reset_index(drop=False, inplace=True)
    df_g910_sum =df_g910_sum.rename(columns= {'sales_sum': "total_money_spent"})


    df_g910_promo_material = df_trans_cut_g910.groupby(['client_id'])['material'].nunique().to_frame()
    df_g910_promo_material.reset_index(drop=False, inplace=True)
    df_g910_promo_material  = df_g910_promo_material.rename(columns = {'material': 'num_unique_goods'})



    df_trans_cut_g910['total_is_promo'] = df_trans_cut_g910['sales_count']  * df_trans_cut_g910['is_promo'] 
    df_trans_cut_g910['total_is_promo_sum']  =  df_trans_cut_g910['sales_sum'] * (df_trans_cut_g910['total_is_promo'] >0)


    df_g910_promo_sum = df_trans_cut_g910.groupby(['client_id'])['total_is_promo_sum'].sum().to_frame()
    df_g910_promo_sum.reset_index(drop=False, inplace=True)


    df_week_sum = df_trans_cut_g910.groupby(['client_id', 'week_encoded'])['sales_sum'].sum().to_frame()
    df_week_sum.reset_index(drop=False, inplace=True)


    wks = pd.get_dummies(df_week_sum.week_encoded)
    wks = wks.astype(int)
    wks = pd.merge(df_week_sum['client_id'], wks, on=df_week_sum.index)
    wks = wks.drop(columns='key_0')


    for col in wks.columns[1:]:
        wks[col] = wks[col] * df_week_sum['sales_sum']  


    df_month_sum = df_trans_cut_g910.groupby(['client_id', 'month_encoded'])['sales_sum'].sum().to_frame()
    df_month_sum.reset_index(drop=False, inplace=True)
    mnths = pd.get_dummies(df_month_sum.month_encoded)
    mnths = mnths.astype(int)
    mnths = pd.merge(df_month_sum['client_id'], mnths, on=df_month_sum.index)
    mnths = mnths.drop(columns='key_0')



    for col in mnths.columns[1:]:
        mnths[col] = mnths[col] * df_month_sum['sales_sum']
        
    wks = wks.groupby(['client_id'])[ 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32].sum()
    wks.reset_index(drop=False, inplace=True)


    mnths = mnths.groupby(['client_id'])[  5,6, 7, 8].sum()
    mnths.reset_index(drop=False, inplace=True)



    mnths['trend_coef'] = ((mnths[8] - mnths[5])/mnths[[5,6 , 7 ,8]].mean(axis=1))
    mnths['sparsity'] = wks[[ 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]].eq(0, axis=0).sum(axis=1).values/16


    df_g910 = pd.merge(df_g910_chq, df_g910_days, on=['client_id'])
    df_g910 = pd.merge(df_g910, df_g910_promo_material, on=['client_id'])
    df_g910 = pd.merge(df_g910, df_g910_sum, on=['client_id'])
    df_g910 = pd.merge(df_g910, df_g910_promo_sum, on=['client_id'])
    df_g910 = pd.merge(df_g910, mnths[['client_id', 'trend_coef', 'sparsity']], on=['client_id'])
    
    return df_g910


# In[4]:


def gone1113(pvalues_clients_2_cut, df_trans_cut):
    clients_g1113= pvalues_clients_2_cut[ (pvalues_clients_2_cut.gone_5_6== 0)
                                     & (pvalues_clients_2_cut.gone_7_8== 0)
                                     & (pvalues_clients_2_cut.gone_9_10== 0)
                                     & (pvalues_clients_2_cut.gone_11_13== 1)].client_id.values
    df_trans_cut_g1113 = df_trans_cut[df_trans_cut.client_id.isin(clients_g1113)]
    df_trans_cut_g1113  = df_trans_cut_g1113[(df_trans_cut_g1113.month_encoded == 7) |
                                    (df_trans_cut_g1113.month_encoded == 8)|
                                    (df_trans_cut_g1113.month_encoded == 9)|
                                    (df_trans_cut_g1113.month_encoded == 10)]
    
    df_g1113_chq = df_trans_cut_g1113.groupby(['client_id'])['chq_id'].nunique().to_frame()
    df_g1113_chq.reset_index(drop=False, inplace=True)
    df_g1113_chq = df_g1113_chq.rename(columns={'chq_id': 'total_num_chqs'})



    df_g1113_days = df_trans_cut_g1113.groupby(['client_id'])['chq_date'].nunique().to_frame()
    df_g1113_days.reset_index(drop=False, inplace=True)
    df_g1113_days = df_g1113_days.rename(columns={'chq_date': 'total_num_days'})


    df_g1113_sum = df_trans_cut_g1113.groupby(['client_id'])['sales_sum'].sum().to_frame()
    df_g1113_sum.reset_index(drop=False, inplace=True)
    df_g1113_sum =df_g1113_sum.rename(columns= {'sales_sum': "total_money_spent"})


    df_g1113_promo_material = df_trans_cut_g1113.groupby(['client_id'])['material'].nunique().to_frame()
    df_g1113_promo_material.reset_index(drop=False, inplace=True)
    df_g1113_promo_material  = df_g1113_promo_material.rename(columns = {'material': 'num_unique_goods'})



    df_trans_cut_g1113['total_is_promo'] = df_trans_cut_g1113['sales_count']  * df_trans_cut_g1113['is_promo'] 
    df_trans_cut_g1113['total_is_promo_sum']  =  df_trans_cut_g1113['sales_sum'] * (df_trans_cut_g1113['total_is_promo'] >0)


    df_g1113_promo_sum = df_trans_cut_g1113.groupby(['client_id'])['total_is_promo_sum'].sum().to_frame()
    df_g1113_promo_sum.reset_index(drop=False, inplace=True)


    df_week_sum = df_trans_cut_g1113.groupby(['client_id', 'week_encoded'])['sales_sum'].sum().to_frame()
    df_week_sum.reset_index(drop=False, inplace=True)


    wks = pd.get_dummies(df_week_sum.week_encoded)
    wks = wks.astype(int)
    wks = pd.merge(df_week_sum['client_id'], wks, on=df_week_sum.index)
    wks = wks.drop(columns='key_0')


    for col in wks.columns[1:]:
        wks[col] = wks[col] * df_week_sum['sales_sum']  


    df_month_sum = df_trans_cut_g1113.groupby(['client_id', 'month_encoded'])['sales_sum'].sum().to_frame()
    df_month_sum.reset_index(drop=False, inplace=True)
    mnths = pd.get_dummies(df_month_sum.month_encoded)
    mnths = mnths.astype(int)
    mnths = pd.merge(df_month_sum['client_id'], mnths, on=df_month_sum.index)
    mnths = mnths.drop(columns='key_0')



    for col in mnths.columns[1:]:
        mnths[col] = mnths[col] * df_month_sum['sales_sum']
        
    wks = wks.groupby(['client_id'])[ 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40].sum()
    wks.reset_index(drop=False, inplace=True)


    mnths = mnths.groupby(['client_id'])[  7, 8, 9, 10].sum()
    mnths.reset_index(drop=False, inplace=True)



    mnths['trend_coef'] = ((mnths[10] - mnths[7])/mnths[[7, 8, 9, 10]].mean(axis=1))
    mnths['sparsity'] = wks[[ 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]].eq(0, axis=0).sum(axis=1).values/16

    df_g1113 = pd.merge(df_g1113_chq, df_g1113_days, on=['client_id'])
    df_g1113 = pd.merge(df_g1113, df_g1113_promo_material, on=['client_id'])
    df_g1113 = pd.merge(df_g1113, df_g1113_sum, on=['client_id'])
    df_g1113 = pd.merge(df_g1113, df_g1113_promo_sum, on=['client_id'])
    df_g1113 = pd.merge(df_g1113, mnths[['client_id', 'trend_coef', 'sparsity']], on=['client_id'])
    
    return df_g1113


# In[5]:


def gone_stayed(pvalues_clients_2_cut, df_trans_cut):
    clients_g56 = pvalues_clients_2_cut[pvalues_clients_2_cut.gone_5_6== 1].client_id.values
    clients_g78 = pvalues_clients_2_cut[ (pvalues_clients_2_cut.gone_5_6== 0)
            & (pvalues_clients_2_cut.gone_7_8== 1)].client_id.values
    clients_g910 = pvalues_clients_2_cut[ (pvalues_clients_2_cut.gone_5_6== 0)
                                     & (pvalues_clients_2_cut.gone_7_8== 0)
                                     & (pvalues_clients_2_cut.gone_9_10== 1)].client_id.values
    clients_g1113= pvalues_clients_2_cut[ (pvalues_clients_2_cut.gone_5_6== 0)
                                     & (pvalues_clients_2_cut.gone_7_8== 0)
                                     & (pvalues_clients_2_cut.gone_9_10== 0)
                                     & (pvalues_clients_2_cut.gone_11_13== 1)].client_id.values
    
    clients_stayed = np.array(list(set(pvalues_clients_3_cut.client_id.values) - (set(clients_g56) | 
                                                       set(clients_g78) |
                                                       set(clients_g910) | 
                                                       set(clients_g1113))))
    
    df_trans_cut_stayed= df_trans_cut[df_trans_cut.client_id.isin(clients_stayed)]
    df_trans_cut_stayed1  = df_trans_cut_stayed[(df_trans_cut_stayed.month_encoded == 7) |
                                    (df_trans_cut_stayed.month_encoded == 8)|
                                    (df_trans_cut_stayed.month_encoded == 9)|
                                    (df_trans_cut_stayed.month_encoded == 10)]
    
    df_stayed_chq = df_trans_cut_stayed1.groupby(['client_id'])['chq_id'].nunique().to_frame()
    df_stayed_chq.reset_index(drop=False, inplace=True)
    df_stayed_chq = df_stayed_chq.rename(columns={'chq_id': 'total_num_chqs'})



    df_stayed_days = df_trans_cut_stayed1.groupby(['client_id'])['chq_date'].nunique().to_frame()
    df_stayed_days.reset_index(drop=False, inplace=True)
    df_stayed_days = df_stayed_days.rename(columns={'chq_date': 'total_num_days'})


    df_stayed_sum = df_trans_cut_stayed1.groupby(['client_id'])['sales_sum'].sum().to_frame()
    df_stayed_sum.reset_index(drop=False, inplace=True)
    df_stayed_sum =df_stayed_sum.rename(columns= {'sales_sum': "total_money_spent"})


    df_stayed_promo_material = df_trans_cut_stayed1.groupby(['client_id'])['material'].nunique().to_frame()
    df_stayed_promo_material.reset_index(drop=False, inplace=True)
    df_stayed_promo_material  = df_stayed_promo_material.rename(columns = {'material': 'num_unique_goods'})



    df_trans_cut_stayed1['total_is_promo'] = df_trans_cut_stayed1['sales_count']  * df_trans_cut_stayed1['is_promo'] 
    df_trans_cut_stayed1['total_is_promo_sum']  =  df_trans_cut_stayed1['sales_sum'] * (df_trans_cut_stayed1['total_is_promo'] >0)


    df_stayed_promo_sum = df_trans_cut_stayed1.groupby(['client_id'])['total_is_promo_sum'].sum().to_frame()
    df_stayed_promo_sum.reset_index(drop=False, inplace=True)


    df_week_sum = df_trans_cut_stayed1.groupby(['client_id', 'week_encoded'])['sales_sum'].sum().to_frame()
    df_week_sum.reset_index(drop=False, inplace=True)


    wks = pd.get_dummies(df_week_sum.week_encoded)
    wks = wks.astype(int)
    wks = pd.merge(df_week_sum['client_id'], wks, on=df_week_sum.index)
    wks = wks.drop(columns='key_0')


    for col in wks.columns[1:]:
        wks[col] = wks[col] * df_week_sum['sales_sum']  


    df_month_sum = df_trans_cut_stayed1.groupby(['client_id', 'month_encoded'])['sales_sum'].sum().to_frame()
    df_month_sum.reset_index(drop=False, inplace=True)
    mnths = pd.get_dummies(df_month_sum.month_encoded)
    mnths = mnths.astype(int)
    mnths = pd.merge(df_month_sum['client_id'], mnths, on=df_month_sum.index)
    mnths = mnths.drop(columns='key_0')



    for col in mnths.columns[1:]:
        mnths[col] = mnths[col] * df_month_sum['sales_sum']
    
    wks = wks.groupby(['client_id'])[ 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40].sum()
    wks.reset_index(drop=False, inplace=True)


    mnths = mnths.groupby(['client_id'])[  7, 8, 9, 10].sum()
    mnths.reset_index(drop=False, inplace=True)



    mnths['trend_coef'] = ((mnths[10] - mnths[7])/mnths[[7, 8, 9, 10]].mean(axis=1))
    mnths['sparsity'] = wks[[ 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]].eq(0, axis=0).sum(axis=1).values/16


    df_stayed = pd.merge(df_stayed_chq, df_stayed_days, on=['client_id'])
    df_stayed = pd.merge(df_stayed, df_stayed_promo_material, on=['client_id'])
    df_stayed = pd.merge(df_stayed, df_stayed_sum, on=['client_id'])
    df_stayed = pd.merge(df_stayed, df_stayed_promo_sum, on=['client_id'])
    df_stayed = pd.merge(df_stayed, mnths[['client_id', 'trend_coef', 'sparsity']], on=['client_id'])
    
    return df_stayed


# In[ ]:





# In[ ]:




