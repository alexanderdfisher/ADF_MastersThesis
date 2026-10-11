import pandas as pd
from datetime import datetime
import math
from scipy.stats import linregress, gmean
import numpy as np
from scipy.optimize import fsolve, newton, brentq
from scipy.optimize import root_scalar
import psycopg2
from psycopg2.extras import execute_values
import multiprocessing
import warnings
import os
import traceback
import csv
import matplotlib.pyplot as plt
import tabulate as tabulate
import matplotlib.dates as mdates
import egz_model as egz

us_30y_fp = r"C:\Users\Fisher\Desktop\Thesis\Main Data\mktyld_us_10y.txt"
us30y_df = pd.read_csv(us_30y_fp, na_values='.')
us30y_df['DATE'] = pd.to_datetime(us30y_df['DATE'], format='%Y-%m-%d')
us30y_df['YIELD'] = us30y_df['YIELD'] / 100

# for every key in dict and every gvkey (value) associated with the key find the value of c that makes the firm mkt lev equal to twosic industry market lev, we end up with proper value for e_xi5 and d_xi5
sp500_dict_fp = r"C:\Users\Fisher\Desktop\Thesis\Main Data\sp500_constituents.xlsx"
sp500_dict_df = pd.read_excel(sp500_dict_fp, 'constituents', na_values=['NaN', 'nan', 'Nan', '', 'NULL', '#N/A'])
sp500_dict = sp500_dict_df.to_dict(orient='list')
# df verison
constituents_fp = r"C:\Users\Fisher\Desktop\Thesis\Main Data\Sample\sp500_constituents_values.xlsx"
df_constituents = pd.read_excel(constituents_fp, 'constituents', na_values=['NaN', 'nan', 'Nan', '', 'NULL', '#N/A'])

#prime data used of short term interest rate
#prime data in whole number format must /100
prime_fp = r"C:\Users\Fisher\Desktop\Thesis\Main Data\USPRIME_DATA.csv"
usprime_df = pd.read_csv(prime_fp, header=0, na_values=['.'],converters={'DPRIME': lambda x: float(x) if x != '.' else np.nan})
usprime_df['DATE'] = pd.to_datetime(usprime_df['DATE'])


#rf data & functions
rf_fp = r"C:\Users\Fisher\Desktop\Thesis\Main Data\Rf_SP500ret_annual.xlsx"
df_rf = pd.read_excel(rf_fp)
df_rf['Date'] = pd.to_datetime(df_rf['Date'], format='%Y%m%d')
def rf_frm(key):
    # Adjust date
    year_adj = key // 10000
    date_adj = datetime(year_adj, 1, 1)
    rf = df_rf[(df_rf['Date'] == date_adj)]['rf'].iloc[0] / 100
    return rf
def rf_prc(gvkey, year):
    # Adjust date
    year_adj = year // 10000
    date_adj = datetime(year_adj, 1, 1)

    # Grab risk-free
    rf = df_rf[(df_rf['Date'] == date_adj)]['rf'].iloc[0] / 100

    df_obs_prc = df_dsf[(df_dsf['DATE'] <= datetime(year_adj, 12, 31)) & (df_dsf['GVKEY'] == gvkey)].sort_values(by='DATE', ascending=False)

    obs_prc = df_obs_prc['PRC'].iloc[0]

    return rf, obs_prc
# Bond equivalent yield data
bond_fp = r"C:\Users\Fisher\Desktop\Thesis\Main Data\CORP_BOND_DTA.xlsx"
aaa_df = pd.read_excel(bond_fp, sheet_name='AAA', parse_dates=['DATE'], na_values=['NaN', '', '#N/A'])
baa_df = pd.read_excel(bond_fp, sheet_name='BAA', parse_dates=['DATE'], na_values=['NaN', '', '#N/A'])
aaa_df['DATE'] = pd.to_datetime(aaa_df['DATE'])
baa_df['DATE'] = pd.to_datetime(baa_df['DATE'])

aaa_baa_spread_fp = r"C:\Users\Fisher\Desktop\Thesis\Main Data\AAA_BAA_-10Y.xlsx"
aaa_baa_spread_df = pd.read_excel(aaa_baa_spread_fp, sheet_name='Combined', parse_dates=['observation_date'], na_values=['NaN', '', '#N/A'])
pd.to_datetime(aaa_baa_spread_df['observation_date'], format='%Y-%m-%d')

OAS_fp = r"C:\Users\Fisher\Desktop\Thesis\Main Data\OAS_1996_2022.xlsx"
OAS_df = pd.read_excel(OAS_fp, sheet_name='Combined', parse_dates=['observation_date'], na_values=['NaN', '', '#N/A'])
pd.to_datetime(OAS_df['observation_date'], format='%Y-%m-%d')

# File paths for the Excel files
file_dbaa = r"C:\Users\Fisher\Desktop\Thesis\Main Data\DBAA.xlsx"  # Replace with the actual path to your DBAA file
file_daaa = r"C:\Users\Fisher\Desktop\Thesis\Main Data\DAAA.xlsx" # Replace with the actual path to your DAAA file

# Load the 'data' sheet from each Excel file into separate DataFrames
df_dbaa = pd.read_excel(file_dbaa, sheet_name='data')
df_daaa = pd.read_excel(file_daaa, sheet_name='data')

# Convert the 'DATE' columns in both DataFrames to pandas datetime
df_dbaa['DATE'] = pd.to_datetime(df_dbaa['DATE'], errors='coerce', format='%Y-%m-%d')
df_daaa['DATE'] = pd.to_datetime(df_daaa['DATE'], errors='coerce', format='%Y-%m-%d')

# Clean null values: Remove rows with any null values in both DataFrames
df_dbaa = df_dbaa.dropna()  # Drops rows where any column has a NaN
df_dbaa = df_daaa.dropna()  # Drops rows where any column has a NaN

# CHH variables and calculation to follow
chs_fp = r"C:\Users\Fisher\Desktop\Thesis\Main Data\all_chs_vars.csv"
df_chs = pd.read_csv(chs_fp, names=['DATE', 'GVKEY', 'TWOSIC', 'SIGMA', 'RSIZE', 'EXRET', 'NIMTA', 'TLMTA', 'MB', 'CASHMTA', 'PRICE'], na_values=['', 'Nan', 'NaN', 'nan', 'null', 'NULL', float('inf'), -float('inf'), 'inf', '-inf'])
df_chs.dropna(inplace=True)
df_chs['DATE'] = pd.to_datetime(df_chs['DATE'], format='%Y-%m-%d')

# Define intercept and coefficients
intercept_chs = -9.16
nimta_constant = 20.26
tlmta_constant = 1.42
exret_constant = -7.13
sigma_constant = 1.41
rsize_constant = 0.045
cashmta_constant = -2.13
mb_constant = 0.075
price_constant = 0.058

# Calculate CHS score for each row

df_chs['CHS'] = intercept_chs \
                - (nimta_constant * df_chs['NIMTA']) \
                + (tlmta_constant * df_chs['TLMTA']) \
                + (exret_constant * df_chs['EXRET']) \
                + (sigma_constant * df_chs['SIGMA']) \
                - (rsize_constant * df_chs['RSIZE']) \
                + (cashmta_constant * df_chs['CASHMTA']) \
                + (mb_constant * df_chs['MB']) \
                - (price_constant * df_chs['PRICE'])

# Applies pd.qcut function to the 'CHS' column within each group, dividing the data into quintiles (5 bins) and assigning labels 0 to 4.
# The resulting quintile labels are stored in a new column named 'QUINTILE' in the DataFrame.
# quintiles are 0,1,2,3,4
df_chs['QUINTILE'] = df_chs.groupby(['TWOSIC', 'DATE'])['CHS'].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))

# Capex and Deprec data
csrdsr_fp = r"C:\Users\Fisher\Desktop\Thesis\Main Data\All\ALL_FIRMS_CSR_DSR.csv"
df_csrdsr = pd.read_csv(csrdsr_fp, header=0)
df_csrdsr['datadate'] = pd.to_datetime(df_csrdsr['datadate'], format='%Y%m%d')
df_csrdsr['csr_pit'] = df_csrdsr['capx'] / df_csrdsr['sale']
df_csrdsr['dsr_pit'] = df_csrdsr['dp'] / df_csrdsr['sale']
# Sort df_csrdsr by 'year' 'gvkey'
df_csrdsr.sort_values(by=['year', 'gvkey'], inplace=True)
df_csrdsr.dropna(inplace=True)

# Sales Data
saleq_fp = r"C:\Users\Fisher\Desktop\Thesis\Main Data\Sample\Qtrly_Sales_COMP.csv"
df_saleq = pd.read_csv(saleq_fp, header=0)
df_saleq['datadate'] = pd.to_datetime(df_saleq['datadate'], format='%Y%m%d')
# Convert and create twosic
df_saleq.dropna(inplace=True)
df_saleq['sic'] = df_saleq['sic'].astype(str)
df_saleq['twosic'] = df_saleq['sic'].str[:2]  # Extract left two characters
df_saleq['twosic'] = df_saleq['twosic'].astype(int) # Convert back to integer
def generate_fiscal_date(df):
    fiscal_dates = []
    for index, row in df.iterrows():
        year = str(row['fiscal_yr'])
        quarter = str(row['fiscal_qtr'])

        if quarter == '1.0':
            fiscal_date = year + '0331'  # March 31
        elif quarter == '2.0':
            fiscal_date = year + '0630'  # June 30
        elif quarter == '3.0':
            fiscal_date = year + '0930'  # September 30
        elif quarter == '4.0':
            fiscal_date = year + '1231'  # December 31
        else:
            fiscal_date = None

        fiscal_dates.append(fiscal_date)

    return fiscal_dates

# Apply the function to create the 'fiscaldate' column
df_saleq['fiscaldate'] = generate_fiscal_date(df_saleq)

# Convert 'fiscaldate' column to datetime
df_saleq['fiscaldate'] = pd.to_datetime(df_saleq['fiscaldate'], format='%Y%m%d')
df_saleq.sort_values(by=['gvkey','fiscaldate'], inplace=True)

df_saleq['saleq_pct_change'] = df_saleq.groupby('gvkey')['saleq'].pct_change()

# Annual fundamental data
fundy_fp = r"C:\Users\Fisher\Desktop\Thesis\Main Data\Sample\fundamentals_pull_csv.csv"
df_fundy = pd.read_csv(fundy_fp, header=0, na_values=['','nan', 'Nan'])
df_fundy['datadate'] = pd.to_datetime(df_fundy['datadate'], format='%Y%m%d')
df_fundy['twosic'] = df_fundy['sic'].astype(str).str[:2].astype(int)

# Market leverage data
mktlev_fp = r"C:\Users\Fisher\Desktop\Thesis\Main Data\All\CCM_MKT_LEV.xlsx"
df_mktlev = pd.read_excel(mktlev_fp, na_values=['', 'na', 'NaN', '#N/A'])
df_mktlev = df_mktlev[df_mktlev['MktLev'] != 0]  # Drops rows where 'mktlev' is 0
df_mktlev = df_mktlev.dropna(subset=['MktLev'])

# Daily stock file data (prices)
dsf_fp = r"C:\Users\Fisher\Desktop\Thesis\Main Data\All\dailyCRSP_CHSdta.csv"
df_dsf = pd.read_csv(dsf_fp, na_values=['C', 'B'], header=0)
df_dsf['DATE'] = pd.to_datetime(df_dsf['DATE'], format='%Y%m%d')
df_dsf['PRC'] = df_dsf['PRC'].abs()
# Add 2-digit SIC code column
df_dsf['SIC2'] = df_dsf['HSICCD'].astype(str).str.zfill(4).str[:2]
df_dsf['SIC2'] = df_dsf['SIC2'].astype('int64')
df_dsf['MKT_CAP'] = df_dsf['PRC'] * df_dsf['SHROUT']

# Function for finding gross margin ratio
def x_gm(gvkey, year):
    # Adjust to match the sp500_dict key (last day of trading) to the date in df_csrdsr (last day of year i.e. dec 31)
    # Extract year and format as datetime
    try:
        end_adj = year // 10000
        end_date = datetime(end_adj, 12, 31)

        # Define x_gm
        x_gm_filt = df_fundy[(df_fundy['gvkey'] == gvkey) & (df_fundy['fyear'] == end_adj)]

        sales = x_gm_filt['sale'].iloc[0]
        cogs = x_gm_filt['cogs'].iloc[0]
        gm = (sales - cogs)
        xgm = gm / sales
        y_0 = np.log(gm)
    except IndexError:
        end_adj = year // 10000
        end_date = datetime(end_adj, 12, 31)

        # Define x_gm
        x_gm_filt = df_fundy[(df_fundy['gvkey'] == gvkey) & (df_fundy['datadate'] <= end_date)].sort_values(
            by='datadate', ascending=False)

        sales = x_gm_filt['sale'].iloc[0]
        cogs = x_gm_filt['cogs'].iloc[0]
        gm = (sales - cogs)
        xgm = gm / sales
        y_0 = np.log(gm)
    return xgm, gm, y_0
# GBM Function
def simulate_xi(dt, years, gm, mu_q, sigma):
    find_n = years / dt  # steps
    N = int(find_n)

    for _ in range(N):
        dW_t = np.random.normal(0, np.sqrt(dt))
        dx_i = mu_q * dt * gm + sigma * dW_t * gm
        gm += dx_i

    return gm
# plus GBM function which generates 50 different simulated GMs to populate the grid/tree
def gm_5_list_create(mu_q, sigma, gm):
    dt = 0.1
    years = 5
    N = int(years / dt)
    gm_5_list = []

    while len(gm_5_list) < 51:
        gm_temp = gm
        for _ in range(N):
            dW_t = np.random.normal(0, np.sqrt(dt))
            dx_i = (mu_q * dt + sigma * dW_t) * gm_temp

            gm_temp += dx_i
        if gm > 0:
            gm_5_list.append(gm)

    return gm_5_list
# Annualized quarterly volatility
def sigma_frm(gvkey, year, twosic):
    # Need gvkey and year to define sigma
    year_adj = year // 10000
    # include 4 qtrs from current and previous year (8 total)
    start_date = datetime(year_adj, 12, 31)
    end_date = datetime((year_adj - 2), 12, 31)

    # Filters for sigma
    filt_twosic = df_saleq['twosic'] == twosic

    # Filter for if no nulls
    df_sigma = df_saleq[(df_saleq['gvkey'] == gvkey) & (df_saleq['fiscaldate'] <= start_date) & (
                df_saleq['fiscaldate'] >= end_date)].sort_values(by='fiscaldate', ascending=False)

    # Filter for if nulls
    df_substitute = df_saleq[(df_saleq['fiscaldate'] <= start_date) & (filt_twosic)].sort_values(by='fiscaldate',
                                                                                                 ascending=False)

    # Checks if there are nulls in the data used to calc sigma,
    # If quarterly sales values are not available in COMPUSTAT,
    # we use the average quarterly volatility of sales of the firms
    # in the same 2-digit SIC industry over the last eight-quarter period.

    if df_sigma['saleq'].head(8).isnull().any():
        std_by_group = df_substitute.groupby('gvkey')['saleq_pct_change'].head(8).std(skipna=True)
        sigma = std_by_group.mean() * (np.sqrt(4))
    else:
        sigma = df_sigma['saleq_pct_change'].head(8).std() * (np.sqrt(4))

    return sigma

def fixed_costs(gvkey, year):


    # Adjust date
    year_adj = year // 10000
    df_sga = df_fundy[(df_fundy['fyear'] == year_adj) & (df_fundy['gvkey'] == gvkey)]
    # for f is a % of sales
    try:
        f_ratio = df_sga['xsga'].iloc[0] / df_sga['sale'].iloc[0]
        f = df_sga['xsga'].iloc[0]
    except:
        f = '.'
        f_ratio = 0

    #try closest available year's data
    try:
        if f == '.' or f == 'nan' or f_ratio == 0 or f_ratio == 'nan' or pd.isna(f) or pd.isna(f_ratio):
            date_adj = datetime(year_adj, 12, 31)
            df_sga = df_fundy[(df_fundy['datadate'] <= date_adj) & (df_fundy['gvkey'] == gvkey)].sort_values(by='datadate', ascending=False)
            f_ratio = df_sga['xsga'].iloc[0] / df_sga['sale'].iloc[0]
            f = df_sga['xsga'].iloc[0]
    except:
        f = None
        f_ratio = None

    return f, f_ratio
def csr_dsr(twosic, year):

    # create new avg columns, this means each years avg_csr or avg_dsr represents the avg of the last
    df_csrdsr['avg_csr'] = df_csrdsr.groupby('gvkey')['csr_pit'].rolling(window=3, min_periods=3).mean().reset_index(level=0, drop=True)
    df_csrdsr['avg_dsr'] = df_csrdsr.groupby('gvkey')['dsr_pit'].rolling(window=3, min_periods=3).mean().reset_index(level=0, drop=True)

    # Convert and create twosic
    df_csrdsr['sic'] = df_csrdsr['sic'].astype(str)
    df_csrdsr['twosic'] = df_csrdsr['sic'].str[:2]  # Extract left two characters
    df_csrdsr['twosic'] = df_csrdsr['twosic'].astype(int) # Convert back to integer

    # adjust dict key
    year_adj = pd.to_datetime(year, format='%Y%m%d').year
    df_csrdsr_ind_filt = df_csrdsr[(df_csrdsr['twosic'] == twosic) & (df_csrdsr['year'] == year_adj)]

    avg_ind_csr = df_csrdsr_ind_filt['avg_csr'].mean(skipna=True)
    avg_ind_dsr = df_csrdsr_ind_filt['avg_dsr'].mean(skipna=True)

    return avg_ind_csr, avg_ind_dsr
def find_ind_mkt_lev(sp500_key, twosic):

    # Grab twosic and make coupon rate calc
    date_in_dt = pd.to_datetime(sp500_key, format='%Y%m%d')
    year = date_in_dt.year

    # Find industry avg leverage based on last 3 years(sic-year intersection) to get debt amt
    df_filt_mktlev = df_mktlev[(df_mktlev['FiscalYr'] == year) & (df_mktlev['twosic'] == twosic)]
    avg_ind_mktlev = df_filt_mktlev['MktLev'].mean(skipna=True)

    return avg_ind_mktlev
# short term interest is assumed as prime rate
def st_i(key):

    year_fd = key // 10000
    fiscal_date = datetime(year_fd, 12, 31)
    prime_filt = usprime_df[usprime_df['DATE'] <= fiscal_date].sort_values(by='DATE', ascending=False)
    sti = prime_filt['DPRIME'].iloc[0] / 100

    return sti
def std_ltd_i(gvkey, key):


    try:
        date = str(key)
        year = int(date[:4])

        debt_filter = df_fundy[(df_fundy['gvkey'] == gvkey) & (df_fundy['fyear'] == year)]
        ltd = debt_filter['dltt'].iloc[0]
        std = debt_filter['dlc'].iloc[0]


    except IndexError:
        date = pd.to_datetime(key, format='%Y%m%d')
        year = date.year
        date_adj = datetime(year, 12, 31)
        debt_filter = df_fundy[(df_fundy['gvkey'] == gvkey) & (df_fundy['datadate'] <= date_adj)].sort_values(by='fyear',
                     ascending=False)
        ltd = debt_filter['dltt'].iloc[0]
        std = debt_filter['dlc'].iloc[0]

    return std, ltd
def get_vars_revival(gvkey, key):
    # Assuming 'key' and 'gvkey' are defined earlier
    year = key // 10000
    fiscal_date = datetime(year, 12, 31)


    try:
        df_filt_findsic = df_chs[(df_chs['DATE'] <= fiscal_date) & (df_chs['GVKEY'] == gvkey)].sort_values(by='DATE',
                                                                                                           ascending=False)
        twosic = df_filt_findsic['TWOSIC'].iloc[0]
    except:
        twosic = None


    if twosic is None:
        try:
            df_filt_findsic = df_fundy[(df_fundy['gvkey'] == gvkey) & (df_fundy['fyear'] <= year)].sort_values(
                by='fyear', ascending=False)
            twosic = df_filt_findsic['TWOSIC'].iloc[0]
        except Exception as e:
            print(f"Error twsoic: {e}", gvkey, key)
            twosic = None



    try:
        shares_thousands = df_dsf[(df_dsf['DATE'] <= fiscal_date) & (df_dsf['GVKEY'] == gvkey)].sort_values('DATE', ascending=False)[
            'SHROUT'].iloc[0]
        shares_millions = shares_thousands / 1000
    except Exception as e:
        shares_millions = None
        print(f"Error shares_millions: {e}", gvkey, key)


    try:
        sigma = sigma_frm(gvkey, key, twosic)
    except Exception as e:
        sigma = None
        print(f"Error sigma: {e}", gvkey, key)

    try:
        rf, prc = rf_prc(gvkey, key)
    except Exception as e:
        rf, prc = None, None

        print(f"Error rf and prc: {e}", gvkey, key)

    try:
        couponrate = coupon_rate(gvkey, key)
    except:
        couponrate = None
        print(f"no CHS data, using AAA:", gvkey, key)

    if couponrate is None:
        try:
            aaa_filt = aaa_df[(aaa_df['DATE'] <= fiscal_date)].sort_values(by='DATE', ascending=False)
            couponrate = _to_decimal(aaa_filt['AAA'].iloc[0])
        except:
            pass



    try:
        std, ltd = std_ltd_i(gvkey, key)
    except Exception as e:
        std, ltd = None, None
        print(f"Error std and ltd: {e}", gvkey, key)




    try:
        xgm, gm, y_0 = x_gm(gvkey, key)
    except Exception as e:
        xgm, gm, y_0 = None, None, None
        print(f"Error xgm, gm, and y_0: {e}", gvkey, key)

    try:
        f, f_ratio = fixed_costs(gvkey, key)
    except Exception as e:
        f, f_ratio = None, None
        print(f"Error f_ratio: {e}", gvkey, key)


    try:
        lev = find_ind_mkt_lev(key, twosic)
    except Exception as e:
        lev = None
        print(f"Error lev: {e}", gvkey, key)

   # ra, ra_u = None, None
    #try:
     #   ra = r_a(gvkey, key, lev)
      #  ra_u = r_a(gvkey, key, 0)
    #except Exception as e:
     #   print(f"Error ra and ra_u: {e}", gvkey, key)
    try:
        sti = st_i(key)
    except Exception as e:
        print(f"Error sti: {e}", gvkey, key)
        sti = None


    try:
        csr, dsr = csr_dsr(twosic, key)
    except Exception as e:
        print(f"Error csr and dsr: {e}", gvkey, key)
        csr, dsr = None, None

    # mu_p = None
    # try:
    #     mu_p = mu_p_derived(mu_q, ra, rf)
    # except Exception as e:
    #     print(f"Error mu_p: {e}", gvkey, key)


    ##########swap location of f_ratio w f in return bc f_ratio is wrong value to use - done

    #cod, xgm, gm, y_0, sigma, f, f_ratio, csr, dsr, couponrate, std, ltd, lev, rf, prc, ra_u, ra, twosic
    return sigma, xgm, gm, y_0, f, couponrate, csr, dsr, std, ltd, lev, rf, prc, shares_millions, sti
def x_star(c, f, dsr, csr, x_gm):

    # Tau
    tau = float(.35)

    # Define denominator for readability
    x_star_den = ((1 - tau + (tau * dsr - csr)) / (x_gm))

    # Solve for x_star
    x_star = ((c + f) * (1-tau)) / (x_star_den)

    return x_star


# output no different from original roots function
def roots_claude(sigma, mu_Q, r):
    # """Calculate positive and negative roots of the quadratic equation"""
    # 1/2 * sigma^2 * beta * (beta - 1) + mu_Q * beta - r = 0
    # Rearranging: (sigma
    #
    # ^2/2) * beta^2 + (mu_Q - sigma^2/2) * beta - r = 0

    a = sigma ** 2 / 2
    b = mu_Q - sigma ** 2 / 2
    c = -r

    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("No real roots for beta equation")

    beta1 = (-b + np.sqrt(discriminant)) / (2 * a)  # positive root
    beta2 = (-b - np.sqrt(discriminant)) / (2 * a)  # negative root

    return beta1, beta2
def roots(sigma, mu_q, rf):

    sigma2 = sigma**2
    # b1 is pos root, b2 is neg root in paper
    beta1 = ((-1)/(sigma2))*((mu_q-0.5*sigma2)-(math.sqrt(((mu_q-0.5*sigma2)**2)+(2*rf*sigma2))))
    beta2 = ((-1)/(sigma2))*((mu_q-0.5*sigma2)+(math.sqrt(((mu_q-0.5*sigma2)**2)+(2*rf*sigma2))))
    #beta1 = (math.sqrt(4*g**2 - 4*g*sigma**2 + 8*rf*sigma**2 + sigma**4) - 2*g + sigma**2)/(2*sigma**2)
    #beta2 = -(math.sqrt(4*g**2 - 4*g*sigma**2 + 8*rf*sigma**2 + sigma**4) + 2*g - sigma**2)/(2*sigma**2)
    return beta1, beta2
def _to_decimal(rate):
    # FRED corporate yields are quoted in percent (e.g. 11.5); the model needs decimals
    return rate / 100 if rate > 1 else rate
def coupon_rate(gvkey, key):
    str_key = str(key)
    year = int(str_key[:4])

    date = datetime(year=year, month=12, day=31)
    date_cod = datetime(year, 12, 31)
    date_rf = datetime((key // 10000), 1, 1)
    # Grab risk-free
    rf = df_rf[(df_rf['Date'] == date_rf)]['rf'].iloc[0] / 100

    # Grab quintile and make coupon rate calc
    df_filt_find = df_chs[(df_chs['DATE'] <= date) & (df_chs['GVKEY'] == gvkey)].sort_values(
        'DATE', ascending=False)

    frm_quint = df_filt_find['QUINTILE'].iloc[0]

    # yields of AAA and BBB corp bonds used as spread in couponrate(aka cost of debt)= r + spread eq
    # Extract the year from 'date_cod'
    year_cod = date_cod.year

    # Filter rows by the same year as 'date_cod' and calculate the average
    aaa_filt_step = aaa_df[aaa_df['DATE'].dt.year == year_cod]
    aaa_filt = aaa_filt_step[aaa_filt_step['AAA'] != '#N/A']
    aaa = aaa_filt['AAA'].mean()

    baa_filt_step = baa_df[baa_df['DATE'].dt.year == year_cod]
    baa_filt = baa_filt_step[baa_filt_step['BAA'] != '#N/A']
    #aaa_filt = aaa_df[(aaa_df['DATE'] <= date_cod)].sort_values(by='DATE', ascending=False)
    #baa_filt = baa_df[baa_df['DATE'] <= date_cod].sort_values(by='DATE', ascending=False)

    if date_cod.year in [1983, 1984, 1985]:
        baa = baa_filt['BAA'].iloc[0]
    elif not date_cod.year in [1983, 1984, 1985]:
        baa = baa_filt['BAA'].mean()
    else:
        print('baa_filt is empty')

    #if not baa_filt.empty:
    #    baa = baa_filt['BAA'].mean()
    #else:
    #    baa = aaa + float(0.01)
    #print(aaa, baa, rf)

    # EGZ Table 1: coupon rate = AAA, BBB and BBB+2% yields for CHS quintiles 1-2, 3-4, 5
    # (mean 8.35%), i.e. the yield itself, not rf + yield. Quintile labels here are 0-4.
    aaa, baa = _to_decimal(aaa), _to_decimal(baa)
    if frm_quint in (0, 1):
        coupon_rate = aaa
    elif frm_quint in (2, 3):
        coupon_rate = baa
    else:
        coupon_rate = baa + 0.02

    return coupon_rate
def check_cf(gm_5, xgm, dsr, csr, c, f):
    tau = 0.35
    eta = 0.15

    dt = 0.1
    #dW_t = np.random.normal(0, np.sqrt(dt))
    #dy_t = (mu_q - 0.5*sigma**2)*dt + sigma*dW_t

    #new state variable
    #y_0 = np.log(gm)
    #dy = sigma*np.sqrt(dt)
    #x_5 = np.exp(y_0 + n*dy)

    last_term = ((1 - tau) * (c + f))

    #cf_check = gm_5 * (((tau * dsr - csr) / xgm) + (1 - tau)) - last_term
    #cf_main = x_5 * (((tau * dsr - csr) / xgm) + (1 - tau)) - last_term
    #cf_distress = x_5 * (((tau * dsr - csr) / xgm) + (1 - tau)) * (1 + eta) - last_term

    cf_main = gm_5 * (((tau * dsr - csr) / xgm) + (1 - tau)) - last_term
    cf_distress = (cf_main) * (1+eta)

    if cf_main < 0:
        # cf is negative so cf * eta will give us a negative value and the proportion of cf we are left with after deducting the proportional distress cost
        cf = cf_distress*dt
    else:

        cf = cf_main*dt
    # print('cf:', cf)
    return cf

def find_Vu_xd_revival2(gm_5, xgm, mu_q, dsr, csr, f, rf):
    tau = 0.35
    g = mu_q

    cashflow_coef = (((tau * dsr - csr) / xgm) + (1 - tau))
    perp_term =  cashflow_coef * gm_5 / rf - g
    costs = (1-tau)*f / rf

    Vu_val_yr5 = perp_term - costs

    if np.isinf(Vu_val_yr5):
        # print('VU IS INF')
         Vu_val_yr5 = 99999999

    if pd.isna(Vu_val_yr5):
        # print('VU IS NAN')
        Vu_val_yr5 = 99999999

    return Vu_val_yr5

def find_Vu_xd_revival(gm_5, xgm, mu_q, b1, b2, dsr, csr, f, rf, prc):
    A = 0
    c = 0
    eta = 0.15
    tau = 0.35

    g = mu_q

    fill = (((tau * dsr - csr) / xgm) + (1 - tau))
    xstar_u = x_star(c, f, dsr, csr, xgm)
    guess_u = ((0-(-fill*(gm_5/(rf-g))-(1-tau)*((c+f)/rf)))/gm_5**b2) # verify this i think (i-t)*c or zero bc copuon is zero
    initial_guess = [guess_u, 0, guess_u, xstar_u]

    # print('arguments:', gm_5, xgm, mu_q, b1, b2, dsr, csr, f, rf)
    # sys of eq assumes a value called xstar is fed, c is hardcoded as 0
    B_u, C_u, D_u, xd_u = fsolve(sys_of_eq_revival, initial_guess, args=(0, f, rf, dsr, csr, xstar_u, mu_q, xgm, b1, b2, prc, tau, eta))
    # print('unknowns:', B_u, C_u, D_u, xd_u)

    if gm_5 >= xstar_u:
        #print(B_u*gm_5**b2, fill*(gm_5/(rf-g)), ((1-tau) * f / rf))
        Vu_val_yr5 = A*gm_5**b1 + B_u*gm_5**b2 + fill*(gm_5/(rf-g)) - ((1-tau) * (f / rf))
        # print('Vu function values:', B_u*gm_5**b2, fill*(gm_5/(rf-g)), - ((1-tau) * f / rf))
    else:
        Vu_val_yr5 = C_u*gm_5**b1 + D_u*gm_5**b2 + (1+eta)*fill*(gm_5/(rf-g)) - ((1-tau) * (f / rf))

    if np.isinf(Vu_val_yr5):
        # print('VU IS INF')
         Vu_val_yr5 = 99999999

    if pd.isna(Vu_val_yr5):
        # print('VU IS NAN')
        Vu_val_yr5 = 99999999

    return Vu_val_yr5

def sys_of_eq_revival(vars, *args):
    # last edited 11/11/24: bx^2_sub changed to prc
    # Set A = 0 (no bubbles)
    # Set Bx^β2 = ObservedEquity - (3rd term - 4th term)
    # Set C = 0 and D = B (following same rationale)
    # vars = [B, xd] (only 2 unknowns now since C=0, D=B)

    # Suppress RuntimeWarning
    #warnings.filterwarnings("ignore", category=RuntimeWarning)

    c, f, rf, dsr, csr, xstar, mu_q, xgm, gm_5, b1, b2, prc, tau, eta = args
    #mu_q is g
    B, C, D, x_d = vars

    step = (((tau*dsr-csr)/xgm) + (1-tau))
    fourth_term = ((1 - tau) * (c + f)) / rf
    # bxb2_sub = prc - ((third_term * gm_5 / (rf - mu_q)) - fourth_term) #full third term includes the xgm/rf-g
    # should i just make this bx^2 = prc? -- powerpoint says wise to use second sub proposed
    # bxb2_sub = prc

    # Define the system of equations

    # left_side = B * xstar ** b2 + third_term * xstar / (rf - mu_q) - fourth_term
    eq1_t3 = xstar / (rf - mu_q)
    left_side = prc + ((step * xstar) / (rf - mu_q)) - fourth_term
    right_side = C * xstar ** b1 + D * xstar ** b2 + (1 + eta) * (
            (step * xstar) / (rf - mu_q) - fourth_term)
    eq1 = left_side - right_side
    # top is my eq i got from paper, bottom is gregorios
    # eq1 = bxb2_sub + third_term * (xstar/(rf-mu_q)) - fourth_term - C*xstar**b1 - D*xstar**b2 - (1 + eta) * ((third_term*(xstar/(rf-mu_q)))-fourth_term)
    #eq1 = B*xstar**b2 + fill * (xstar/(rf-mu_q)) - fourth_term - C*xstar**b1 + D*xstar**b2 + (1 + eta) * ((fill*(xstar/(rf-mu_q)))-fourth_term)

    left_deriv = b2 * prc ** (-1) + (step * 1) / (rf - mu_q)
    # left_deriv = b2 * B * (xstar ** (b2 - 1)) + third_term * 1 / (rf - mu_q)
    right_deriv = b1 * C * (xstar ** (b1 - 1)) + b2 * D * (xstar ** (b2 - 1)) + ((1 + eta) * step * 1) / (rf - mu_q)
    eq2 = left_deriv - right_deriv

    # top is my eq i got from paper, bottom is gregorios
    # eq2 = b2*B*xstar**(b2-1) + third_term * (1/(rf-mu_q)) - b1*C*xstar**(b1-1) - b2*D*xstar**(b2-1) - (1 + eta)*third_term*(1/(rf-mu_q))
    #eq2 =  b1*C*xstar**(b1-1) + b2*D*xstar**(b2-1) + (1 + eta)*fill*(1/(rf-mu_q)) - b2*B*xstar**(b2-1) + fill * (1/(rf-mu_q))

    # With C=0 and D=B, these simplify significantly:
    # eq1 becomes much simpler since C=0
    # eq2 also simplifies


def find_c_to_match_leverage_revivalclaude(c, f, rf, dsr, csr, mu_q, gm_5, xgm, b1, b2, prc, shares, target_leverage):
    if gm_5 <= 0:
        print('gm_5 <= 0, firm defaults')
        return 0, 0, 0

    # Define knowns
    A = 0
    eta = 0.15
    tau = 0.35
    tolerance = .075
    max_iterations = 100

    fill = (((tau * dsr - csr) / xgm) + (1 - tau))
    fill_guess = (-((tau * dsr - csr) / xgm) + (1 - tau))
    g = mu_q

    Vu_val_yr5 = find_Vu_xd_revival2(gm_5, xgm, mu_q, dsr, csr, f, rf)

    # Newton's method parameters
    epsilon = 1e-6  # Small increment for numerical derivative
    alpha = 0.5  # Damping factor to prevent overshooting

    for iteration in range(max_iterations):
        # Compute leverage at current c
        leverage_c, equity_val, D_val, B, C, D, xd = compute_leverageclaude(
            c, f, rf, dsr, csr, xgm, gm_5, b1, b2, tau, eta, fill, fill_guess, g, Vu_val_yr5
        )

        if leverage_c is None:  # Error in computation
            return Vu_val_yr5, 0, 0, B, C, D, xd

        # Check convergence
        error = leverage_c - target_leverage
        print(
            f'Iter {iteration}: c={c:.4f}, leverage={leverage_c:.4f}, target={target_leverage:.4f}, error={error:.4f}')

        if abs(error) <= tolerance:
            print(f"CONVERGED at c={c:.4f}")
            return equity_val, D_val, c, B, C, D, xd

        # Compute numerical derivative: d(leverage)/d(c)
        c_perturbed = c + epsilon
        leverage_perturbed, _, _, _, _, _, _ = compute_leverageclaude(
            c_perturbed, f, rf, dsr, csr, xgm, gm_5, b1, b2, tau, eta, fill, fill_guess, g, Vu_val_yr5
        )

        if leverage_perturbed is None:
            leverage_perturbed = leverage_c  # Fallback

        derivative = (leverage_perturbed - leverage_c) / epsilon

        # Newton update with damping
        if abs(derivative) > 1e-10:  # Avoid division by zero
            c_new = c - alpha * error / derivative
        else:
            # Fall back to simple step if derivative is too small
            c_new = c + 0.1 * (-1 if error > 0 else 1)

        # Ensure c stays positive
        if c_new < 0:
            print("COUPON NEGATIVE, VU ASSIGNED", Vu_val_yr5)
            return Vu_val_yr5, 0, 0, B, C, D, xd

        c = c_new

    raise ValueError(f"Failed to converge: last leverage={leverage_c:.4f}, target={target_leverage:.4f}")


def compute_leverageclaude(c, f, rf, dsr, csr, xgm, gm_5, b1, b2, tau, eta, fill, fill_guess, g, Vu_val_yr5):
    """Helper function to compute leverage ratio for a given c"""
    try:
        A = 0
        xstar = x_star(c, f, dsr, csr, xgm)
        guess2 = (0 - fill_guess * (gm_5 / (rf - g)) - (1 - tau) * ((c + f) / rf)) / (gm_5 ** b2)

        if math.isinf(guess2):
            return None, None, None, None, None, None, None

        fourth_term = (1 - tau) * (c + f) / rf
        initial_guess = [guess2, 0, guess2, xstar]

        B, C, D, xd = fsolve(sys_of_eq_revival, initial_guess,
                             args=(c, f, rf, dsr, csr, xstar, g, xgm, gm_5, b1, b2, prc, tau, eta))

        if gm_5 >= xstar:
            equity_val = A * gm_5 ** b1 + B * gm_5 ** b2 + fill * gm_5 / (rf - g) - fourth_term
        else:
            equity_val = C * gm_5 ** b1 + D * gm_5 ** b2 + (1 + eta) * (fill * gm_5 / (rf - g) - fourth_term)

        D_val = c / rf + ((gm_5 / xd) ** b2) * ((1 - eta) * Vu_val_yr5 - (c / rf))
        leverage_ratio = D_val / (D_val + equity_val)

        return leverage_ratio, equity_val, D_val, B, C, D, xd
    except:
        return None, None, None, None, None, None, None

def find_c_to_match_leverage_revival(c, f, rf, dsr, csr, mu_q, gm_5, xgm, b1, b2, prc, shares, target_leverage):
    if gm_5 <= 0:
        print('gm_5 <= 0, firm defaults')
        return 0, 0, 0

    # count = 0
    leverage_ratios_history = []
    coupon_iter = 0

    # Define knowns
    A = 0
    eta = 0.15
    tau = 0.35
    tolerance = .075 # Tolerance for convergence - could put logic so we try within 10% then go smaller and smaller until we get as close as we can and take the best iteration
    max_iterations = 100000  # Maximum number of iteration


    # Third Term fill
    fill = (((tau * dsr - csr) / xgm) + (1 - tau))
    fill_guess = (-((tau * dsr - csr) / xgm) + (1 - tau))
    g = mu_q


    Vu_val_yr5 = find_Vu_xd_revival2(gm_5, xgm, mu_q, dsr, csr, f, rf)


    for _ in range(max_iterations):

        # c parameter is initial guess

        # guess
        # print("gm_5", gm_5, "b1",b1, "b2", b2)
        xstar = x_star(c, f, dsr, csr, xgm)
        # guess = (fill*gm_5/(rf-g)-((1-tau)*(c+f)/rf))/(gm_5**b2)
        guess2 = (0 - fill_guess * (gm_5 / (rf - g)) - (1 - tau) * ((c + f) / rf)) / (gm_5 ** b2)

        if math.isinf(guess2):
            print("B_guess", guess2)
            print("fill_guess", fill_guess, "gm_5:", gm_5, "rf:", rf, "miuq:", g, "xgm:", xgm, "c:", c, "f:", f, "b2:", b2, "tau:", tau)
            return



        fourth_term = (1 - tau) * (c + f) / rf
        #[B,C,D,xd]
        initial_guess = [guess2, 0, guess2, xstar] # [B, C, D, xd]

        # solve system of equations w arbitrary c and store in variable names
        B, C, D, xd = fsolve(sys_of_eq_revival, initial_guess, args=(c, f, rf, dsr, csr, xstar, g, xgm, gm_5, b1, b2, prc, tau, eta))
        print("B_guess", guess2, "B:", B, "C:", C, "D:", D, "xd:", xd)
        # get e xi5 and d xi5 logs
        if gm_5 >= xstar:
            equity_val_yr5_postfin = A*gm_5**b1 + B*gm_5**b2 + fill*gm_5/(rf-g) - fourth_term
            # Vu_val_yr5 = A*gm_5**b1 + B*gm_5**b2 + fill*(gm_5/(rf-g)) - (((1-tau) * f) / rf)
        else:
            equity_val_yr5_postfin = C*gm_5**b1 + D*gm_5**b2 + (1+eta)*(fill*gm_5/(rf-g) - fourth_term)
            # Vu_val_yr5 = C*gm_5**b1 + D*gm_5**b2 + (1+eta)*(fill*(gm_5/(rf-g)) - ((1-tau) * f) / rf)

        D_val_yr5 = c / rf + ((gm_5 / xd) ** b2) * ((1 - eta) * Vu_val_yr5 - (c / rf))
        # D_val_yr5 = c / rf - ((gm_5 / xd) ** b2) * ((1 - eta) * (c/rf) - Vu_val_yr5)
        # this should be market leverage ratio not model leverage ratio - 7/6/25
        leverage_ratio = D_val_yr5 / (D_val_yr5 + equity_val_yr5_postfin)
        # leverage_ratio = D_val_yr5 / (D_val_yr5 + (prc*shares))
        # print("coupon guess:", c, "target leverage vs. guess leverage:", target_leverage, "vs. ", leverage_ratio )
        print('D:', D_val_yr5, 'E:', equity_val_yr5_postfin, "last lev: ", leverage_ratio, "Coupon:", c, "Goal:", target_leverage)
        # print("last lev: ", leverage_ratio, "Goal:", target_leverage, "Coupon:", c)
        if len(leverage_ratios_history) >= 3 and (leverage_ratios_history[0] - leverage_ratios_history[2]) == 0.00:
            # print("last lev: ", leverage_ratio, "Goal:", target_leverage)
            # print('D:', D_val_yr5, 'E:', equity_val_yr5_postfin)
            # print("COULD NOT CONVERGE, VU ASSIGNED", Vu_val_yr5)
            # if Vu_val_yr5 < 0:
            #      Vu_val_yr5 = 0
            #     print("VU negative, 0 assigned")
            #     print("unknowns:", B, C, D, xd)
            # else:
            #     print("last lev: ", leverage_ratio, "Goal:", target_leverage)
            #     print('D:', D_val_yr5, 'E:', equity_val_yr5_postfin)
            #     print("COULD NOT CONVERGE, VU ASSIGNED", Vu_val_yr5)
            # print("match:", leverage_ratios_history[0], leverage_ratios_history[2])
            print('could not converge,', "Vu val:", Vu_val_yr5)
            print("last lev: ", leverage_ratio, "Goal:", target_leverage)
            return Vu_val_yr5, 0, 0, B, C, D, xd # Return Vu_val_yr5

            # Append the current leverage ratio to the history
        leverage_ratios_history.append(leverage_ratio)

        # If the history exceeds 3 elements, remove the oldest element
        if len(leverage_ratios_history) > 3:
            leverage_ratios_history.pop(0)

        if abs(leverage_ratio - target_leverage) <= tolerance:
            # print('CONVERGED', 'D:', D_val_yr5, 'E:', equity_val_yr5_postfin, 'c:', c)
            # print("last lev: ", leverage_ratio, "Goal:", target_leverage)
            # print('Vu:', Vu_val_yr5)
            # if equity_val_yr5_postfin < 0:
            #     equity_val_yr5_postfin = 0
            #     print("EQUITY VALUE NEGATIVE, 0 ASSIGNED")
            #     print("unknowns:", B, C, D, xd)
            # else:
            #     print('CONVERGED', 'D:', D_val_yr5, 'E:', equity_val_yr5_postfin, 'c:', c)
            print("last lev: ", leverage_ratio, "Goal:", target_leverage, "coupon:", c)
            #     print('Vu:', Vu_val_yr5)
            # print('relevered successfully')
            print("Vu val:", Vu_val_yr5)
            print("B:", B, "C:", C, "D:", D, "xd:", xd)
            return equity_val_yr5_postfin, D_val_yr5, c, B, C, D, xd

        # Update c using binary search or another optimization method
        if (leverage_ratio - target_leverage) < 0:
            c += .25
            coupon_iter += 1
        else:
            c += -.25
            coupon_iter += 1

        print("Coupon:", c)

        if c < 0.00:
            # return value of unlev firm bc fixed costs already too high
            # Vu_val_yr5 = find_Vu_xd(gm_5, xgm, mu_q, b1, b2, dsr, csr, f, rf)

            # if Vu_val_yr5 < 0:
                # Vu_val_yr5 = 0
                # print("VU negative, 0 assigned")
                # print("unknowns:", B, C, D, xd)
            # else:
            #     print("last lev: ", leverage_ratio, "Goal:", target_leverage)
            #     print('D:', D_val_yr5, 'E:', equity_val_yr5_postfin)
            #     print("COUPON NEGATIVE, VU ASSIGNED", Vu_val_yr5)

            # print("last lev: ", leverage_ratio, "Goal:", target_leverage)
            # print('D:', D_val_yr5, 'E:', equity_val_yr5_postfin, 'c:', c)
            print("COUPON NEGATIVE, VU ASSIGNED", Vu_val_yr5)
            labels_and_values = [
                "c", c,
                "f", f,
                "xstar", xstar,
                "MiuQ", g,
                "xgm", xgm,
                "gm_5", gm_5,
                "prc", prc,
                "lev", leverage_ratio,
                "target_lev", target_leverage,
            ]

            print(labels_and_values)

            return Vu_val_yr5, 0, 0, B, C, D, xd
        #c += (target_leverage - leverage_ratio)

    raise ValueError("Failed to converge within maximum iterations", "last lev: ", leverage_ratio, "Goal:", target_leverage, "Coupon:", c)
def cf_node(mu_q, sigma, f, y_0, ltd, couponrate, dsr, csr, xgm, n):
    # assumptions
    tau = 0.35
    eta = 0.15
    dt = 0.1
    # v1 of solving for y, use v1
    y_t = y_0
    sigma_wp = np.sqrt(dt)

    # Arithmetic Brownian Motion
    for _ in range(n):
        dW_t = np.random.normal(loc=0, scale=sigma_wp)
        dy_t = (mu_q - 0.5 * sigma ** 2) * dt + sigma * dW_t
        # add each iteration's dy_t to the y_0 that was fed to function
        y_t += dy_t

    # restrict y to these bounds per the paper
    if y_t > 10:
        y_t = 10
    elif y_t < -5:
        y_t = -5
    # v2 solving for y
    # y_t = y_0
    # sigma_wp = np.sqrt(dt)
    # dy = sigma * math.sqrt(dt)
    # # Arithmetic Brownian Motion
    # for _ in range(n):
    #     dy = sigma * math.sqrt(dt)
    #     dy_t = (mu_q - 0.5 * sigma ** 2) * dt + dy
    #     # add each iteration's dy_t to the y_0 that was fed to function
    #     y_t += dy_t


    # create state variable and simplify
    st_var_node = np.exp(y_t)

    ###### determine i and f #######
    # f is proportional to sales, i/c is based on coupon rate = r+spread set by chs (just based on year 0 data then applied to the whole tree)
    # sale_dt
    #sale_node = st_var_node / xgm

    # f_dt, c_dt/i_dt
    #f = f_ratio * sale_node

    i = couponrate * ltd

    fill = (((tau * dsr - csr) / xgm) + (1 - tau))
    st_var_fill = st_var_node * fill
    fourth_term = (i + f) * (1 - tau)

    if st_var_fill < 0:
        cf = ((st_var_fill - fourth_term) * (1 + eta)) * dt
    else:
        cf = (st_var_fill - fourth_term)*dt


    return cf
def model_diff_revival(mu_q, sigma, xgm, gm, y_0, f, couponrate, csr, dsr, std, ltd, lev, rf, prc, shares, sti):

    # Get gm_5 list given mu_q guess
    gm_5_list = gm_5_list_create(mu_q, sigma, gm)

    # check for negative values in gm list
    for value in gm_5_list:
        if value < 0:
            print("gm_5_list has negative value")

    # find model_prc given mu_q
    model_eq, B, C, D, xd = get_model_prc_yr5_revival(mu_q, sigma, xgm, y_0, f, couponrate, csr, dsr, std, ltd, lev, rf, gm_5_list, prc, shares, sti)
    model_prc = model_eq / shares

    # return difference
    diff = model_prc - prc
    # print('model prc:', model_prc, 'observed prc:', obs_prc, 'diff:', diff, 'miu_q:', mu_q)

    return diff, B, C, D, xd
# Populates the grid with year five
def get_model_prc_yr5_revival(mu_q, sigma, xgm, y_0, f, couponrate, csr, dsr, std, ltd, lev, rf, gm_5_list, prc, shares, sti):

    # print('mu_q:', mu_q)

    b1, b2 = roots(sigma, mu_q, rf)

    N_t = 50
    dt = 0.1  # for tree/grid

    E = np.zeros((N_t + 1, N_t + 1))

    alpha = mu_q - 0.5 * sigma ** 2  # fill
    p_u = 0.5 + (alpha) * (math.sqrt(dt) / (2 * sigma))
    p_d = 1 - p_u

    # # loops 0 to 50 inclusive (51 cells)
    # for _ in range(N_t + 1):
    #
    #     # gets current year gross margin ratio, find gross margin at yr 5 for each branch at end of binomial tree
    #     # print('sigma:', sigma)
    #     # print('ra:', ra, 'rf:', rf)
    #     # Get gm5
    #     gm_5_list = simulate_xi_outcome_list(dt, years, gm, mu_p, sigma)

    i = 0
    for gm_5 in gm_5_list:
        # y_5 = y_it(y_0, mu_q, sigma, dt, 50) # i think this is redundant needs to be in the body of the tree section

        #we feed fixed costs directly and do not keep as proportional to sales bc that's what variable costs are dummy :)
        #f = fixed_costs_yr5(gm_5, xgm, f_ratio)
        # f = (gm/xgm) * f_ratio
        # print('gm_yr5:', gm_5, 'f_yr5:', f, 'f_ratio:', f_ratio)
        # print('ra:', ra)
        # print('mu_p:', mu_p, 'mu_q:', mu_q, 'b1:', b1, 'b2:', b2)
        # needd to update to f_5
        # print('csr:', csr, 'dsr:', dsr, 'mu_p:', mu_p, 'f_ratio:', f_ratio, 'f:', f)
        # print('roots:', b1, b2, 'mu_q:', mu_q)

        # # Get unlev val, c should be zero in arguments always so we get unknowns for firm with no leverage
        #  # coupon of zero for unlevered
        # c_unlev = 0
        # xstar_u = x_star(c_unlev, f, dsr, csr, xgm) #get xstar, we overwrite this variable later
        # ra_u = r_a(gvkey, key, 0)
        # mu_p_u = mu_p_ocf(ra_u, csr, dsr, xgm)
        # mu_q_u = mu_q_ocf(mu_p_u, ra_u, key)
        # Vu_val_yr5 = find_Vu_xd(c_unlev, gm_5, xgm, b1, b2, dsr, csr, mu_q_u, f, xstar_u, rf) # find value w constants from sys of eq
        # print('Vu_val_yr5', Vu_val_yr5)
        # find yr 5 levered values
        # print('xstar for vu:', xstar)
        # coupon guess is 200
        # i re calc xstar for each iteration of c until i converge within the find_c_to_match_leverage function
        # trying gm_5 as initial guess for c
        # could try mkt cap * some fraction as thats a good chunk of mkt leverage eq
        c_guess = gm_5/15
        e_xi5, d_xi5, c, B, C, D, xd = find_c_to_match_leverage_revivalclaude(c_guess, f, rf, dsr, csr, mu_q, gm_5, xgm, b1, b2, prc, shares, lev)

        # e_xi5, d_xi5, c, B, C, D, xd = find_c_to_match_leverage_revival(c_guess, f, rf, dsr, csr, mu_q, gm_5, xgm, b1, b2, prc, shares, lev)
        #print('e_xi5:', e_xi5, 'd_xi5:', d_xi5, 'frm_coupon:', c)

        # store them as so: [date, gvkey] = eq_yr5, debt_yr5, leverage, coupon
        tv_eq = e_xi5 + d_xi5
        # n is 50 for cf at period where terminal value is
        debt = (ltd + std)
        cf_5 = check_cf(gm_5, xgm, dsr, csr, c, f)
        eq_5 = tv_eq + cf_5 - debt
        # value_5 = eq_5
        value_5 = max(eq_5, 0)
        # does 1
        E[i, N_t] = value_5
        # print('inserted equity value:', value_5, 'to grid location:', i, ',', N_t)
        # print('# of paths filled:', i)
        i += 1


    count = 0
    # loop 0 to 49 inclusive starting at 49 going down (50 cells), m will be the year element
    for m in range(N_t - 1, -1, -1):
        # starts w i as 0
        i = count
        # next loop starts at 1 and so forth
        count += 1
        # print('i:', i)# Start from the second-to-last time period and go backward
        # does row 1 to 49 and then for th

        # each dt has values at i to 50 to fill (starts with 0 to 50 inclusive)
        for n in range(i, (N_t)):  # Iterate over nodes in each time period
            # Calculate option value using the binomial tree formula

            #if on 10th time layer then charge us prime interest
            if m == 10:
                #get ST int exp if at end of year 1
                sti_exp = std * sti
            else:
                #if not at end of yr 1 then ST int exp is 0
                sti_exp = 0

            cf = cf_node(mu_q, sigma, f, y_0, ltd, couponrate, dsr, csr, xgm, m)
            # print()

            u_value = E[(n + 1), (m + 1)]
            d_value = E[(n - 1), (m + 1)]
            # print('up_value:', u_value, 'down_value:', d_value)
            pv_eq = np.exp(-rf * dt) * (p_u * u_value + p_d * d_value)
            # print('pv_eq:', pv_eq, 'cf_node:', cf)

            eq_node = pv_eq + cf - (ltd + std) - sti_exp #PV of firm + cashflow - debt - short term interest ///// sti_exp is zero if not yr 1 (m==10)
            node_value = eq_node

            # E[n, m] = node_value
            E[n, m] = max(node_value, 0)
            # print('grid location filled at:', n, ',', m, 'with value:', E[n, m])

    return E[49, 0], B, C, D, xd
def find_root_brent_revivalclaude(a, b, gvkey, key):
    #initial_guess, obs_prc,
    sigma, xgm, gm, y_0, f, couponrate, csr, dsr, std, ltd, lev, rf, prc, shares, sti = get_vars_revival(gvkey, key)

    mu_q, B, C, D, xd = brentq(model_diff_revival, a, b, args=(sigma, xgm, gm, y_0, f, couponrate, csr, dsr, std, ltd, lev, rf, prc, shares, sti), xtol=1e-3)

    return mu_q, B, C, D, xd, sigma, xgm, gm, y_0, f, couponrate, csr, dsr, std, ltd, lev, rf, prc, shares, sti

def find_root_brent_revival(a, b, gvkey, key):
    #initial_guess, obs_prc,
    sigma, xgm, gm, y_0, f, couponrate, csr, dsr, std, ltd, lev, rf, prc, shares, sti = get_vars_revival(gvkey, key)

    mu_q, B, C, D, xd = brentq(model_diff_revival, a, b, args=(sigma, xgm, gm, y_0, f, couponrate, csr, dsr, std, ltd, lev, rf, prc, shares, sti), xtol=1e-3)

    return mu_q, B, C, D, xd, sigma, xgm, gm, y_0, f, couponrate, csr, dsr, std, ltd, lev, rf, prc, shares, sti
def validate_data_sqlcheck_wholebatch_revival(keyval_list):
    # Specify the file path
    # file_path = r"C:/Users/Fisher/Desktop/nan_none.txt"
    # file_path_2 = r"C:/Users/Fisher/Desktop/negative_gm.txt"
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password=os.environ.get("PGPASSWORD"),
        host="localhost",
        port="5432"
    )
    # Create a cursor
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT gvkey, date FROM frm_miuq.firm_yr")
    delete_list_sql = cur.fetchall()
    delete_list = [[gvkey, date] for gvkey, date in delete_list_sql]
    # print('to delete:', len(delete_list))
    cur.close()
    conn.close()

    list_adj = [item for item in keyval_list if item not in delete_list]

    list_adj = [item for item in list_adj if not any(pd.isna(value) for value in item)]

    print('length after deleting pairs in db already:', len(list_adj))

    list_revised = []
    list_neg_gm = []
    list_none = []

    count = 0
    count_neg_gm = 0
    for pair in list_adj:
        firm = pair[0]
        int_date = pair[1]

        # sigma, xgm, gm, y_0, f, couponrate, csr, dsr, std, ltd, lev, rf, prc, shares_millions, sti
        args = get_vars_revival(firm, int_date)

        #not sure if i should exclude i should hanlde it in function that uses it for casH FLWO
        if any(pd.isna(var) for var in args):
            list_none.append([int_date, firm])
            count += 1
            continue
        # Check if any element in vars is None
        elif any(var is None for var in args):
            list_none.append([int_date, firm])
            count += 1
            continue
        elif args[2] <= 0:
            list_neg_gm.append([int_date, firm])
            # print("gross margin <0, firm-year discarded:",
            #       gvkey, key, vars[2])
            count_neg_gm += 1
            continue

        else:
            list_revised.append([int_date, firm])


    # Write the list to the CSV file with a tab delimiter
    # with open(file_path, mode='a', newline='') as file:
    #     writer = csv.writer(file, delimiter=',')  # Specify the delimiter as tab
    #     for item in list_none:
    #         writer.writerow([item])

    # with open(file_path_2, mode='a', newline='') as file:
    #     writer = csv.writer(file, delimiter=',')  # Specify the delimiter as tab
    #     for item in list_neg_gm:
    #         writer.writerow([item])

    print('# of nan/none:', count, '# of negative gm:', count_neg_gm)
    return list_revised, list_none, list_neg_gm


def main_multiprocess_wholebatch_revival(keyval_pairs):
    # number of processes max seems to be 13/14, i am around 89-90% utilization w 12
    num_processes = 12

    #create queue
    #queue = multiprocessing.Queue()
    manager = multiprocessing.Manager()
    #combined_revise_guess = manager.list()
    gvkey_manager = manager.list(keyval_pairs)
    # shared list and shared output
     # i want this to store the output of get_sample_miuq
    pool = multiprocessing.Pool(processes=num_processes)

    # start worker processes
    print('starting multiprocessing')
    processes = []
    while not len(gvkey_manager) == 0:  # Continue until gvkey_manager is empty
        pair = gvkey_manager.pop(-1)  # Get the last element from gvkey_manager to avoid shifting comp time
        process = pool.apply_async(get_sample_miu_q_revival, args=(pair[1], pair[0])) #gvkey, key
        processes.append(process)

# processes = [pool.apply_async(get_sample_miu_q, args=(gvkey_manager.pop(0), key)) for _ in range(num_processes)]

# wait for the processes to finish
    for process in processes:
        process.get()
        #combined_revise_guess.extend(revised_guesses_workers)
    # Close the pool to release resources
    pool.close()
    pool.join()
    print('FINISHED LOADING')

    return

def get_sample_miu_q_revival(gvkey, key):
    # get_sample_miu_q(sp500_dict, initial_guess)

    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password=os.environ.get("PGPASSWORD"),
        host="localhost",
        port="5432"
    )
    # Create a cursor
    cur = conn.cursor()

    revise_guess_list = []

    # for key in sp500_dict:
    rf = rf_frm(key)

    try:
        # min(((mu_p + rf) * .99), .3)
        # initial guesses here as a and b
        # Maybe i need tighter starting guesses like -.05 to .05
        a = .1
        b = -.1 # try -rf?
        mu_q, B, C, D, xd, sigma, xgm, gm, y_0, f, couponrate, csr, dsr, std, ltd, lev, rf, prc, shares, sti = find_root_brent_revival(a, b, gvkey, key)

        #if miu_q_frmyr == str('gm<0'):
            #return print('gm<0, skipped gvkey:', gvkey)

        insert = [gvkey, mu_q, B, C, D, xd, sigma, xgm, gm, y_0, f, couponrate, csr, dsr, std, ltd, lev, rf, prc, shares, sti, key]

        # frm_miuq_df = frm_miuq_df.append(insert, ignore_index=True)

        # New Table Column Order (firm_yr)
        # gvkey, miuq, date, B, C, D, xd, sigma, xgm, gm, y_0,f, couponrate, csr, dsr, std, ltd, lev, rf, prc, shares, sti

        sql = """
                 INSERT INTO frm_miuq.data_revival2
                 (gvkey, miuq, B, C, D, xd, sigma, xgm, gm, y_0, f, couponrate, csr, dsr, std, ltd, lev, rf, prc, shares, sti, date)
                 VALUES %s
             """

        execute_values(cur, sql, insert)
        conn.commit()

        #pid = os.getpid()
        #Process ID: {pid},
        print("ROW ADDED:", insert)

    except ValueError:
        intersection = [gvkey, key]
       # pid = os.getpid()
        print("guess needs diff signs:", intersection)
        # revise_guess_list.append(intersection)

        # except IndexError:
        #     missing_datapoint = [gvkey, key]
        #     missing_data.append(missing_datapoint)
        #     print("missing data", missing_datapoint)


    return revise_guess_list
def get_frmyr_miu_q_revival(gvkey, key):
    # get_sample_miu_q(sp500_dict, initial_guess)

    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password=os.environ.get("PGPASSWORD"),
        host="localhost",
        port="5432"
    )
    # Create a cursor
    cur = conn.cursor()

    # for key in sp500_dict:
    rf = rf_frm(key)

    try:
        # min(((mu_p + rf) * .99), .3)
        # initial guesses here as a and b
        # Maybe i need tighter starting guesses like -.05 to .05
        a = 0.07
        b = -0.07 # try -rf?
        mu_q, B, C, D, xd, sigma, xgm, gm, y_0, f, couponrate, csr, dsr, std, ltd, lev, rf, prc, shares, sti = find_root_brent_revival(a, b, gvkey, key)

        #if miu_q_frmyr == str('gm<0'):
            #return print('gm<0, skipped gvkey:', gvkey)

        insert = [gvkey, mu_q, B, C, D, xd, sigma, xgm, gm, y_0, f, couponrate, csr, dsr, std, ltd, lev, rf, prc, shares, sti, key]

        # frm_miuq_df = frm_miuq_df.append(insert, ignore_index=True)

        # New Table Column Order (firm_yr)
        # gvkey, miuq, date, B, C, D, xd, sigma, xgm, gm, y_0,f, couponrate, csr, dsr, std, ltd, lev, rf, prc, shares, sti

        sql = """
                 INSERT INTO frm_miuq.data_revival2
                 (gvkey, miuq, B, C, D, xd, sigma, xgm, gm, y_0, f, couponrate, csr, dsr, std, ltd, lev, rf, prc, shares, sti, date)
                 VALUES %s
             """

        execute_values(cur, sql, insert)
        conn.commit()

        #pid = os.getpid()
        #Process ID: {pid},
        print("ROW ADDED:", insert)

    except ValueError:
        intersection = [gvkey, key]
       # pid = os.getpid()
        print("guess needs diff signs:", intersection)
        # revise_guess_list.append(intersection)

        # except IndexError:
        #     missing_datapoint = [gvkey, key]
        #     missing_data.append(missing_datapoint)
        #     print("missing data", missing_datapoint)


    return

def get_frmyr_miu_q_egz(gvkey, key):
    """
    Implied mu_Q (and R_A) for one firm-year using the paper-faithful engine in
    egz_model.py. Replaces the find_root_brent_revival / get_model_prc_yr5_revival path.
    """
    sigma, xgm, gm, y_0, f, couponrate, csr, dsr, std, ltd, lev, rf, prc, shares, sti = get_vars_revival(gvkey, key)
    firm = egz.FirmInputs(gm=gm, gm_ratio=xgm, csr=csr, dsr=dsr, F=f, sigma=sigma,
                          ltd=ltd, std=std, coupon_rate=couponrate, st_rate=sti,
                          target_lev=lev, r=rf, shares=shares)
    mu_q = egz.implied_mu_q(prc, firm)
    r_a = egz.implied_asset_return(mu_q, rf, xgm, csr, dsr)
    return {'gvkey': gvkey, 'date': key, 'mu_q': mu_q, 'r_a': r_a, 'sigma': sigma, 'xgm': xgm,
            'gm': gm, 'f': f, 'couponrate': couponrate, 'csr': csr, 'dsr': dsr, 'std': std,
            'ltd': ltd, 'lev': lev, 'rf': rf, 'prc': prc, 'shares': shares, 'sti': sti}

def build_rating_by_sic_df(df_dsf, df_sp_ratings, sp500_dict):

    # Copy and normalize
    df_sp_ratings = df_sp_ratings.copy()
    # df_dsf = df_dsf.copy()
    # df_dsf['MKT_CAP'] = df_dsf['PRC'] * df_dsf['SHROUT']
    # df_dsf['SIC2'] = df_dsf['SIC2'].astype('int64')

    df_sp_ratings.rename(columns={'gvkey': 'GVKEY', 'datadate': 'DATE'}, inplace=True)


    # Ratings map
    #overwrite so no +/- ratings
    rating_map = {
        'AAA': 1,
        'AA': 2, 'AA': 3, 'AA': 4,
        'A': 5, 'A': 6, 'A': 7,
        'BBB': 8, 'BBB': 9, 'BBB': 10,
        'BB': 11, 'BB': 12, 'BB': 13,
        'B': 14, 'B': 15, 'B': 16,
        'CCC': 17, 'CCC': 18, 'CCC': 19,
        'CC': 20, 'C': 21, 'D': 22
    }
    reverse_rating_map = {
        1: 'AAA',
        2: 'AA', 3: 'AA', 4: 'AA',
        5: 'A', 6: 'A', 7: 'A',
        8: 'BBB', 9: 'BBB', 10: 'BBB',
        11: 'BB', 12: 'BB', 13: 'BB',
        14: 'B', 15: 'B', 16: 'B',
        17: 'CCC', 18: 'CCC', 19: 'CCC',
        20: 'CC',
        21: 'C',
        22: 'D'
    }


    df_sp_ratings['splticrm'] = df_sp_ratings['splticrm'].str.strip().str.upper()
    df_sp_ratings['rating'] = df_sp_ratings['splticrm'].map(rating_map)
    df_sp_ratings = df_sp_ratings.dropna(subset=['rating'])
    # df_merged = pd.merge(df_dsf, df_sp_ratings[['GVKEY', 'rating']], on='GVKEY', how='left')

    # Extract 2-digit SIC codes
    df_sp_ratings['SIC_STR'] = df_sp_ratings['sic'].astype(str).str.zfill(4)
    df_sp_ratings['SIC2'] = df_sp_ratings['SIC_STR'].str[:2]
    df_sp_ratings['SIC2'] = df_sp_ratings['SIC_STR'].astype('int64')
    # df_dsf['HSICCD'] = df_dsf['HSICCD'].astype(str).str.zfill(4)
    # df_dsf['SIC'] = df_dsf['HSICCD'].str[:2]
    unique_sics = sorted(df_dsf['SIC2'].unique().tolist())

    # Store results as rows
    output_rows = []

    for key in sp500_dict:
        date_key = pd.to_datetime(key, format="%Y%m%d")
        year_str = str(date_key.year)

        for sic in unique_sics:
            df_filt = df_dsf[(df_dsf['SIC2'] == sic) & (df_dsf['DATE'] <= date_key)].sort_values(by='DATE', ascending=False)

            if not df_filt.empty:
                latest_date = df_filt.iloc[0]['DATE']
                # df_latest = df_filt[df_filt['DATE'] == latest_date]
                df_filt = df_dsf[(df_dsf['SIC2'] == sic) & (df_dsf['DATE'] == latest_date)].copy()
                df_merged = pd.merge(df_filt, df_sp_ratings[['GVKEY', 'rating']], on='GVKEY', how='left')
                total_mkt_cap = df_filt['MKT_CAP'].sum()

                if total_mkt_cap == 0:
                    print('mkt cap 0', sic, date_key)
                    continue
                if np.isnan(total_mkt_cap):
                    print('mkt cap empty', sic, date_key)
                    continue
                #issue with this funciton not getting all sic-year-ratings
                # df_merged = pd.merge(df_latest, df_sp_ratings[['GVKEY', 'rating']], on='GVKEY', how='left')
                # df_merged = df_merged.dropna(subset=['rating'])

                if df_merged.empty:
                    print('df_merged empty', sic, date_key)
                    continue

                df_merged['weighted_rating'] = df_merged['rating'] * df_merged['MKT_CAP']
                vw_rating = df_merged['weighted_rating'].sum() / df_merged['MKT_CAP'].sum()
                vw_rating_rounded = round(vw_rating)

                letter_rating = reverse_rating_map.get(vw_rating_rounded)

                if letter_rating:

                    row = {
                        'year': date_key,
                        'sic': sic,
                        'rating': letter_rating
                    }

                    output_rows.append(row)
                    # print(row)

    return pd.DataFrame(output_rows)
def find_cod_sp500_2(weight_dict, sp500_dict, df_sp_ratings, rating_by_year_sic_df, OAS_df, aaa_baa_spread_df, df_dsf, df_rf):
    cod_dict = {}
    output_rows = []
    df_sp_ratings = df_sp_ratings.copy()
    df_sp_ratings['splticrm'] = df_sp_ratings['splticrm'].str.strip().str.upper()
    df_sp_ratings = df_sp_ratings.sort_values(by='datadate', ascending=False)
    df_sp_ratings['datadate'] = pd.to_datetime(df_sp_ratings['datadate'], format='%Y%m%d')

    df_dsf = df_dsf.copy()
    # df_dsf['GVKEY'] = df_dsf['GVKEY'].astype(str)
    # df_dsf['HSICCD'] = df_dsf['HSICCD'].astype(str).str.zfill(4)
    # df_dsf['SIC2'] = df_dsf['HSICCD'].str[:2]
    # df_dsf['DATE'] = pd.to_datetime(df_dsf['DATE'], format='%Y%m%d')

    for date_key in sp500_dict:
        target_dt = pd.to_datetime(date_key, format="%Y%m%d")
        year = target_dt.year
        dec_31 = pd.Timestamp(year=int(year), month=12, day=31)

        cod_step = 0
        # weight_total = 0

        for gvkey in sp500_dict[date_key]:

            if gvkey is np.nan:
                continue
            if (date_key, gvkey) not in weight_dict:
                continue
            # print('STARTING', date_key, gvkey)
            frm_weight = weight_dict[(date_key, gvkey)]

            if frm_weight == 0:
                print("weight is zero:", date_key, gvkey)
                continue

            # check for if weight sum = 1 (it does checked via separate function)
            #weight_total += frm_weight

            # Try to get firm rating first

            rating_row = df_sp_ratings[
                (df_sp_ratings['gvkey'] == gvkey) &
                (df_sp_ratings['datadate'] <= target_dt)
                ]
            if not rating_row.empty and pd.notna(rating_row.iloc[0]['splticrm']):
                frm_rating = rating_row.iloc[0]['splticrm']
                #remove +/- bc no yield data available that granular
                frm_rating = frm_rating.replace('+', '').replace('-', '')
                # print('direct firm rating of:', frm_rating)

            else:
                # Fallback to most recent SIC code for this gvkey <= year-end
                #Getting sic filtered empty print so i think dec_31 var isn't working correctly
                #Finding firm SIC industry
                sic_filtered = df_dsf[
                    (df_dsf['GVKEY'] == int(gvkey)) &
                    (df_dsf['DATE'] <= dec_31)
                    ].sort_values(by='DATE', ascending=False)

                if not sic_filtered.empty:
                    #ratingbyyearsic has trimmed the + or - already in parent function
                    #store SIC industry value
                    sic2 = sic_filtered.iloc[0]['SIC2']
                    #get SIC industry equivalent rating
                    fallback_row = rating_by_year_sic_df[
                        (rating_by_year_sic_df['year'] <= dec_31) &
                        (rating_by_year_sic_df['sic'] == int(sic2))
                        ].sort_values(by='year', ascending=False)
                    if not fallback_row.empty:
                        frm_rating = fallback_row.iloc[0]['rating']
                        # print('sic-based firm rating of:', frm_rating)
                    else:
                        print("rating by sic row empty", date_key, gvkey, sic2)
                        continue  # No fallback rating
                else:
                    print("sic filtered empty", date_key, gvkey)
                    continue  # No SIC info to fallback on

            # Get spread from OAS_df or aaa_baa_spread_df (FYI both spread dfs are already net of 10Y)
            spread = None
            if year >= 1996:
                df_spread = OAS_df[OAS_df['observation_date'] <= dec_31].sort_values(by='observation_date', ascending=False).copy()
                df_spread.dropna(inplace=True)

                #if firm rating is CCC or lower use the CCC rating to get the yield for the ICE BofA CCC & Lower index
                if frm_rating in ['D', 'DD', 'DDD', 'C', 'CC', 'CCC']:
                    frm_rating = 'CCC'

                if not df_spread.empty and frm_rating in df_spread.columns:
                    spread = df_spread.iloc[0][frm_rating] / 100 #columns are titled by the rating ie A,AA,B,BBB,etc
                    # print('1996-2022 rating equivalent spread:', spread)
                else:
                    print('OAS row empty', dec_31, gvkey, frm_rating)

            else:
                #need work here - we only have AAA and BAA Moody (AAA and BBB S&P equiv)
                # we only have yield data for AAA and BAA (BBB) from Moody's for pre-1996
                if frm_rating in ['AAA', 'AA', 'A']:
                    frm_rating = 'AAA'
                elif frm_rating in ['BBB', 'BB', 'B']:
                    frm_rating = 'BBB'

                spread_row = aaa_baa_spread_df[aaa_baa_spread_df['observation_date'] <= dec_31].sort_values(by='observation_date', ascending=False)
                if not spread_row.empty and frm_rating in spread_row.columns:
                    spread = spread_row.iloc[0][frm_rating] / 100
                    # print('1983-1995 rating approximate spread:', spread)

            if spread is None or pd.isna(spread):
                print("spread is empty", date_key, gvkey)
                continue


            cod_step += spread * frm_weight

        rf = (df_rf[(df_rf['Date'] == pd.Timestamp(year=year, month=1, day=1))].iloc[0]['10y'] / 100)
        cod_dict[date_key] = cod_step + rf
        row = {
            'year': date_key,
            'cod': cod_step + rf
        }
        output_rows.append(row)


        # print('CoD:', cod_dict[date_key])


    cod_df = pd.DataFrame(output_rows)
    return cod_dict, cod_df
def get_sp500_weights_revival(sp500_dict):
    # initialize firm and sp500 mkt cap dictionaries. Also firm mkt cap weight dictionary.
    frm_mktcap_dict = {}
    sp500_mktcap_dict = {}
    weight_dict = {}


    #connect to db
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password=os.environ.get("PGPASSWORD"),
        host="localhost",
        port="5432"
    )
    # Create a cursor
    cur = conn.cursor()

    #for every key/year in the dictionary --
    for key in sp500_dict:
        # -- grab distinct GVKEY (for each year) *** CHANGE DATA TABLE ***
        cur.execute("SELECT DISTINCT gvkey FROM frm_miuq.data_revival WHERE date = %s", (key,))
        sp500_frms_yr = [row[0] for row in cur.fetchall()] # store data
        date = pd.to_datetime(key, format="%Y%m%d")

        #Set sp500 mkt cap to 0 at start of every year iteration
        sp500_mktcap = 0
        for gvkey in sp500_frms_yr:
            #filtered list on date and firm
            df_mktcap = df_dsf[(df_dsf['GVKEY'] == gvkey) & (df_dsf['DATE'] <= date)].sort_values(by='DATE',
                                                                                                  ascending=False)
            #calculate mkt cap based on filtered list NOTE THIS HAS SHARES IN THOUSANDS NOT MILLIONS
            frm_mktcap = df_mktcap['PRC'].iloc[0] * df_mktcap['SHROUT'].iloc[0]
            frm_mktcap_dict[key, gvkey] = frm_mktcap #store firm mkt cap in dictionary

            sp500_mktcap += frm_mktcap #add firm mkt cap to sp500 mkt cap variable

        #store sp500 mkt cap at end of each year iteration
        sp500_mktcap_dict[key] = sp500_mktcap

    for key in sp500_dict:
        cur.execute("SELECT DISTINCT gvkey FROM frm_miuq.data_revival WHERE date = %s", (key,))
        sp500_frms_yr = [row[0] for row in cur.fetchall()]
        for gvkey in sp500_frms_yr:
            frm_eqval = frm_mktcap_dict[key, gvkey] # grab firm mkt cap

            sp500_val = sp500_mktcap_dict[key] # grab sp500 mkt cap

            weight_frm = frm_eqval / sp500_val # calculate firm weight
            weight_dict[key, gvkey] = weight_frm # store firm weight

    #returns mkt cap weights for each firm relative to the sample, also gives mkt cap of sp500 and each firm in dictionary
    return sp500_mktcap_dict, weight_dict, frm_mktcap_dict
def sp500_miuq(sp500_dict, weight_dict):
    miuq_dict = {}

    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password=os.environ.get("PGPASSWORD"),
        host="localhost",
        port="5432"
    )
    # Create a cursor
    cur = conn.cursor()

    # CREATE MIUQ_DF
    # Execute the query to select all rows from frm_miuq.data
    cur.execute("SELECT * FROM frm_miuq.data_revival")

    # Fetch all rows from the cursor
    rows = cur.fetchall()

    # Get column names from the cursor description
    colnames = [desc[0] for desc in cur.description]

    # Create a DataFrame from the fetched rows and column names
    miuq_df = pd.DataFrame(rows, columns=colnames)
    count = 0
    for key in sp500_dict:
        # derive date
        date = pd.to_datetime(key, format="%Y%m%d")
        year_key = date.year
        date_cod = datetime(year_key, 12, 31)

        # frm_gvkeys = miuq_df[miuq_df['date'] == key]['gvkey'].unique()
        # print('firms in economy:', len(frm_gvkeys))

        # get econ miuq
        miuq_econ = 0

        for gvkey in sp500_dict[key]:
            if np.isnan(gvkey):
                continue

            weight = weight_dict.get((key, int(gvkey)))
            if weight is None or np.isnan(weight):
                continue

            # based on sql db table, now a df
            miu_q_frm = miuq_df[(miuq_df['date'] == key) & (miuq_df['gvkey'] == gvkey)]['miuq'].iloc[0]
            if np.isnan(miu_q_frm):
                print('miuq df error')
                continue

            # *new* for when the miuq is too negative
            # if miu_q_frm < -0.1:
            #     count += 1
            #     print(count, 'miuq < -0.2')
            #     continue

            # will have miuq for sp500 for specified date
            miuq_econ += (miu_q_frm * weight)

        print("ECON MIUQ:", miuq_econ)

        miuq_dict[key] = miuq_econ

    return miuq_dict
def find_yr_erp_revival(miuq_dict, cod_df, df_sp500_cap_structure, df_rf, us30y_df):
    count = 0
    output_rows = []
    for key in miuq_dict.keys():
        date = pd.to_datetime(key, format="%Y%m%d")
        year = date.year
        date_cod = datetime(year, 1, 1)

        # rf for both sides
        # rf = df_rf[df_rf['Date'] == date_cod]['10y'].iloc[0] / 100
        rf = df_rf[df_rf['Date'] == date_cod]['rf'].iloc[0] / 100

        # get cod from cod_df
        cod = cod_df[cod_df['year'] == key]['cod'].iloc[0]

        # 30y yield
        # mup_filt = us30y_df[us30y_df['DATE'] <= date_cod].sort_values(by='DATE', ascending=False)
        # mu_p = mup_filt['YIELD'].iloc[0]

        # avg quarterly gdp growth for that year
        # gdp_filt = gdp_df[(gdp_df['DATE'] <= date) & (gdp_df['DATE'] >= date_cod)]
        mu_p = .04
        # mu_p = gdp_filt['QTRLY_GDPGROWTH'].mean()

        miuq_econ = miuq_dict[key]

        # givens
        tau = 0.35
        w_e = df_sp500_cap_structure[df_sp500_cap_structure['date'] == date]['w_e'].iloc[0]
        w_d = df_sp500_cap_structure[df_sp500_cap_structure['date'] == date]['w_d'].iloc[0]

        # rearrange equality
        coe = (mu_p - miuq_econ + rf - ((1 - tau) * cod * w_d)) / w_e
        # erp = coe - rf

        # step = mu_p-miuq_econ+rf - (1-tau)*cod*w_d
        # print(step)
        # coe = step / w_e

        # we now capm (coe) is rf + Beta*(rm-rf), beta is 1 in this case as it is the market so
        # ERP = rm-rf, so coe = rf + ERP thus ERP = coe - rf

        erp_yr = coe - rf

        row = {
            'year': date,
            'erp': erp_yr,
            'coe': coe,
            'cod': cod,
            'rf': rf,
            'mu_p': mu_p,
            'miuq_econ': miuq_econ,
            'w_e': w_e,
            'w_d': w_d
        }

        output_rows.append(row)

    df_erp = pd.DataFrame(output_rows)

    return df_erp

# if __name__ == "__main__":
#     all_pairs = []
#
#     #target = [19831231, 19841231, 19841231]
#     for key, values in sp500_dict.items():
#         if key >= 19831231 and key <= 19841230:
#             for value in values:
#                 all_pairs.append([value, key])
#
#     valid_pairs, list_none, list_neg_gm = validate_data_sqlcheck_wholebatch_revival(all_pairs)
#
#     #works but need to fix the guess logic for c in find c to match lev function
#     main_multiprocess_wholebatch_revival(valid_pairs)

# Run single company for testing
gvkey = 1010
year = 19831231
get_frmyr_miu_q_revival(gvkey, year)

# key = year
#
# rf, obs_prc = rf_prc(gvkey, year)
#
# a = rf
# b = -0.1
# mu_q, B, C, D, xd, sigma, xgm, gm, y_0, f, couponrate, csr, dsr, std, ltd, lev, rf, prc, shares, sti = find_root_brent_revival(a, b, gvkey, key)


