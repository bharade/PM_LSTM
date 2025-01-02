import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self,train_pkl_file:Path,test_pkl_file:Path):
        self.train_pkl_file = train_pkl_file
        self.test_pkl_file = test_pkl_file
        self.train_data = None
        self.test_data = None
    
    def prepare_dataframe(self,df):
        print("Removing unnecessary columns")
        cols = [c for c in df.columns if (c.lower().find("normalized")!=-1)]
        df=df.drop(columns=cols)
        df = df.drop(columns=['model','capacity_bytes'])
        df['date'] = pd.to_datetime(df['date'])
        print("Sorting the data frame based on serial number and date")
        df = df.sort_values(by=['serial_number', 'date'], axis=0, ascending=True)
        df = df.reset_index(drop=True)
        df = df.fillna(0)
        return df
    
    def adjust_dates(df_loc): 
        df_mod = pd.DataFrame()
        cur_serial = df_loc['serial_number'].unique().tolist()[0]
        col_list = df_loc.columns.tolist()
        cur_dates = df_loc['date'].values
        
        # determine number of days between last record and first record
        num_date_range = int((cur_dates[-1] - cur_dates[0]).astype('timedelta64[D]')/ np.timedelta64(1, 'D'))+1
        
        # do we have records for each day or are there holes ?  If so, fill them.
        if num_date_range > cur_dates.shape[0]:
            i_low = 0
            
            # step through all days to ensure next date correct
            for i in range(cur_dates.shape[0]-1): 
                
                # calculate number of days between current data and next data - should be 1 day
                diff_days = int((cur_dates[i+1] - cur_dates[i]).astype('timedelta64[D]')/ np.timedelta64(1, 'D'))
                
                # if not 1 day, fill in missing days with forward fill
                if diff_days > 1:
                    df_mod = df_mod.append(df_loc.iloc[i_low:i+1])
                    tmp_array = np.empty((diff_days-1,len(col_list),))
                    tmp_array[:] = np.nan
                    df_add = pd.DataFrame(tmp_array,columns=col_list)
                    df_add['date'] = [ cur_dates[i] + np.timedelta64(1, 'D')*j for j in range(1,diff_days)]
                    df_mod = df_mod.append(df_add)
                    i_low = i+1

            # add missing records and use forward fill to update missing sensor data
            df_mod = df_mod.append(df_loc.iloc[i_low:])
            df_mod = df_mod.fillna(method="ffill")
        else:
            df_mod = df_loc 
        
        return df_mod 
    
    def fix_date_gaps(df, normal_serials=None, failed_serials=None):
        df_fixed = pd.DataFrame()

        serials_list = normal_serials + failed_serials 
        for i, cur_serial in enumerate(serials_list): 
            df_fixed = df_fixed.append(adjust_dates(df[df['serial_number'] == cur_serial]))
            
        return df_fixed.reset_index(drop=True)
    
        # Routine to return failed sequences 
    def create_failed_sequences(df, sequence_length, lookahead):

        failed_serials = df.serial_number.unique().tolist()
        print("Number of failed serials : ", len(failed_serials)) 

        failed_seq_list = []
        for serial in failed_serials:
            df_tmp = df[df['serial_number'] == serial]
            df_tmp = df_tmp.reset_index(drop=True)
            num_recs = df_tmp.index.size
            
            # if enough records, add failed sequence
            if num_recs > (sequence_length+lookahead): 
                # find first failure
                df_failed = df_tmp[df_tmp['failure'] == 1]
                
                # find end of sequence - going back "lookahead" days from failure
                idx2 = df_failed.index[0] - lookahead + 1
                
                # find beginning of sequence
                idx1 = idx2 - sequence_length
                
                if idx1 > 0: 
                    failed_seq_list.append(df_tmp.iloc[idx1:idx2,:])
        
        print("Number of failed sequences :", len(failed_seq_list)) 
        
        return pd.concat(failed_seq_list)
    
        # Routine to pick some serial_numbers and create all sequences from those disks up to num_normal sequences
    def create_normal_sequences(df, sequence_length, num_normal, lookahead, day_step=1):
        normal_seq_list = []
        num_seq = 0
        
        # ensure no failed sequences
        if df[df['failure'] == 1].index.size > 0: return None 
        
        # get list of normal serial numbers
        normal_serials = df.serial_number.unique().tolist()
        
        print("Number of normal serials : ", len(normal_serials)) 
        
        for serial in normal_serials:
            df_tmp = df[df['serial_number'] == serial]
            num_recs = df_tmp.shape[0]
            
            # 
            for i in range(0, num_recs-(sequence_length+lookahead)+1, day_step):
                if (num_seq < num_normal):
                    normal_seq_list.append(df_tmp.iloc[i:i+sequence_length])
                    num_seq += 1
        
        print("Number of normal sequences :", len(normal_seq_list))
        
        return pd.concat(normal_seq_list)

    # Routine to add column "sequence_label" indicating whether this was a normal or failed sequence.
    # We will use this later for as label for training.
    def label_sequence(df, label):
        df.insert(2, 'sequence_label', np.full(df.shape[0], label), True)
        return

    def get_disk_serials(df, num_disks):
        # Get failed serial numbers
        failed_serials = df[df['failure'] == 1]['serial_number'].unique().tolist()

        # Get serial numbers for disks that didn't fail - first remove failed disks
        df_tmp = df[~df.serial_number.isin(failed_serials)]
        normal_serials = df_tmp.serial_number.value_counts()[:num_disks].index.tolist()

        print('Normal Disk Serials:',len(normal_serials))
        print('Failed Disk Serials:',len(failed_serials))

        return normal_serials, failed_serials
    


    def load_data(self):
        """load and preprocess data from the pkl files"""
        # Load our training and test set from previous step
        print("Reading in training and test SMART..")
        df_train = pd.read_pickle(self.train_pkl_file)
        df_test = pd.read_pickle(self.test_pkl_file)
        print(df_train.shape)
        print(df_test.shape)
        
        # Prepare the data frame
        print("Processing train set...")
        df_train = self.prepare_dataframe(df_train)
        print("Processing test set...")
        df_test = self.prepare_dataframe(df_test)

    # Routine to return serial numbers of good and bad disks

    