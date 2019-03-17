#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd, matplotlib as mpl
import os, ast # This is to read in a list of valid columns


# In[2]:


# Dependency checks
import sys
print("Developed using: ")
print("Python version: 3.6.0")
print("Pandas version: 0.24.2")
print("Matplotlib version: 2.1.1") 
print("\n")
print("Current versions:")
print("Python version: "+sys.version[:5]) # 3.6.0
print("Pandas version: "+pd.__version__) # 0.23.4
print("Matplotlib version: "+mpl.__version__) # 2.1.1


# # Create sample.csv files for each Batch folder and aggregate master file

# In[3]:


# I manuallly trimmed some columns
# cols = str(df.columns.values.tolist())
# f = open("filtered_columns.txt", "w")
# f.write(cols)
# f.close()

# Here is a list of the bad columns
# bad_cols = str(df.loc[:,~df.columns.isin(cols)].columns.values.tolist())
# f = open("bad_columns.txt","w")
# f.write(bad_cols)
# f.close()

# Here is a list of columns to keep everything in order for edited_survey_data.csv
# final_cols = str(pd.read_csv("transformed_data/edited_survey_data.csv").columns.values.tolist())
# f = open("col_order.txt","w")
# f.write(final_cols)
# f.close()


# In[4]:


latest_batch = 4

# Our survey has an extra loop function. I created a list of valid columns
# file = open("filtered_columns.txt","r")
# good_cols = ast.literal_eval(file.read())
# file.close()

file = open("bad_columns.txt","r")
bad_cols = ast.literal_eval(file.read())
file.close()

# Create a master dataframe to hold all the batches
df_master = pd.DataFrame([])    

for i in range(1,latest_batch+1):
# Set batch number for file naming
    batch_n = str(i)
    # Read in batch data file 
    fname = "raw_data/Batch"+batch_n+"/qual.csv"
    df = pd.read_csv(fname,sep=",")[2:] # This index trims the first two rows which are metadata
    
    # In the fourth batch, we changed some column names (e.g. care_1 -> care)
    if int(batch_n) < 4:
        df.rename(columns = lambda x: x[:-2] if x.find("care_1")>0 else x,inplace=True)
        df.rename(columns = lambda x: x[:-2] if x.find("surprise_1")>0 else x,inplace=True)
    # Save a list of the renamed columns
    good_cols = df.loc[:,~(df.columns.isin(bad_cols))].columns
    # Save sample.csv which has trimmed columns
    df.loc[:,good_cols].to_csv("raw_data/Batch"+batch_n+"/sample.csv",index=False)
    # Smash this table against the existing table
    df_master = pd.concat([df_master,df[good_cols]],sort=False)

# Grab the preset column order
file = open("col_order.txt","r")
col_order = ast.literal_eval(file.read())
file.close()

# Drop the qualtrics identifying data
df_master["duration"] = df_master["Duration (in seconds)"] # Rename b/c it is easier
df_master.loc[:,col_order].to_csv("transformed_data/edited_survey_data.csv",sep=",",index=False)

# Save an unedited data file
df_master.to_csv("transformed_data/un-edited_survey_data.csv",sep=",",index=False)


# # Create data for analysis

# ## Recode variables to binary/numeric

# In[5]:


# Make a new df for the table
# Table columns will be [subject, fb_use,fb_fight,ig_use,ig_fight,pol,edu]
df_final = pd.DataFrame([])
df_final["subject"] = [x for x in df_master.reward_code] # Set subject number to reward code for now


# In[6]:


# Set column name - platform mapping
pref = [["fb_", "Facebook"],
        ["ig_", "Instagram"],
        ["fbm_", "Facebook Messenger"],
        ["wa_", "WhatsApp"],
        ["wc_","WeChat"],
        ["tblr_", "Tumblr"],
        ["red_", "Reddit"],
        ["sc_", "Snapchat"],
        ["yt_", "Youtube"],
        ["twt_", "Twitter"],]


# In[7]:


# Generate a list of all the platforms
platforms = [x[1] for x in pref]


# ## Mapping functions
# These reshape the values in the master data sheet into a table we can run analysis on

# In[8]:


# Find education level
def evaluate_eduation_level(_df):
    d = {
        "Some high school":1,
        "High school degree or equivalent (e.g. GED)":2,
        "Some college":3,
        "Associate degree (e.g. AA, AS)":4,
        "Bachelor's degree (e.g. BA, BS)":5,
        "Master's degree (e.g. MA, MS, MEd)":6,
        "Professional degree (e.g. MD, DDS, DVM)":7,
        "Doctorate (e.g. PhD, EdD)":8,
    }
    return _df.education.replace(d).values 


# In[9]:


# Find if a string paramater (platform) was used by a participant
def evaluate_used_platforms(_df, platform):
    # Create a list of lists of individual platforms
    platforms = _df.Platform_Use.str.split(",")
    
    # Return a list of binaries (True if platform found, else False)
    return [1 if platform in x else 0 for x in platforms ]


# In[10]:


# Find political leaning
def evaluate_political_leaning(_df):
    # Create a dictionary to hold the scale values
    d = {
        "Extremely liberal":1,
        "Very liberal":2,
        "Somewhat liberal":3,
        "Moderate":4,
        "Somewhat conservative":5,
        "Very conservative":6,
        "Extremely conservative":7,
    }
    return _df.political_leaning.replace(d).values # Use values property to ignore indexing problems


# In[11]:


# Find if a string paramater (platform) was used to fight by a participant
def evaluate_fighting_platforms(_df, platform):
    # Create a list of lists of individual platforms
    platforms = _df.Platform_Fights.str.split(",")
    # Return a list of binaries (True if platform found, else False)
    return [1 if platform in x else 0 for x in platforms ]


# ## Populate the table with values

# In[12]:


suf = "_use"
for item in pref:
    df_final[item[0]+suf] = evaluate_used_platforms(df_master,item[1])


# In[13]:


suf = "_fight"

for item in pref:
    df_final[item[0]+suf] = evaluate_fighting_platforms(df_master,item[1])


# In[14]:


df_final["edu"] = evaluate_eduation_level(df_master)
df_final["pol"] = evaluate_political_leaning(df_master)


# In[15]:


df_final = df_final.astype("int64",errors="ignore") # Change binary to numeric, ignore values that can't be changed to int (NA, eduation)


# In[16]:


df_final.set_index("subject",inplace=True)


# In[17]:


df_final.head()


# In[18]:


df_final.to_csv("transformed_data/wide_table.csv",sep=",")


# In[ ]:





# ## Make long data format

# In[19]:


df_fmt = pd.DataFrame([])


# In[20]:


for user in df_master.reward_code:
    temp_df = pd.DataFrame([])
    temp_df["platform"] = [x for x in platforms]
    temp_df["subject"] = user
    df_fmt = pd.concat([df_fmt,temp_df])


# In[21]:


df_fmt.head()


# In[23]:


def find_platform_use(plat_user):
    # Find the platforms used by each user
    plats = df_master.loc[(df_master.reward_code==plat_user[1]),"Platform_Use"].str.split(",").values
    # If the platform is in the platform_use column, return 1
    if (plat_user[0] in plats[0]):
        return 1
    else:
        return 0


# In[24]:


df_fmt["use"] = df_fmt.apply(find_platform_use,axis=1)


# In[25]:


def find_platform_fights(plat_user):
    # plat_user is a list with Platform and Subject
    # Find the platforms used by each user
    plats = df_master.loc[(df_master.reward_code==plat_user[1]),"Platform_Fights"].str.split(",").values
    # If the platform is in the platform_use column, return 1
    if (plat_user[0] in plats[0]):
        return 1
    else:
        return 0


# In[26]:


df_fmt["fight"] = df_fmt.apply(find_platform_fights,axis=1)


# In[27]:


# df_master.columns


# In[28]:


df_fmt["pol"] = df_fmt.subject.apply(lambda x: df_final.loc[df_final.index==x,"pol"].values[0])


# In[29]:


df_fmt["edu"] = df_fmt.subject.apply(lambda x: df_final.loc[df_final.index==x,"edu"].values[0])


# In[30]:


df_fmt.to_csv("transformed_data/long_table.csv",sep=",",index=False)


# ## Visualizations

# In[31]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


# What are fight / use rates like


# In[33]:


use_cols = df_final.columns[df_final.columns.str.contains("use")]
fight_cols = df_final.columns[df_final.columns.str.contains("fight")]


# In[34]:


df_final[use_cols].sum()


# In[38]:


fig = plt.figure(num=None, figsize=(16,7))

use = df_final[use_cols].sum()
fights = df_final[fight_cols].sum()

N = 10 #number of platforms
ind = np.arange(N)

p1 = plt.bar(ind, use.values, color="c")
p2 = plt.bar(ind, fights.values, color="red")

plt.ylabel("Count")
plt.title("Platform Usage and Fights")
plt.xlabel("Platforms")
plt.xticks(ind, platforms)
y_max = use.max()+2
plt.yticks(np.arange(0,y_max,step=4))
plt.legend(["Use","Fights"])

plt.show()

fig.savefig("platform_use.png")


# In[ ]:





# In[ ]:




