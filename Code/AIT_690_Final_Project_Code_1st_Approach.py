# AIT 690 Smart Cities and Urban Computing
# Japneet S. Kohli

################################################################################################
#
#This file contains the code for the 1st appraoch using Gaussian Mixture Models GMM.
#
#The order of execution is stated below.
#
#1.  Import libraries and set working directory.
#2.  Read data.
#
#*** Multi-Label Violence Classification ***
#
#3.  Add new column to determine violent vs non violent crimes based on key code values. This determination has total 6 class labels.
#4.  Keep subset of categorical attributes to be used as training data.
#5.  Convert string, dates, and other types to numerical type by one hot encoding.
#6.  Identify columns to remove from final analysis and drop them.
#7.  Create standalone target variable and assign violent vs.non-violent values to it.
#8.  Remove target variable from dataframe as standalone will be used going forward.
#9.  Split data into train and test sets. 
#10. Define GMM model and its parameters.
#11. Fit the GMM model with training data.
#12. Make predictions on test data.
#13. Analyze the accuracy of the GMM model.
#14. Create a confusion matrix and a classification report.
#15. Create a kmeans model, fit training data, redict test data, and obtain accuracy score, confusion matrix and classification report. This is done only to see how the GMM model compares with kmeans.
#
#*** Binary-Label Violence Classification ***
#
#16. Drop attributes not to be used in training data.
#17. Add new column to determine violent vs non violent crimes based on key code values. This determination has total only 2 class labels.
#18. Create standalone target variable and assign violent vs.non-violent values to it.
#19. Remove target variable from dataframe as standalone will be used going forward.
#20. Split data into train and test sets. 
#21. Define GMM model and its parameters.
#22. Fit the GMM model with training data.
#23. Make predictions on test data.
#24. Analyze the accuracy of the GMM model.
#25. Create a confusion matrix and a classification report.
#
#
#*** Binary-Label Violence Classification With Fewer Training Attributes ***
#
#26. Drop more categorical attributes not to be used in training data.
#27. Add new column to determine violent vs non violent crimes based on key code values. This determination has total only 2 class labels.
#28. Create standalone target variable and assign violent vs.non-violent values to it.
#29. Remove target variable from dataframe as standalone will be used going forward.
#30. Split data into train and test sets. 
#31. Define GMM model and its parameters.
#32. Fit the GMM model with training data.
#33. Make predictions on test data.
#34. Analyze the accuracy of the GMM model.
#35. Create a confusion matrix and a classification report.
#
#*** Predicting Cluster Values based on 3 GMM Models Created Above ***
#
#36. For multi-label GMM model, predict cluster assignments for all data points and get AIC and BIC numbers.
#37. For the first binary-label GMM model, predict cluster assignments for all data points and get AIC and BIC numbers.
#38. For the second binary-label GMM model, predict cluster assignments for all data points and get AIC and BIC numbers.
#
#*** Save Predictions ***
#
#39. For the three models above, export clustering predictions out so they could be used in ArcMap.

################################################################################################

# set working directory
#import os
import pandas as pd
import os
import numpy as np

os.getcwd()
os.chdir('Data')   # use your own path depending on where your data file is saved
os.getcwd()

# read the dataset

data = pd.read_csv('NYPD_Complaint_Map__Year_to_Date_.csv', sep=',', na_values=' ?')

data.info()
data.describe()

data.columns
data.KY_CD.unique()
data.OFNS_DESC.fillna(0).unique()
data.PD_DESC.fillna(0).unique()

data['PD_CD'] = data['PD_CD'].fillna(0)

# add new column to determine violent vs non violent crimes based on key code values

KY_CD_VNV = data.OFNS_DESC.fillna(0)


data.info()

newdf = pd.concat([data['CMPLNT_NUM'], 
                  data['ADDR_PCT_CD'], 
                  data['BORO_NM'], 
                  data['CMPLNT_FR_DT'], 
                  data['CMPLNT_FR_TM'], 
                  data['CMPLNT_TO_DT'], 
                  data['CMPLNT_TO_TM'], 
                  data['CRM_ATPT_CPTD_CD'], 
                  data['HADEVELOPT'], 
                  data['JURIS_DESC'], 
                  data['KY_CD'], 
                  data['LAW_CAT_CD'], 
                  data['LOC_OF_OCCUR_DESC'], 
                  data['OFNS_DESC'], 
                  data['PARKS_NM'], 
                  data['PD_CD'], 
                  data['PD_DESC'], 
                  data['PREM_TYP_DESC'], 
                  data['RPT_DT'], 
                  data['Lat_Lon'], 
                  data['X_COORD_CD'], 
                  data['Y_COORD_CD'], 
                  data['Latitude'], 
                  data['Longitude'],
                  KY_CD_VNV],axis=1,keys=['CMPLNT_NUM', 
                                          'ADDR_PCT_CD', 
                                          'BORO_NM', 
                                          'CMPLNT_FR_DT', 
                                          'CMPLNT_FR_TM', 
                                          'CMPLNT_TO_DT', 
                                          'CMPLNT_TO_TM', 
                                          'CRM_ATPT_CPTD_CD', 
                                          'HADEVELOPT', 
                                          'JURIS_DESC', 
                                          'KY_CD', 
                                          'LAW_CAT_CD', 
                                          'LOC_OF_OCCUR_DESC', 
                                          'OFNS_DESC', 
                                          'PARKS_NM', 
                                          'PD_CD', 
                                          'PD_DESC', 
                                          'PREM_TYP_DESC', 
                                          'RPT_DT', 
                                          'Lat_Lon', 
                                          'X_COORD_CD', 
                                          'Y_COORD_CD', 
                                          'Latitude', 
                                          'Longitude', 
                                          'KY_CD_VNV'])

newdf.info()
newdf
dict = {'ROBBERY':0, 
'PETIT LARCENY':0, 
'FELONY ASSAULT':5,
'ASSAULT 3 & RELATED OFFENSES':5, 
'SEX CRIMES':3, 
'HARRASSMENT 2':2,
'GRAND LARCENY':0, 
'THEFT-FRAUD':0, 
'BURGLARY':0,
'INTOXICATED & IMPAIRED DRIVING':0, 
'VEHICLE AND TRAFFIC LAWS':0,
'FORGERY':0, 
'DANGEROUS WEAPONS':0, 
'RAPE':5,
'GRAND LARCENY OF MOTOR VEHICLE':0, 
'DANGEROUS DRUGS':0,
'MURDER & NON-NEGL. MANSLAUGHTER':5, 
'KIDNAPPING':3,
'CRIMINAL MISCHIEF & RELATED OF':1, 
'MISCELLANEOUS PENAL LAW':0,
'FRAUDS':0, 
'POSSESSION OF STOLEN PROPERTY':0, 
'CRIMINAL TRESPASS':0,
'ARSON':0, 
'OFFENSES INVOLVING FRAUD':0,
'OFFENSES AGAINST PUBLIC ADMINI':0, 
'ADMINISTRATIVE CODE':0,
'UNAUTHORIZED USE OF A VEHICLE':0, 
'GAMBLING':0,
'OFF. AGNST PUB ORD SENSBLTY &':0, 
0:0,
'NYS LAWS-UNCLASSIFIED FELONY':0, 
'OFFENSES AGAINST THE PERSON':4,
'THEFT OF SERVICES':0, 
'KIDNAPPING & RELATED OFFENSES':3,
'OTHER OFFENSES RELATED TO THEF':0, 
"BURGLAR'S TOOLS":0, 
'ESCAPE 3':0,
'ENDAN WELFARE INCOMP':0, 
'FRAUDULENT ACCOSTING':1,
'AGRICULTURE & MRKTS LAW-UNCLASSIFIED':0,
'OTHER STATE LAWS (NON PENAL LA':0,
'OFFENSES AGAINST PUBLIC SAFETY':0,
'PETIT LARCENY OF MOTOR VEHICLE':0, 
'OFFENSES RELATED TO CHILDREN':0,
'ALCOHOLIC BEVERAGE CONTROL LAW':0, 
'FELONY SEX CRIMES':3,
'ANTICIPATORY OFFENSES':0, 
'LOITERING/GAMBLING (CARDS, DIC':0,
'JOSTLING':4, 
'HOMICIDE-NEGLIGENT,UNCLASSIFIE':5,
'PROSTITUTION & RELATED OFFENSES':0, 
'CHILD ABANDONMENT/NON SUPPORT':0,
'OTHER STATE LAWS':0, 
'NYS LAWS-UNCLASSIFIED VIOLATION':0,
'DISRUPTION OF A RELIGIOUS SERV':0, 
'DISORDERLY CONDUCT':0,
'OFFENSES AGAINST MARRIAGE UNCL':0, 
'HOMICIDE-NEGLIGENT-VEHICLE':5,
'INTOXICATED/IMPAIRED DRIVING':0, 
'KIDNAPPING AND RELATED OFFENSES':3,
'UNLAWFUL POSS. WEAP. ON SCHOOL':0,
'OTHER STATE LAWS (NON PENAL LAW)':0, 
'OTHER TRAFFIC INFRACTION':0}

newdf.KY_CD_VNV = newdf.KY_CD_VNV.map(dict)
newdf.info()
newdf

newdf2 = pd.concat([newdf['ADDR_PCT_CD'], 
                  newdf['BORO_NM'], 
                  newdf['CMPLNT_FR_DT'], 
                  newdf['CMPLNT_FR_TM'], 
                  newdf['CMPLNT_TO_DT'], 
                  newdf['CMPLNT_TO_TM'], 
                  newdf['CRM_ATPT_CPTD_CD'], 
                  newdf['HADEVELOPT'], 
                  newdf['JURIS_DESC'], 
                  newdf['LAW_CAT_CD'], 
                  newdf['LOC_OF_OCCUR_DESC'], 
                  newdf['PARKS_NM'], 
                  newdf['PD_CD'], 
                  newdf['PREM_TYP_DESC'], 
                  newdf['RPT_DT'], 
                  newdf['Latitude'], 
                  newdf['Longitude']],axis=1,keys=['ADDR_PCT_CD', 
                                          'BORO_NM', 
                                          'CMPLNT_FR_DT', 
                                          'CMPLNT_FR_TM', 
                                          'CMPLNT_TO_DT', 
                                          'CMPLNT_TO_TM', 
                                          'CRM_ATPT_CPTD_CD', 
                                          'HADEVELOPT', 
                                          'JURIS_DESC', 
                                          'LAW_CAT_CD', 
                                          'LOC_OF_OCCUR_DESC', 
                                          'PARKS_NM', 
                                          'PD_CD', 
                                          'PREM_TYP_DESC', 
                                          'RPT_DT', 
                                          'Latitude', 
                                          'Longitude'])

newdf2.info()
newdf2

## Convert string, dates, and other types to numerical type by one hot encoding
# Using get_dummies function in pandas library to perform this task

#newdf3 = pd.get_dummies(newdf, 
#                        columns=['ADDR_PCT_CD', 'BORO_NM', 'CMPLNT_FR_DT', 'CMPLNT_FR_TM', 'CMPLNT_TO_DT', 'CMPLNT_TO_TM', 'CRM_ATPT_CPTD_CD', 'HADEVELOPT', 'JURIS_DESC', 'LAW_CAT_CD', 'LOC_OF_OCCUR_DESC', 'PARKS_NM', 'PREM_TYP_DESC', 'RPT_DT'],
#                        prefix=['ADDR_PCT_CD', 'BORO_NM', 'CMPLNT_FR_DT', 'CMPLNT_FR_TM', 'CMPLNT_TO_DT', 'CMPLNT_TO_TM', 'CRM_ATPT_CPTD_CD', 'HADEVELOPT', 'JURIS_DESC', 'LAW_CAT_CD', 'LOC_OF_OCCUR_DESC', 'PARKS_NM', 'PREM_TYP_DESC', 'RPT_DT',])


newdf3 = pd.get_dummies(newdf, 
                        columns=['ADDR_PCT_CD', 'BORO_NM', 'CRM_ATPT_CPTD_CD', 'HADEVELOPT', 'JURIS_DESC', 'LAW_CAT_CD', 'LOC_OF_OCCUR_DESC', 'PARKS_NM', 'PREM_TYP_DESC'],
                        prefix=['ADDR_PCT_CD', 'BORO_NM', 'CRM_ATPT_CPTD_CD', 'HADEVELOPT', 'JURIS_DESC', 'LAW_CAT_CD', 'LOC_OF_OCCUR_DESC', 'PARKS_NM', 'PREM_TYP_DESC'])

newdf3.info() 

# columns to remove from final analysis
newdf3.CMPLNT_NUM
newdf3.KY_CD
newdf3.OFNS_DESC # can remove earlier as target variable
newdf3.PD_CD # same as KY_CD, so remove
newdf3.PD_DESC # same as pd_cd, so remove
newdf3.Lat_Lon # same as latitude and longitude column, so remove

# copy newdf3 to newdf4 to preserve newdf3 if required
newdf4 = newdf3

newdf4 = newdf4.drop('CMPLNT_NUM', axis=1)
#newdf4 = newdf4.drop('KY_CD', axis=1)
newdf4 = newdf4.drop('OFNS_DESC', axis=1)
#newdf4 = newdf4.drop('PD_CD', axis=1)
newdf4 = newdf4.drop('PD_DESC', axis=1)
newdf4 = newdf4.drop('Lat_Lon', axis=1)

newdf4 = newdf4.drop('CMPLNT_FR_DT', axis=1)
newdf4 = newdf4.drop('CMPLNT_FR_TM', axis=1)
newdf4 = newdf4.drop('CMPLNT_TO_DT', axis=1)
newdf4 = newdf4.drop('CMPLNT_TO_TM', axis=1)
newdf4 = newdf4.drop('RPT_DT', axis=1)

#newdf4 = newdf4.drop('Latitude', axis=1)
#newdf4 = newdf4.drop('Longitude', axis=1)

newdf4.info()

#target variable
target = newdf4.KY_CD_VNV

# remove target from dataframe
newdf4 = newdf4.drop('KY_CD_VNV', axis=1)



###################################

# split data into train and test

from sklearn.model_selection import train_test_split #training and testing data split

print(newdf4.shape,target.shape)

x_train, x_test, y_train, y_test = train_test_split(newdf4, target, test_size=0.2)

####################

#  Gaussian Mixture Models Method
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=6,
                      covariance_type='full',
                      tol=0.001,
                      reg_covar=1e-6,
                      max_iter=100,
                      n_init=1,
                      init_params='kmeans',
                      weights_init=None,
                      means_init=None,
                      precisions_init=None,
                      random_state=None,
                      warm_start=False,
                      verbose=0,
                      verbose_interval=10)

gmm.fit(x_train, y_train)
y_pred = gmm.predict(x_test)
score_gmm = gmm.score(x_test,y_test)
print('The accuracy of the Gaussian Mixture Models is', score_gmm)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



kmeans = KMeans(n_clusters=2, algorithm='auto', init='k-means++',n_init=10, max_iter=100, precompute_distances='auto')

kmeans.fit(x_train, y_train)
y_pred = kmeans.predict(x_test)
score_kmeans = kmeans.score(x_test,y_test)
print('The accuracy of the K Means Model is', score_kmeans)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



#################

# Binary Model
# copy newdf3 to newdf4 to preserve newdf3 if required
newdf5 = newdf3

newdf5 = newdf5.drop('CMPLNT_NUM', axis=1)
#newdf5 = newdf4.drop('KY_CD', axis=1)
newdf5 = newdf5.drop('OFNS_DESC', axis=1)
#newdf5 = newdf5.drop('PD_CD', axis=1)
newdf5 = newdf5.drop('PD_DESC', axis=1)
newdf5 = newdf5.drop('Lat_Lon', axis=1)

newdf5 = newdf5.drop('CMPLNT_FR_DT', axis=1)
newdf5 = newdf5.drop('CMPLNT_FR_TM', axis=1)
newdf5 = newdf5.drop('CMPLNT_TO_DT', axis=1)
newdf5 = newdf5.drop('CMPLNT_TO_TM', axis=1)
newdf5 = newdf5.drop('RPT_DT', axis=1)

#newdf5 = newdf5.drop('Latitude', axis=1)
#newdf5 = newdf5.drop('Longitude', axis=1)

newdf5.info()

newdf5.KY_CD_VNV = newdf5.KY_CD_VNV.replace([1,2,3,4,5],1)
#target variable
target2 = newdf5.KY_CD_VNV

# remove target from dataframe
newdf5 = newdf5.drop('KY_CD_VNV', axis=1)


####

print(newdf5.shape,target2.shape)

x_train2, x_test2, y_train2, y_test2 = train_test_split(newdf5, target2, test_size=0.2)

####################

#  Gaussian Mixture Models Method
from sklearn.mixture import GaussianMixture

gmm2 = GaussianMixture(n_components=2,
                      covariance_type='full',
                      tol=0.001,
                      reg_covar=1e-6,
                      max_iter=100,
                      n_init=1,
                      init_params='kmeans',
                      weights_init=None,
                      means_init=None,
                      precisions_init=None,
                      random_state=None,
                      warm_start=False,
                      verbose=0,
                      verbose_interval=10)

gmm2.fit(x_train2, y_train2)
y_pred2 = gmm2.predict(x_test2)
score_gmm2 = gmm2.score(x_test2,y_test2)
print('The accuracy of the Gaussian Mixture Models is', score_gmm2)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test2, y_pred2))
print(classification_report(y_test2, y_pred2))


######################################

essentialdf = pd.get_dummies(newdf, 
                        columns=['KY_CD','PD_CD'],
                        prefix=['KY_CD','PD_CD'])

essentialdf.info() 
essentialdf = essentialdf.drop('ADDR_PCT_CD',axis=1)
essentialdf = essentialdf.drop('BORO_NM',axis=1)
essentialdf = essentialdf.drop('CRM_ATPT_CPTD_CD',axis=1)
essentialdf = essentialdf.drop('HADEVELOPT',axis=1)
essentialdf = essentialdf.drop('JURIS_DESC',axis=1)
essentialdf = essentialdf.drop('LAW_CAT_CD',axis=1)
essentialdf = essentialdf.drop('LOC_OF_OCCUR_DESC',axis=1)
essentialdf = essentialdf.drop('PARKS_NM',axis=1)
essentialdf = essentialdf.drop('PREM_TYP_DESC',axis=1)
#'ADDR_PCT_CD', 'BORO_NM', 'CRM_ATPT_CPTD_CD', 'HADEVELOPT', 'JURIS_DESC', 'LAW_CAT_CD', 'LOC_OF_OCCUR_DESC', 'PARKS_NM', 'PREM_TYP_DESC'

########################################


# Binary Model
# copy newdf3 to newdf4 to preserve newdf3 if required
newdf6 = essentialdf

newdf6 = newdf6.drop('CMPLNT_NUM', axis=1)
#newdf6 = newdf6.drop('KY_CD', axis=1)
newdf6 = newdf6.drop('OFNS_DESC', axis=1)
#newdf6 = newdf6.drop('PD_CD', axis=1)
newdf6 = newdf6.drop('PD_DESC', axis=1)
newdf6 = newdf6.drop('Lat_Lon', axis=1)

newdf6 = newdf6.drop('CMPLNT_FR_DT', axis=1)
newdf6 = newdf6.drop('CMPLNT_FR_TM', axis=1)
newdf6 = newdf6.drop('CMPLNT_TO_DT', axis=1)
newdf6 = newdf6.drop('CMPLNT_TO_TM', axis=1)
newdf6 = newdf6.drop('RPT_DT', axis=1)

#newdf = newdf6.drop('Latitude', axis=1)
#newdf6 = newdf6.drop('Longitude', axis=1)

newdf6.info()

newdf6.KY_CD_VNV = newdf6.KY_CD_VNV.replace([1,2,3,4,5],1)
#target variable
target3 = newdf6.KY_CD_VNV

# remove target from dataframe
newdf6 = newdf6.drop('KY_CD_VNV', axis=1)


####

print(newdf6.shape,target3.shape)

x_train3, x_test3, y_train3, y_test3 = train_test_split(newdf6, target3, test_size=0.2)

####################

#  Gaussian Mixture Models Method
from sklearn.mixture import GaussianMixture

gmm3 = GaussianMixture(n_components=2,
                      covariance_type='full',
                      tol=0.001,
                      reg_covar=1e-6,
                      max_iter=100,
                      n_init=1,
                      init_params='kmeans',
                      weights_init=None,
                      means_init=None,
                      precisions_init=None,
                      random_state=None,
                      warm_start=False,
                      verbose=0,
                      verbose_interval=10)

gmm3.fit(x_train3, y_train3)
y_pred3 = gmm3.predict(x_test3)
score_gmm3 = gmm3.score(x_test3,y_test3)
print('The accuracy of the Gaussian Mixture Models is', score_gmm3)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test3, y_pred3))
print(classification_report(y_test3, y_pred3))

############################################################################################

# GMM Fit Predict Points for Clustering; 6 Class Model

# fit gmm object to data
gmm.fit(newdf4)

# save new clusters for chart
y_gmm1 = gmm.fit_predict(newdf4)

#aic
aic1 = gmm.aic(newdf4)
print('The AIC score of the Gaussian Mixture Model is ', aic1)
#bic
bic1 = gmm.bic(newdf4)
print('The BIC score of the Gaussian Mixture Model is', bic1)

################

# GMM Fit Predict Points for Clustering; 2 Class Model with all features

# fit gmm object to data
gmm2.fit(newdf5)

# save new clusters for chart
y_gmm2 = gmm2.fit_predict(newdf5)

#aic
aic2 = gmm2.aic(newdf5)
print('The AIC score of the Gaussian Mixture Model is', aic2)
#bic
bic2 = gmm2.bic(newdf5)
print('The AIC score of the Gaussian Mixture Model is', bic2)

##############

# GMM Fit Predict Points for Clustering; 2 Class Model with 2 features hot encoded

# fit gmm object to data
gmm3.fit(newdf6)

# save new clusters for chart
y_gmm3 = gmm3.fit_predict(newdf6)

#aic
aic3 = gmm3.aic(newdf6)
print('The AIC score of the Gaussian Mixture Model is', aic3)
#bic
bic3 = gmm3.bic(newdf6)
print('The AIC score of the Gaussian Mixture Model is', bic3)

##############
# Export clustering results from the 3 models

np.savetxt('y_gmm1.csv',y_gmm1, delimiter=',')
np.savetxt('y_gmm2.csv',y_gmm2, delimiter=',')
np.savetxt('y_gmm3.csv',y_gmm3, delimiter=',')



##################
