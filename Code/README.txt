AIT 690 Smart Cities and Urban Computing
Final Project By Japneet S. Kohli

***************************************************************************************************************************************************************************************************************
Methodology Approach 1
***************************************************************************************************************************************************************************************************************
Method: Supervised Learning Using GMM Algorithm to Classify Crimes in Heuristically Determined Violence Labels
Code File: AIT_690_Final_Project_Code_1st_Approach.py

***************************************************************************************************************************************************************************************************************
Order of Execution of associated code:
***************************************************************************************************************************************************************************************************************

1.  Import libraries and set working directory.
2.  Read data.

*** Multi-Label Violence Classification ***

3.  Add new column to determine violent vs non violent crimes based on key code values. This determination has total 6 class labels.
4.  Keep subset of categorical attributes to be used as training data.
5.  Convert string, dates, and other types to numerical type by one hot encoding.
6.  Identify columns to remove from final analysis and drop them.
7.  Create standalone target variable and assign violent vs.non-violent values to it.
8.  Remove target variable from dataframe as standalone will be used going forward.
9.  Split data into train and test sets. 
10. Define GMM model and its parameters.
11. Fit the GMM model with training data.
12. Make predictions on test data.
13. Analyze the accuracy of the GMM model.
14. Create a confusion matrix and a classification report.
15. Create a kmeans model, fit training data, redict test data, and obtain accuracy score, confusion matrix and classification report. This is done only to see how the GMM model compares with kmeans.

*** Binary-Label Violence Classification ***

16. Drop attributes not to be used in training data.
17. Add new column to determine violent vs non violent crimes based on key code values. This determination has total only 2 class labels.
18. Create standalone target variable and assign violent vs.non-violent values to it.
19. Remove target variable from dataframe as standalone will be used going forward.
20. Split data into train and test sets. 
21. Define GMM model and its parameters.
22. Fit the GMM model with training data.
23. Make predictions on test data.
24. Analyze the accuracy of the GMM model.
25. Create a confusion matrix and a classification report.


*** Binary-Label Violence Classification With Fewer Training Attributes ***

26. Drop more categorical attributes not to be used in training data.
27. Add new column to determine violent vs non violent crimes based on key code values. This determination has total only 2 class labels.
28. Create standalone target variable and assign violent vs.non-violent values to it.
29. Remove target variable from dataframe as standalone will be used going forward.
30. Split data into train and test sets. 
31. Define GMM model and its parameters.
32. Fit the GMM model with training data.
33. Make predictions on test data.
34. Analyze the accuracy of the GMM model.
35. Create a confusion matrix and a classification report.

*** Predicting Cluster Values based on 3 GMM Models Created Above ***

36. For multi-label GMM model, predict cluster assignments for all data points and get AIC and BIC numbers.
37. For the first binary-label GMM model, predict cluster assignments for all data points and get AIC and BIC numbers.
38. For the second binary-label GMM model, predict cluster assignments for all data points and get AIC and BIC numbers.

*** Save Predictions ***

39. For the three models above, export clustering predictions out so they could be used in ArcMap.

***************************************************************************************************************************************************************************************************************
Methodology Approach 2
***************************************************************************************************************************************************************************************************************
Method: Unsupervised Learning Using GMM Algorithm to Identify Spatial Regions for Types or Patterns of Data
Code File: AIT_690_Final_Project_Code_2nd_Approach.ipynb

***************************************************************************************************************************************************************************************************************
Order of Execution of associated code:
***************************************************************************************************************************************************************************************************************

1.  Import required libraries and packages
2.  Mount Google Drive. Google Colab was used as the platform to run this code.
3.  Load NY Crime Complaint Data.

*** Data Cleaning ***

4.  Drop unnecessary columns from dataset and rename columns.
5.  Convert categorical column "Law Code" into dummies for each category and add to dataset. 
6.  Merge complain date and time columns into one datetime column and assign it as index. In this step, records outside 2019 year are assigned "None" value so they will be removed in the next step.
7.  Remove non-date values from the dataframe. This eliminates all bad dates and dates outside 2019.

*** Data Exploration ***

8.  Analyze the Law Code Categories for 2019 for annual frequency of crime.
9.  Analyze the Law Code Categories for 2019 for monthly frequency of crime.
10. Analyze the Law Code Categories for 2019 for day-wise frequency of crime.
11. Analyze the Law Code Categories for 2019 for hour-wise frequency of crime.
12. Create subset of data with only 4 types of categories, namely i. sex crimes; ii. murder and non-negligent manslaughter crimes; iii. robbery crimes; iv. dangerous drugs crimes.
13. Add columns for above 4 categories using dummy values.
14. Analyze all 4 categories based on hour-of-day frequencies to notice interesting patterns.
15. Analyze drugs related crime data to find proportion of marijuana and sale related crimes respectively.
16. Plot the types of crimes based on Offense Descriptions and annual frequency to visualize commonness of various types of crimes.
17. Create dataframes containing data for the interesting patterns observed thus far.

*** Implementing Gaussian Mixture Models ***

18. Define function to extract latitude/longitude coordinate data from interesting pattern dataframes.
19. Define function to analyze coord data extracted above using GMMs iterated over varying number of components and plot aic/bic curve for all models belonging to 1 dataframe. 
20. Define function to draw ellipse with given position and covariance.
21. Define function to plot a GMM with clusters shown within ellipses.
22. Run the above functions over the interesting pattern dataframes and select best models based on evaluation of aic/bic.
23. Export the interesting pattern dataframes' coordinates and cluster prediction values for best models selected to visualize them in ArcMap.


***************************************************************************************************************************************************************************************************************
***************************************************************************************************************************************************************************************************************
***************************************************************************************************************************************************************************************************************