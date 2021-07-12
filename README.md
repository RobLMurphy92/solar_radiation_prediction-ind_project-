# Individual Project - Predicting Solar Radiation.
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

#### Abstract
The availability of Fossil Fuels is continuing to deplete due to hydrocarbons being finite. In the future as the renewable industry continues to grow it will eventually be necessary for individuals to predict when Solar Radiation levels will be adequate to utilize for daily use or industrial use.
The main focus of this project is to see if with certain features a model can be created which accuratley predicts solar radiation levels.
Utilizing a regression model, select features which had a correlation value greater than 20% wre utilized within a Tweedie Regressor model to predict radiation levels. The model peformed significantly better than the baseline and was proven valid.




#### Project Objectives
> - Document code, process (data acquistion, preparation, exploratory data analysis utilizing clustering and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook report.
> - Create a Model which can accurately predicts Solar Radiation.




#### Goals
> - Find features which can predict Radiation Levels
> - Construct a model that accurately predicts Raditation levels
> - Document your process well enough to be presented or read like a report.
> - Create final notebook which will include code and markdowns.



#### Audience
> - Audience is any individual looking into the project.


#### Project Deliverables
> - A clearly named final notebook. This notebook will be what you present and should contain plenty of markdown documentation and cleaned up code.
> - A README that explains what the project is, how to reproduce you work, and your notes from project planning.
> - A Python module or modules that automates the data acquisistion and preparation process. These modules should be imported and used in your final notebook.




<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

#### Data Dictionary

|Target|Datatype|Definition|
|:-------|:--------|:----------|
| Radiation| float64 | 

|Feature|Datatype|Definition|
|:-------|:--------|:----------|
|UNIXTime| 32686 non-null  int64 | 
|Date| 32686 non-null datetime64[ns] | date in yyyy-mm-dd format |
|Time| 32686 non-null  object | 24 Hour format |
|Radiation| 32686 non-null  float64 | watts per meter^2 |
|Temperature| 32686 non-null  int64 | temperature in Fahrenheit | 
|Pressure| 32686 non-null  float64 | barometric pressure Hg (inch of mercury) |
|Humidity| 32686 non-null  int64 | humidity percentage | 
|WindDirection(Degrees)| 32686 non-null  float64 | direction of wind in degrees |
|Speed| 32686 non-null  float64 | wind speed in mph |
|TimeSunRise| 32686 non-null  object |  When Sun rises in Hawaii, time (24Hr format) |
|TimeSunSet| 32686 non-null  object |  When Sun sets in Hawaii, time (24hr format) |
|day| 32686 non-null  int64 | Numeric based categorical 1 for day, 0 for not day(night) |       
|0.00-06.00| 32686 non-null  uint8 | time period between 0.00-06.00 (24hr format) |        
|06.00-10.00| 32686 non-null  uint8 | time period between 06.00-10.00 (24hr format) |          
|10.00-12.00| 32686 non-null  uint8 | time period between 10.00-12.00 (24hr format) |          
|12.00-14.00| 32686 non-null  uint8 | time period between 12.00-14.00 (24hr format) |          
|14.00-16.00| 32686 non-null  uint8 | time period between 14.00-16.00 (24hr format) |          
|16.00-18.00| 32686 non-null  uint8 | time period between 16.00-18.00 (24hr format) |          
|18.00--24.00| 32686 non-null  uint8 | time period between 18.00-24.00 (24hr format) |           
|E| 32686 non-null  uint8 | Numeric based categorical 1 for if wind blows from E, 0 if not |          
|N| 32686 non-null  uint8 | Numeric based categorical 1 for if wind blows from N, 0 if not |       
|NE| 32686 non-null  uint8 | Numeric based categorical 1 for if wind blows from NE, 0 if not |        
|NW| 32686 non-null  uint8 | Numeric based categorical 1 for if wind blows from NW, 0 if not |        
|S| 32686 non-null  uint8 | Numeric based categorical 1 for if wind blows from S, 0 if not |        
|SE| 32686 non-null  uint8 | Numeric based categorical 1 for if wind blows from SE, 0 if not |        
|SW| 32686 non-null  uint8 | Numeric based categorical 1 for if wind blows from SW, 0 if not |        
|W| 32686 non-null  uint8 | Numeric based categorical 1 for if wind blows from W, 0 if not | 




<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

#### Project Planning:

> - Create a README.md which contains a data dictionary, project objectives, business goals, initial hypothesis.
> - Acquire the solar dataset utilizing a csv obtained from the kaggle website.
> - Link to CSV: https://www.kaggle.com/dronio/SolarEnergy?select=SolarPrediction.csv
> - Prep work will be basic dropping of nulls and not worrying about outliers, will see if any datatypes need to be changed.
> - Investigate any missing values also.
> - Explore the dataset on unscaled data, the target variable will be radiation, will utilize univariate, bivariate and multivar.
> - Will utilize feature engineering to see which features will be useful in prediciton radiation level.
> - Target is continous so this will be a regression model.
> - Will utilize 4 models and compare the performance, 
> - Will evaluate on unscaled train, validate datasets.
> - Will evluate test on best performing model
> - Present findings and give a conclusion.
> - If time permits will go back and deal with outliers and scaling.




#### Hypotheses: 

> - **Hypothesis 1 -** I rejected the Null Hypothesis.
> - alpha = .05
> - Hypothesis Null : 'there is no linear correlation between Radiation and Humidity'.
> - Hypothesis Alternative : 'There is a relationship between Radiation and Humidity'.

> - **Hypothesis 2 -** I rejected the Null Hypothesis.
> - alpha = .05
> - Hypothesis Null : 'there is no linear correlation between Radiation and Pressure'.
> - Hypothesis Alternative : 'There is a relationship between Radiation and Pressure'.

> - **Hypothesis 3 -** I rejected the Null Hypothesis.
> - alpha = .05
> - Hypothesis Null : 'There is no difference in radiation levels for timeframe 10:00-12:00 and not in the timeframe 10:00-12:00'
> - Hypothesis Alternative : 'There is a difference in radiation levels for timeframe 10:00-12:00 and not in the timeframe 10:00-12:00'

> - **Hypothesis 4 -** I rejected the Null Hypothesis.
> - alpha = .05
> - Hypothesis Null : 'There is no significant difference in radiation levels for winds blowing North and the overal population mean.'
> - Hypothesis Alternative : "There is a significant difference in the logerrors for those who have four or more bedrooms than those who dont.


### Reproduce My Project:

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook. 
- [x] Read this README.md
- [ ] Download the SolarPredictions CSV from Kaggle: https://www.kaggle.com/dronio/SolarEnergy?select=SolarPrediction.csv
- [ ] Download the aquire.py, prepare.py, explore.py,  and final_notebook.ipynb files into your working directory
- [ ] Run the final_notebook.ipynb 



