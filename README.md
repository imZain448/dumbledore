# Dumbledore
this is a python module which is intended for data scientist and statistician to visualize features and understand relationships between different features without writing lots of codes. <br>
This module is like an aggregator which combines multiple functionalities of different modules and brings them together to keep you out of trouble

------
----
__installation__
```
pip install dumbledore
```

__requirements__ <br>
this module depends on all most of scientific libraries like `pandas` , `matplotlib` , `seaborn`, `numpy` etc. all of which you must have already installed if you are a data scientist or statistician but don't worry if you haven't the pip command will automatically install all the dependencies.

__examples__ <br>
the  `vis_feature()` function helps you understand and visualize between different types of variables
1. for a continuous feature and categorical target variable
```python
import dumbledore as db
import pandas as pd
...
# load data
...
db.basics.vis_feature((data , 'gender' , 'tenure', 'gist_earth_r' , target_continous=True , jitter=0.3)
```
![data](https://github.com/imZain448/dumbledore/blob/master/images/data1.png?raw=true)
![plot](https://github.com/imZain448/dumbledore/blob/master/images/plot1.png?raw=true)

1. when your feature and target both are continous
```python
db.basics.vis_feature((data , 'tenure' , 'MonthlyCharges', 'gist_earth_r' , continous=True , target_continous=True )
```
![data2](https://github.com/imZain448/dumbledore/blob/master/images/data2.png?raw=true)
![plot2](https://github.com/imZain448/dumbledore/blob/master/images/plot2.png?raw=true)

similarly you can feed in categorical target or categorical feature you have to keep in mind the two arguements
>__continous__ : True if feature is continous, False if feature categorical<br> 
>__target_continous__ : True if target variable is continous , False if target variable is categorical


----
__note__ : this module is still under development and has been released as alpha. so if you get any bug please open a issue without hesitation. or if you want a new feature please make a feature request

> _THIS PRODUCT IS DISTRIBUTED UNDER GNU GPLv3 WHICH CAN BE FOUND IN THE LICENSE FILE_

__developed by__ : [imzain448](https://github.com/imZain448) <br>
meet me on
- [Linkedin](https://www.linkedin.com/in/zain-ahmad-15aa25162/)
- [Instagram](https://www.instagram.com/imzain448/?hl=en)
- [Gmail : ahmadzain.448@gmail.com]

