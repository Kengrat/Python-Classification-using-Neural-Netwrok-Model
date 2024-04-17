# The NBA rookie performance dataset is deployed for this project and contains twenty-one variables which are the statistical data of each basketball player in NBA league,
# The variable are the numbers of games played, points per game, 3-point attempts, rebounds, assists, blocks, steals, turnovers, free throw attempt etc. 
# These data sets were analyzed using Neural network and also compared with variety of other machine-learning algorithms in Python programming 
# The aim is to predicts if a player will last 5 years in the NBA leaque. 
# The target variable in the dataset is the Target_5Yrs variable with a conditional statement that is 1: if career length >= 5 years. or 0: if career length < 5 years.
# Neural Networks models are used to compare with other variable in the dataset including Logistic Regression and Gausian Naive Baise for accuracy with the target variable. 
# The game played and point per game has the highest accuracy with the target and is used as the predictors for this analysis
# Games played and Points per game are the input while the target is Target_5Yrs with 0 as the < 5-year career length (Blue) while >= 5Yrs career length is 1 (Red color). 
# The neural network model finds a curve boundary between the red and blue colors, the mislabelled points are the white dots in the plot. 
# The accuracy is 69% while the mislabelled points are 104 out of 335, 
# more data points are plotted in the red section which denotes players with higher games played with higher points per game between 5 to 25 will have career length >= 5 years. 
# The players with fewer points per game are between 0 to 10 (low) points and lower games played between 20 to 50 games 
# Blue section are the players with low stats and will have a career length of less than < 5 years.
# Players on the red section are categorised with very good statistics, the upper categories will certainly have a career length greater than > 5 years in NBA.
