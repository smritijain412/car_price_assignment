 # Car Price Prediction
 Car prices depend on many features like brand,model,segment fuel type,mileage and many more. Main task is to predict car price and find the best feature to predict car price.
 
 In this  assesment i need to draw some graphs which gives the information of dataset.
 
 ## A.priceUSD vs mileage(kilometers)
 ![price vs image](https://user-images.githubusercontent.com/61602017/99527133-4ef4ea80-29c2-11eb-8441-933417c38ecf.png)
 
 ## B.Groupby graph year,transmission vs price 
![year vs transmission](https://user-images.githubusercontent.com/61602017/99527229-7946a800-29c2-11eb-9715-5c6290dc1586.png)

## C.Plot a graph between make vs number of cars, here we can see which make or brand is highest value 
![graph1 make vs count](https://user-images.githubusercontent.com/61602017/99527408-c75bab80-29c2-11eb-89c2-776ab571a70f.png)

On the basis of survey and study,I opt for some features and check the correlation between priceUSD and others.After the correlation method,delete some features and save in a new dataframe.

![Screenshot (21)](https://user-images.githubusercontent.com/61602017/99539067-c8e19f80-29d3-11eb-8115-2d7407d614b1.png)

Split data set into train and test dataframe. And apply chisquare test which gives you chi value and p values.

After splitting the dataset i used linearregression,lasso,logistic and extra tree regressor method. And they give different scores.

![Screenshot (22)](https://user-images.githubusercontent.com/61602017/99539155-e57dd780-29d3-11eb-8a56-d2da57aca2b6.png)




