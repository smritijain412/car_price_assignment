 # Car Price Prediction
 Car prices depend on many features like brand,model,segment fuel type,mileage and many more. Main task is to predict car price and find the best feature to predict car price.
 
 In this  assesment i need to draw some graphs which gives the information of dataset.
 
 ### A.priceUSD vs mileage(kilometers)
 ![price vs image](https://user-images.githubusercontent.com/61602017/99527133-4ef4ea80-29c2-11eb-8441-933417c38ecf.png)
 
 ### B.Groupby graph year,transmission vs price 
![year vs transmission](https://user-images.githubusercontent.com/61602017/99527229-7946a800-29c2-11eb-9715-5c6290dc1586.png)

### C.Plot a graph between make vs number of cars, here we can see which make or brand is highest value 
![graph1 make vs count](https://user-images.githubusercontent.com/61602017/99527408-c75bab80-29c2-11eb-89c2-776ab571a70f.png)

> On the basis of survey and study,I opt for some features and check the correlation between priceUSD and others.After the correlation method,delete some features and save in a     new dataframe.

![Screenshot (23)](https://user-images.githubusercontent.com/61602017/99539479-4c9b8c00-29d4-11eb-922d-56f4381689e4.png)

Split data set into train and test dataframe. And apply chisquare test which gives you chi value and p values.

After splitting the dataset i used linearregression,lasso,logistic and extra tree regressor method. And they give different scores.

![Screenshot (24)](https://user-images.githubusercontent.com/61602017/99539553-61781f80-29d4-11eb-9e20-41bc88000fa5.png)

#pull docker image to access the work 

https://hub.docker.com/repository/docker/smriti0412/car_price

> steps to pull and run docker image.>

#step1: docker pull smriti0412/car_price:latest

#step2: docker run -p 8888:8888 smriti0412/car_price:latest
