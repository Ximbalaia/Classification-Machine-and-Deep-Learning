library(tidyverse)
library(summarytools)
library(RColorBrewer)
library(gridExtra)
library(e1071)
library(caTools)
library(caret)
library(class)
library(rpart)
library(randomForest)
library(h2o)
library(xgboost)


train = read.csv('train.csv' ,na.strings = c("", "NA"))
test = read.csv('test.csv', na.strings = c("", "NA"))
test_values = read.csv('sample_submission.csv')
test$Transported = NA

#Data desciption
#PassengerId - A unique id for each passenger, Each id takes the form gggg_pp where gggg indicates a group the passenger is travelling
#with and pp is the number within the group. People in a group are often family members, but not always.

#HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.

#CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in 
#cryosleep are confined to their cabins.

#Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.

#Destination - The planet the passenger will be debarking to.

#Age - The age of the passenger.

#VIP -  Whether the passenger has paid for special VIP service during the voyage.

#RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.

#Name - The first and last names of the passenger.

#Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

merged = rbind(train, test)

#First we need to know many missing variable are in the dataset
NAcol = colnames(merged)[colSums(is.na(merged))>0]
sort(colSums(sapply(merged[NAcol], is.na)), decreasing = T) #columns who have the most missing values

#Fix it
#CryoSleep
table(merged$CryoSleep) #see how the variable is distributed
merged$CryoSleep[is.na(merged$CryoSleep)] = names(sort(-table(merged$CryoSleep))) [1]

#ShoppingMall
table(merged$ShoppingMall)
merged$ShoppingMall[is.na(merged$ShoppingMall)] = median(merged$ShoppingMall, na.rm = T)

#Cabin
table(merged$Cabin)
#When we analyze the Cabin column, it's difficult not to drop the variable, but if we check the description of this variable we can see that the 
#first letter means the deck of this passenger, and the last letter means what side of the passenger cabin (Port - left, Starboard - right). I'll
#then, create two new variables called CabinDeck and CabinSide, before handle the missing values.
CabinDeck = merged$Cabin %>% str_replace("/.*", "")
merged$CabinDeck = CabinDeck

CabinSide = merged$Cabin %>% str_replace(".*/", "")
merged$CabinSide = CabinSide

#CabinDeck
merged$CabinDeck[is.na(merged$CabinDeck)] = median(merged$CabinDeck, na.rm = T)

#CabinSide 
table(merged$CabinSide) #Maybe was logical to think that each side of the spaceship would have the same number of cabins, but like i don't know the
#architecture of the ship, it's better to fill the missing values with the median.
merged$CabinSide[is.na(merged$CabinSide)] = median(merged$CabinSide, na.rm = T)

#CabinShared 
merged$CabinShared = merged$Cabin %in% unique(merged$Cabin[duplicated(merged$Cabin, incomparables = NA)]) #I created a variable called CabinShared for
#duplicated Cabin values (including deck/num/side). I'm also going to create a variable for family members based on surname, that perhaps could 
#replace this variable that i just created. But some persons on the dataset that didn't shared the same surname was sharing cabins.

#We don't need the Cabin columns anymore, so it's time to drop it.
merged$Cabin = NULL

#VIP
merged$VIP[is.na(merged$VIP)] = names(sort(-table(merged$VIP))) [1]

#Name
#Some LastNames have a lot of observations, like Dickley, with 16. Analyzing this name we can that some of them stay at completely different CabinSide and CabinDeck, that so
#i consider that some of them is not part of the same family. Furthermore, the PassengerID in some SurNames is far to be continuous, another indication that they are not part 
#of the same family.
#merged = merged %>% separate(Name, c("FirstName", "SurName"), " ") %>% select(-"FirstName")
#merged$InFamily = merged$SurName %in% unique(merged$SurName[duplicated(merged$SurName)]) & merged$CabinShared == T
merged$Name = NULL

#FoodCourt
merged$FoodCourt[is.na(merged$FoodCourt)] = median(merged$FoodCourt, na.rm = T)

#HomePlanet
merged$HomePlanet[is.na(merged$HomePlanet)] = names(sort(-table(merged$HomePlanet))) [1]

#Spa
merged$Spa[is.na(merged$Spa)] = median(merged$Spa, na.rm = T)

#Destination
merged$Destination[is.na(merged$Destination)] = names(sort(-table(merged$Destination))) [1]

#Age
table(merged$Age) #Instead of use the median to fill the missing values, i did chose to fill them stochastically.
set.seed(1)
merged$Age[is.na(merged$Age)] = round(runif(sum(is.na(merged$Age)), min = 0, max = 79),0)

#VRDeck
merged$VRDeck[is.na(merged$VRDeck)] = median(merged$VRDeck, na.rm = T)

#RoomService
merged$RoomService[is.na(merged$RoomService)] = median(merged$RoomService, na.rm = T)

NAcol = colnames(merged)[colSums(is.na(merged))>0]
sort(colSums(sapply(merged[NAcol], is.na)), decreasing = T) #check for NA values again
#Only Transported column still have NA values, so the data is almost ready.

#EDA
train_x = merged[1:8693, ]
test_x = merged[8694:12970, ]

my_colors <- RColorBrewer::brewer.pal(4, "Set1")[2:4]

#HomePlanet by Transported
ggplot(train_x, aes(HomePlanet, fill = Transported)) + 
  geom_bar() +
  scale_fill_manual(values = my_colors, labels = c('False', 'True')) +
  ggtitle('Bar Plot - HomePlanet (Transported or Not)') + theme(plot.title = element_text(hjust = 0.5)) +
  ylab('Number of Obs.')

#CryoSleep
ggplot(train_x, aes(CryoSleep, fill = Transported)) + 
  geom_bar() +
  scale_fill_manual(values = my_colors, labels = c('False', 'True')) +
  ggtitle('Bar Plot - CryoSleep (Transported or Not)') + theme(plot.title = element_text(hjust = 0.5)) +
  ylab('Number of Obs.')

#Destination
ggplot(train_x, aes(Destination, fill = Transported)) + 
  geom_bar() +
  scale_fill_manual(values = my_colors, labels = c('False', 'True')) +
  ggtitle('Bar Plot - Destination (Transported or Not)') + theme(plot.title = element_text(hjust = 0.5)) +
  ylab('Number of Obs.')

#Age
TransportedF = train_x %>% filter(Transported == 'False')
TransportedT = train_x %>% filter(Transported == 'True')

TplotT = ggplot(TransportedT, aes(Age, fill = Transported)) + 
  geom_histogram(aes(y = ..density..), bins = 80) + 
  geom_density(alpha = 0.2) 

TplotF = ggplot(TransportedF, aes(Age, fill = Transported)) + 
  geom_histogram(aes(y = ..density..), bins = 80) + scale_fill_manual(values = my_colors) + 
  geom_density(alpha = 0.2) 

grid.arrange(TplotT, TplotF, top = 'Comparison of Ages (Transported or Not)')

#VIP
ggplot(train_x, aes(VIP, fill = Transported)) + 
  geom_bar() +
  scale_fill_manual(values = my_colors, labels = c('False', 'True')) +
  ggtitle('Bar Plot - VIP (Transported or Not)') + theme(plot.title = element_text(hjust = 0.5)) +
  ylab('Number of Obs.')

#CabinDeck
ggplot(train_x, aes(CabinDeck, fill = Transported)) + 
  geom_bar() +
  scale_fill_manual(values = my_colors, labels = c('False', 'True')) +
  ggtitle('Bar Plot - CabinDeck (Transported or Not)') + theme(plot.title = element_text(hjust = 0.5)) +
  ylab('Number of Obs.') + 
  scale_y_continuous(breaks = seq(0, 3200, by = 200))

#CabinSide
ggplot(train_x, aes(CabinSide, fill = Transported)) + 
  geom_bar() +
  scale_fill_manual(values = my_colors, labels = c('False', 'True')) +
  ggtitle('Bar Plot - CabinSide (Transported or Not)') + theme(plot.title = element_text(hjust = 0.5)) +
  ylab('Number of Obs.') + 
  scale_y_continuous(breaks = seq(0, 5000, by = 500))

#CabinShared
ggplot(train_x, aes(CabinShared, fill = Transported)) + 
  geom_bar() +
  scale_fill_manual(values = my_colors, labels = c('False', 'True')) +
  ggtitle('Bar Plot - CabinShared (Transported or Not)') + theme(plot.title = element_text(hjust = 0.5)) +
  ylab('Number of Obs.') + 
  scale_y_continuous(breaks = seq(0, 5000, by = 500))

ggplot(train_x, aes(CabinShared, fill = Transported)) + 
  geom_bar() +
  scale_fill_manual(values = my_colors, labels = c('False', 'True')) +
  ggtitle('Bar Plot - CabinSide (Transported or Not)') + theme(plot.title = element_text(hjust = 0.5)) +
  ylab('Number of Obs.') + 
  scale_y_continuous(breaks = seq(0, 5000, by = 500)) +
  facet_grid(rows = vars(Destination), cols = vars(CryoSleep))

#Tables
#HomePlanet
ctable(train_x$HomePlanet, train_x$Transported,  dnn = c('HomePlanet', 'Transported'))

#CryoSleep
ctable(train_x$CryoSleep, train_x$Transported,  dnn = c('CryoSleep', 'Transported'))

#Destination
ctable(train_x$Destination, train_x$Transported,  dnn = c('Destination', 'Transported'))

#VIP
ctable(train_x$VIP, train_x$Transported,  dnn = c('VIP', 'Transported'))

#CabinDeck
ctable(train_x$CabinDeck, train_x$Transported,  dnn = c('CabinDeck', 'Transported'))

#CabinSide
ctable(train_x$CabinSide, train_x$Transported,  dnn = c('CabinSide', 'Transported'))

#CabinShared
ctable(train_x$CabinShared, train_x$Transported,  dnn = c('CabinShared', 'Transported'))

#CabinDeck x HomePlanet
ctable(train_x$CabinDeck, train_x$HomePlanet,  dnn = c('CabinDeck', 'HomePlanet'))

#CabinDeck x Destination
ctable(train_x$CabinDeck, train_x$Destination,  dnn = c('CabinDeck', 'Destination'))

#Destination x HomePlanet
ctable(train_x$Destination, train_x$HomePlanet,  dnn = c('Destination', 'HomePlanet'))

#Label Columns
str(merged)
#HomePlanet
merged$HomePlanet = factor(merged$HomePlanet, levels = c('Earth', 'Europa', 'Mars'),
                           labels = c(0, 1, 2))

#CryoSleep
merged$CryoSleep = factor(merged$CryoSleep, levels = c('False', 'True'), labels = c(0, 1))

#Destination
merged$Destination = factor(merged$Destination, levels = c('TRAPPIST-1e', '55 Cancri e', 'PSO J318.5-22'),
                            labels = c(0, 1, 2))

#VIP
merged$VIP = factor(merged$VIP, levels = c('False', 'True'), labels = c(0, 1))

#CabinDeck
merged$CabinDeck = factor(merged$CabinDeck, levels = c('G', 'F', 'E', 'D', 'C', 'B', 'A', 'T'),
                          labels = c(0, 1, 2, 3, 4, 5, 6, 7))

#CabinSide
merged$CabinSide = factor(merged$CabinSide, levels = c('S', 'P'), labels = c(0, 1))

#CabinShared
merged$CabinShared = factor(merged$CabinShared, levels = c('FALSE', 'TRUE'), labels = c(0, 1))

#Transported
merged$Transported = factor(merged$Transported, levels = c('False', 'True'), labels = c(0, 1))

merged$PassengerId = NULL

str(train_x)
train_x = merged[1:8693, ]
test_x = merged[8694:12970, ]
test_x$Transported = test_values$Transported

#Make numeric variables factors
#RoomService
train_x[train_x$RoomService == 0, ]$RoomService = 0 
train_x[train_x$RoomService >= 1, ]$RoomService = 1 
train_x$RoomService = as.factor(train_x$RoomService)

#FoodCourt
train_x[train_x$FoodCourt == 0, ]$FoodCourt = 0
train_x[train_x$FoodCourt >= 1, ]$FoodCourt = 1 
train_x$FoodCourt = as.factor(train_x$FoodCourt)

#ShoppingMall
train_x[train_x$ShoppingMall == 0, ]$ShoppingMall = 0
train_x[train_x$ShoppingMall >= 1, ]$ShoppingMall = 1 
train_x$ShoppingMall = as.factor(train_x$ShoppingMall)

#Spa
train_x[train_x$Spa == 0, ]$Spa = 0
train_x[train_x$Spa >= 1, ]$Spa = 1 
train_x$Spa = as.factor(train_x$Spa)

#VRDeck
train_x[train_x$VRDeck == 0, ]$VRDeck = 0
train_x[train_x$VRDeck >= 1, ]$VRDeck = 1 
train_x$VRDeck = as.factor(train_x$VRDeck)

#Age
train_x$AgeBand = cut(train_x$Age, breaks = 4)
train_x$Age[train_x$Age <= 19.8] = 0
train_x$Age[train_x$Age > 19.8 & train_x$Age <= 39.5] = 1
train_x$Age[train_x$Age > 39.5 & train_x$Age <= 59.2] = 2
train_x$Age[train_x$Age > 59.2] = 3
train_x$Age = as.factor(train_x$Age)
train_x$AgeBand = NULL

#RoomService
ctable(train_x$RoomService, train_x$Transported,  dnn = c('RoomService', 'Transported'))

#FoodCourt
ctable(train_x$FoodCourt, train_x$Transported,  dnn = c('FoodCourt', 'Transported'))

#ShoppingMall
ctable(train_x$ShoppingMall, train_x$Transported,  dnn = c('ShoppingMall', 'Transported'))

#Spa
ctable(train_x$Spa, train_x$Transported,  dnn = c('Spa', 'Transported'))

#VRDeck
ctable(train_x$VRDeck, train_x$Transported,  dnn = c('VRDeck', 'Transported'))

#Age
ctable(train_x$Age, train_x$Transported,  dnn = c('Age', 'Transported'))

#Split
set.seed(1)
sample = sample.split(train_x$Transported, SplitRatio = 0.7)
train_X = subset(train_x, sample == T)
test_X = subset(train_x, sample == F)

#Logistic Regression
log.fit = glm(Transported~., data = train_X, family = 'binomial')
prob.log = predict(log.fit, type = 'response', newdata = test_X)
pred.log = ifelse(prob.log > 0.5, 1, 0) %>% as.factor()
matrix.log = table(test_X$Transported, pred.log)
matrix.log

acc.log = (matrix.log[1, 1] + matrix.log[2, 2]) / (matrix.log[1, 1] + matrix.log[1, 2] + matrix.log[2, 1] + matrix.log[2, 2])
acc.log

#SVM - Linear Kernel
svm.fit = svm(Transported ~ ., data = train_X, type = 'C-classification', kernel = 'linear')
pred.svm = predict(svm.fit, newdata = test_X)
matrix.svm = table(test_X$Transported, pred.svm)
matrix.svm

acc.svm = (matrix.svm[1, 1] + matrix.svm[2, 2]) / (matrix.svm[1, 1] + matrix.svm[1, 2] + matrix.svm[2, 1] + matrix.svm[2, 2])
acc.svm

#SVM - Radial Kernel
svm.fit2 = svm(Transported ~ ., data = train_X, type = 'C-classification', kernel = 'radial')
pred.svm2 = predict(svm.fit2, newdata = test_X)
matrix.svm2 = table(test_X$Transported, pred.svm2)
matrix.svm2

acc.svm2 = (matrix.svm2[1, 1] + matrix.svm2[2, 2]) / (matrix.svm2[1, 1] + matrix.svm2[1, 2] + matrix.svm2[2, 1] + matrix.svm2[2, 2])
acc.svm2

#Naive Bayes
nb.fit = naiveBayes(x = train_X[, -11],
                    y = train_X$Transported)
pred.nb = predict(nb.fit, newdata = test_X)
matrix.nb = table(test_X$Transported, pred.nb)
matrix.nb

acc.nb = (matrix.nb[1, 1] + matrix.nb[2, 2]) / (matrix.nb[1, 1] + matrix.nb[1, 2] + matrix.nb[2, 1] + matrix.nb[2, 2])
acc.nb

#K-NN
knn.fit = knn(train = train_X[, -11],
              test = test_X[, -11],
              cl = train_X[, 11],
              k = 5)
matrix.knn = table(test_X$Transported, knn.fit)
matrix.knn

acc.knn = (matrix.knn[1, 1] + matrix.knn[2, 2]) / (matrix.knn[1, 1] + matrix.knn[1, 2] + matrix.knn[2, 1] + matrix.knn[2, 2])
acc.knn

#Decision Tree
dt.fit = rpart(Transported ~., data = train_X)
pred.dt = predict(dt.fit, newdata = test_X, type = 'class') %>% as.factor()
matrix.dt = table(test_X$Transported, pred.dt)
matrix.dt

acc.dt = (matrix.dt[1, 1] + matrix.dt[2, 2]) / (matrix.dt[1, 1] + matrix.dt[1, 2] + matrix.dt[2, 1] + matrix.dt[2, 2])
acc.dt

#Random Forest
rf.fit = randomForest(x = train_X[, -11], y = train_X$Transported, ntree = 300)
pred.rf = predict(rf.fit, newdata = test_X)
matrix.rf = table(test_X$Transported, pred.rf)
matrix.rf

acc.rf = (matrix.rf[1, 1] + matrix.rf[2, 2]) / (matrix.rf[1, 1] + matrix.rf[1, 2] + matrix.rf[2, 1] + matrix.rf[2, 2])
acc.rf

#Accuracy DF
Accuracy = data.frame(Model = c('Logistic Regression', 'SVM - Linear', 'SVM - Radial', 'Naive Bayes', 'K-NN', 'Decision Tree', 'Random Forest'),
                      Accuracy = c(acc.log, acc.svm, acc.svm2, acc.nb, acc.knn, acc.rf, acc.dt))

Accuracy

##ANN
#train_x$HomePlanet = as.numeric(train_x$HomePlanet)
#train_x$CryoSleep = as.numeric(train_x$CryoSleep)
#train_x$Destination = as.numeric(train_x$Destination)
#train_x$Age = as.numeric(train_x$Age)
#train_x$VIP = as.numeric(train_x$VIP)
#train_x$RoomService = as.numeric(train_x$RoomService)
#train_x$FoodCourt = as.numeric(train_x$FoodCourt)
#train_x$ShoppingMall = as.numeric(train_x$ShoppingMall)
#train_x$Spa = as.numeric(train_x$Spa)
#train_x$VRDeck = as.numeric(train_x$VRDeck)
#train_x$CabinDeck = as.numeric(train_x$CabinDeck)
#train_x$CabinSide = as.numeric(train_x$CabinSide)
#train_x$CabinShared = as.numeric(train_x$CabinShared)


set.seed(1)
#sample = sample.split(train_x$Transported, SplitRatio = 0.7)
#train_X = subset(train_x, sample == T)
#test_X = subset(train_x, sample == F)


#train_X[-11] = scale(train_X[-11])
#test_X[-11] = scale(test_X[-11])

h2o.init(nthreads = -1)

ann.fit = h2o.deeplearning(y = 'Transported', training_frame = as.h2o(train_X),
                              activation = 'TanhWithDropout', hidden = c(7, 7),
                              epochs = 100, train_samples_per_iteration = -2)


prob.ann = h2o.predict(ann.fit, newdata = as.h2o(test_X))
pred.ann = prob.ann$predict %>% as.vector
matrix.ann = table(test_X$Transported, pred.ann)
matrix.ann

acc.ann = (matrix[1, 1] + matrix[2, 2]) / (matrix[1, 1] + matrix[1, 2] + matrix[2, 1] + matrix[2, 2])
acc.ann

#Predicting the Test set Results

pred.ann = (prob.ann > 0.5) %>% as.vector()
y_pred_factor = as.factor(y_pred)

confusionMatrix(y_pred_factor, test_x$Exited)

h2o.shutdown()



prob.ann > 0.5
prob.ann$predict

