# Credit to YouTuber: David Langer (https://www.youtube.com/watch?v=32o0DnuRjfg&list=PLTJTBoU5HOCRrTs3cJK-PbHM39cwCU0PF)
# Watched the YouTube videos and typed it up myself (didn't copy, paste)
install.packages("ggplot2")
install.packages("stringr")
install.packages("randomForest")
install.packages("caret")
install.packages("doSNOW")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("infotheo")
install.packages("Rtsne")

# Load Raw Data
train <- read.csv("train.csv", header = TRUE)
test <- read.csv("test.csv", header = TRUE)

# Add a "Survived" variable to the test set to allow for combining data sets
test.survived <- data.frame(Survived = rep("None", nrow(test)), test[,])

# Combine data sets
data.combined <- rbind(train, test.survived)

# R data types
str(data.combined)

data.combined$Survived <- as.factor(data.combined$Survived)
data.combined$Pclass <- as.factor(data.combined$Pclass)

# Take a look at gross survival rates
table(data.combined$Survived)

# Distributions across classses
table(data.combined$Pclass)

# Load up ggplot2 package to use for visualizations
library(ggplot2)

# Hypothesis - Rich folks survived at a higher rate
train$Pclass <- as.factor(train$Pclass)
ggplot(train, aes(x = Pclass, fill = factor(Survived))) +
  geom_bar(width = 0.5) +
  xlab("Pclass") +
  ylab("Total Count") +
  labs(fill = "Survived")

# Exam the first few names in the training data set
head(as.character(train$Name))

# How many unique names are there across both train and test?
length(unique(as.character(data.combined$Name)))
# This gives 1307, but there should be 1309.
# So there must be two duplicates. Or somethings wrong. Let's take a closer look.

# First, get the duplicate names and store them as a vector
dup.names <- as.character(data.combined[which(duplicated(as.character(data.combined$Name))), "Name"])

# Next, take a look at the records in the combined data set
data.combined[which(data.combined$Name %in% dup.names),]

# Dealing with 'Miss.' and 'Mr.'
library(stringr)

# Any correlation with other variables (e.g. sibsp)?
misses <- data.combined[which(str_detect(data.combined$Name, "Miss.")),]
misses[1:5,]

# Hypothesis: Name titles correlate with age
mrses <- data.combined[which(str_detect(data.combined$Name, "Mrs.")),]
mrses[1:5,]

# Check males to see if pattern continues
males <- data.combined[which(train$Sex == "male"),]
males[1:5,]


# Expand upon the relationship between 'Survived' and 'Pclass' by adding a new 'Title' variable
# to the data set and then explore a potential 3-dimensional relationship.

# Create a utility function to help with the title extraction
extractTitle <- function(name) {
  name <- as.character(name)
  
  if(length(grep("Miss.", name)) > 0){
    return("Miss.")
  }
  else if(length(grep("Master.", name)) > 0){
    return("Master.")
  }
  else if(length(grep("Mrs.", name)) > 0){
    return("Mrs.")
  }
  else if(length(grep("Mr.", name)) > 0){
    return("Mr.")
  }
  else{
    return("Other")
  }
}

titles <- NULL
for(i in 1:nrow(data.combined)){
  titles <- c(titles, extractTitle(data.combined[i, "Name"]))
}
data.combined$Title <- as.factor(titles)

# Since we only have survived labels for the train set, only use the first 891 rows
ggplot(data.combined[1:891,], aes(x = Title, fill = Survived)) +
  geom_bar(width = 0.5) +
  facet_wrap(~Pclass) +
  ggtitle("Pclass") +
  xlab("Title") +
  ylab("Total Count") +
  labs(fill = "Survived")

# What's the distribution of females and males across train and test?
table(data.combined$Sex)

# Visualize the 3-way relationship of sex, pclass, and survival, compare to analysis
ggplot(data.combined[1:891,], aes(x = Sex, fill = Survived)) +
  geom_bar(width = 0.5) +
  facet_wrap(~Pclass) +
  ggtitle("Pclass") +
  xlab("Sex") +
  ylab("Total Count") +
  labs(fill = "Survived")

# We learned that 'Age' and 'Sex' is important as derived from analysis of 'Title'.
# Let's take a look at the distributions of age over entire data set.
summary(data.combined$Age)
summary(data.combined[1:891, "Age"])

# Just to be thorough, take a look at survival rates broken out of 'Sex', 'Pclass', and 'Age'
ggplot(data.combined[1:891,], aes(x = Age, fill = Survived)) +
  facet_wrap(~Sex + Pclass) +
  geom_histogram(binwidth = 10) +
  xlab("Age") +
  ylab("Total Count")

# Validating that 'Master.' is a good proxy for male children
boys <- data.combined[which(data.combined$Title == "Master."),]
summary(boys$Age)

# We know that 'Miss.' is more complicated. So let's examine further
misses <- data.combined[which(data.combined$Title == "Miss."),]
summary(misses$Age)

ggplot(misses[misses$Survived != "None",], aes(x = Age, fill = Survived)) +
  facet_wrap(~Pclass) +
  geom_histogram(binwidth = 5) +
  ggtitle("Age for 'Miss.' by Pclass") +
  xlab("Age") +
  ylab("Total Count")

# Appears female children may have different survival rate.
# This could be a candidate for feature engineering later.
misses.alone <- misses[which(misses$SibSp == 0 & misses$Parch == 0),]
summary(misses.alone$Age)
length(which(misses.alone$Age <= 14.5))


# Summarizing the 'SibSp' variable
summary(data.combined$SibSp)

# Check to see if it is reasonable to make 'SibSp' a factor
length(unique(data.combined$SibSp))

# Only 7, so let's treat 'SibSp' as a factor
data.combined$SibSp <- as.factor(data.combined$SibSp)

# We believe 'Title' is predictive. Let's visualize survival rates by 'SibSp', 'Pclass', and 'Title'
ggplot(data.combined[1:891,], aes(x = SibSp, fill = Survived)) +
  geom_bar(width = 1) +
  facet_wrap(~Pclass + Title) +
  ggtitle("Pclass, Title") +
  xlab("SibSp") +
  ylab("Total Count") +
  ylim(0, 300) +
  labs(fill = "Survived")

# Treat the 'Parch' variable as a factor and visualize
data.combined$Parch <- as.factor(data.combined$Parch)
ggplot(data.combined[1:891,], aes(x = Parch, fill = Survived)) +
  geom_bar(width = 1) +
  facet_wrap(~Pclass + Title) +
  ggtitle("Pclass, Title") +
  xlab("Parch") +
  ylab("Total Count") +
  ylim(0, 300) +
  labs(fill = "Survived")

# Let's try some feature engineering. Let's create a family size feature
temp.SibSp <- c(train$SibSp, test$SibSp)
temp.Parch <- c(train$Parch, test$Parch)
data.combined$Family.size <- as.factor(temp.SibSp + temp.Parch + 1)

ggplot(data.combined[1:891,], aes(x = Family.size, fill = Survived)) +
  geom_bar(width = 1) +
  facet_wrap(~Pclass + Title) +
  ggtitle("Pclass, Title") +
  xlab("Family.size") +
  ylab("Total Count") +
  ylim(0, 300) +
  labs(fill = "Survived")


# Take a look at the 'Ticket' variable
str(data.combined$Ticket)

# Based on the huge number of levels, 'Ticket' really isn't a factor variable. It is a string.
# Convert 'Ticket' to characters (strings), and display first 20.
data.combined$Ticket <- as.character(data.combined$Ticket)
data.combined$Ticket[1:20]

# There's no immediately apparent structure in the data, let's see if we can find some.
# We'll start with taking a look at just the first char for each 'Ticket'
Ticket.first.char <- ifelse(data.combined$Ticket == "", " ", substr(data.combined$Ticket, 1, 1))
unique(Ticket.first.char)

# There are only 16 unique 'Ticket.first.char, so make a factor for analysis purposes and visualize
data.combined$Ticket.first.char <- as.factor(Ticket.first.char)

# First, make a high-level plot of the data
ggplot(data.combined[1:891,], aes(x = Ticket.first.char, fill = Survived)) +
  geom_bar() +
  ggtitle("Survivability by Ticket.first.char") +
  xlab("Ticket.first.char") +
  ylab("Total Count") +
  ylim(0, 350) +
  labs(fill = "Survived")

# 'Ticket' seems like it might be predictive. Let's take a deeper look.
ggplot(data.combined[1:891,], aes(x = Ticket.first.char, fill = Survived)) +
  geom_bar() +
  facet_wrap(~Pclass) +
  ggtitle("Survivability by Ticket.first.char") +
  xlab("Ticket.first.char") +
  ylab("Total Count") +
  ylim(0, 150) +
  labs(fill = "Survived")

# Lastly, see if we get a pattern when using combinations of 'Pclass' & 'Title'
ggplot(data.combined[1:891,], aes(x = Ticket.first.char, fill = Survived)) +
  geom_bar() +
  facet_wrap(~Pclass + Title) +
  ggtitle("Pclass, Title") +
  xlab("Ticket.first.char") +
  ylab("Total Count") +
  ylim(0, 200) +
  labs(fill = "Survived")
# So, 'Tickets' wasn't that useful. (At least for now)


# Next: The fares Titanic passengers paid
summary(data.combined$Fare)
length(unique(data.combined$Fare))

# Can't make 'Fare' a factor, so treat as numeric and visualize with histogram
ggplot(data.combined, aes(x = Fare)) +
  geom_histogram(binwidth = 5) +
  ggtitle("Combined Fare Distribution") +
  xlab("Fare") +
  ylab("Total Count") +
  ylim(0, 200)

# Let's check to see if 'Fare' has any predictive power
ggplot(data.combined[1:891,], aes(x = Fare, fill = Survived)) +
  geom_histogram(binwidth = 5) +
  facet_wrap(~Pclass + Title) +
  ggtitle("Pclass, Title") +
  xlab("Fare") +
  ylab("Total Count") +
  ylim(0, 50) +
  labs(fill = "Survived")
# Not very usful. (No additional valuable data)

# Analysis of the 'Cabin' variable
str(data.combined$Cabin)

# 'Cabin' really isn't a factor. Let's convert it to a string and display first 100
data.combined$Cabin <- as.character(data.combined$Cabin)
data.combined$Cabin[1:100]

# Replace empty 'Cabin' with a 'U' (So we can make the first character a factor)
data.combined[which(data.combined$Cabin == ""), "Cabin"] <- "U"
data.combined$Cabin[1:100]

# Take a look at just the first character as a factor
Cabin.first.char <- as.factor(substr(data.combined$Cabin, 1, 1))
str(Cabin.first.char)
levels(Cabin.first.char)

# Add to data.combined and plot
data.combined$Cabin.first.char <- Cabin.first.char

# High level plot
ggplot(data.combined[1:891,], aes(x = Cabin.first.char, fill = Survived)) +
  geom_bar() +
  ggtitle("Survivability by Cabin.first.char") +
  xlab("Cabin.first.char") +
  ylab("Total Count") +
  ylim(0, 750) +
  labs(fill = "Survived")

# Probably won't have predictive power, but let's take a closer look just in case
ggplot(data.combined[1:891,], aes(x = Cabin.first.char, fill = Survived)) +
  geom_bar() +
  facet_wrap(~Pclass) +
  ggtitle("Survivability by Cabin.first.char") +
  xlab("Pclass") +
  ylab("Total Count") +
  ylim(0, 500) +
  labs(fill = "Survived")

# Does this feature improve upon 'Pclass' + 'Title'?
ggplot(data.combined[1:891,], aes(x = Cabin.first.char, fill = Survived)) +
  geom_bar() +
  facet_wrap(~Pclass + Title) +
  ggtitle("Pclass, Title") +
  xlab("Cabin.first.char") +
  ylab("Total Count") +
  ylim(0, 500) +
  labs(fill = "Survived")

# What about people with multiple cabins?
data.combined$Cabin.multiple <- as.factor(ifelse(str_detect(data.combined$Cabin, " "), "Y", "N"))

ggplot(data.combined[1:891,], aes(x = Cabin.multiple, fill = Survived)) +
  geom_bar() +
  facet_wrap(~Pclass + Title) +
  ggtitle("Pclass, Title") +
  xlab("Cabin.multiple") +
  ylab("Total Count") +
  ylim(0, 350) +
  labs(fill = "Survived")
# Not too useful, either. (For now)

# Does survivability depend on where you got onboard the Titanic?
str(data.combined$Embarked)
levels(data.combined$Embarked)

# Plot data for analysis
ggplot(data.combined[1:891,], aes(x = Embarked, fill = Survived)) +
  geom_bar() +
  facet_wrap(~Pclass + Title) +
  ggtitle("Pclass, Title") +
  xlab("Embarked") +
  ylab("Total Count") +
  ylim(0, 300) +
  labs(fill = "Survived")
# Not too usefule, either.



#-----------------------
# Exploratory Modeling
#-----------------------
library(randomForest)

# Train a Random Forest with the default parameters using 'Pclass' and 'Title'
rf.train.1 <- data.combined[1:891, c("Pclass", "Title")]
rf.label <- as.factor(train$Survived)

set.seed(1234)
rf.1 <- randomForest(x = rf.train.1, y = rf.label, importance = TRUE, ntree = 1000)
rf.1 # Error rate: 20.99%
# OOB = "Out Of Bag"
varImpPlot(rf.1)
# More on the right => More important (variable)

# Train a Random Forest using 'Pclass', 'Title', and 'SibSp'
rf.train.2 <- data.combined[1:891, c("Pclass", "Title", "SibSp")]

set.seed(1234)
rf.2 <- randomForest(x = rf.train.2, y = rf.label, importance = TRUE, ntree = 1000)
rf.2 # Error rate: 19.53%
varImpPlot(rf.2)

# Train a Random Forest using 'Pclass', 'Title', and 'Parch'
rf.train.3 <- data.combined[1:891, c("Pclass", "Title", "Parch")]

set.seed(1234)
rf.3 <- randomForest(x = rf.train.3, y = rf.label, importance = TRUE, ntree = 1000)
rf.3 # Error rate: 19.87%
varImpPlot(rf.3)

# Train a Random Forest using 'Pclass', 'Title', 'SibSp', and 'Parch'
rf.train.4 <- data.combined[1:891, c("Pclass", "Title", "SibSp", "Parch")]

set.seed(1234)
rf.4 <- randomForest(x = rf.train.4, y = rf.label, importance = TRUE, ntree = 1000)
rf.4 # Error rate: 19.08%
varImpPlot(rf.4)

# Train a Random Forest using 'Pclass', 'Title', and 'Family.size'
# Note: 'SibSp' + 'Parch' == 'Family.size', but algorithm might produce something different
rf.train.5 <- data.combined[1:891, c("Pclass", "Title", "Family.size")]

set.seed(1234)
rf.5 <- randomForest(x = rf.train.5, y = rf.label, importance = TRUE, ntree = 1000)
rf.5 # Error rate: 18.18%
varImpPlot(rf.5)

# Out of curiousity, let's put 'SibSp' and 'Family.size' together
# Train a Random Forest using 'Pclass', 'Title', 'SibSp', and 'Family.size'
rf.train.6 <- data.combined[1:891, c("Pclass", "Title", "SibSp", "Family.size")]

set.seed(1234)
rf.6 <- randomForest(x = rf.train.6, y = rf.label, importance = TRUE, ntree = 1000)
rf.6 # Error rate: 19.64% (Went up. Not good)
varImpPlot(rf.6)

# Out of curiousity, let's put 'Parch' and 'Family.size' together
# Train a Random Forest using 'Pclass', 'Title', 'Parch', and 'Family.size'
rf.train.7 <- data.combined[1:891, c("Pclass", "Title", "Parch", "Family.size")]

set.seed(1234)
rf.7 <- randomForest(x = rf.train.7, y = rf.label, importance = TRUE, ntree = 1000)
rf.7 # Error rate: 19.19% (Still higher than just 'Family.size')
varImpPlot(rf.7)



#-------------------
# Cross Validation
#-------------------
# Before we jump into features engineering, we need to establish a methodology for estimating
# our error rate on the Test set (i.e. unseen data). This is critical because without this, we are
# more likely to overfit. Let's start with a submission of rf.5 to Kaggle to see if our OOB error
# estimate is accurate

# Subset our test records and features
test.submit.df <- data.combined[892:1309, c("Pclass", "Title", "Family.size")]

# Make predictions
rf.5.preds <- predict(rf.5, test.submit.df)
table(rf.5.preds)

# Write out a CSV file for submission to Kaggle
submit.df <- data.frame(PassengerId = rep(892:1309), Survived = rf.5.preds)

write.csv(submit.df, "RF_SUB_20190110_1.csv", row.names = FALSE)

# Submission score was different than the OOB prediction.
# Let's look into cross-validation using a caret package to see if we can get more accurate estimates
library(caret)
library(doSNOW)

# Research has shown that 10-fold CV repeated 10 times is the best place to start.
# However, there are no hard and fast rules - this is where the experience of the 
# Data Scientist (i.e. the "art") comes into play.
# We'll start with 10-fold CV, repeated 10 times and see how it goes

# Leverage caret to create 100 total folds, but ensure that the ratio of those that survived and
# perished in each fold matches the overall training set.
# This is known as Stratified-Cross-Validation and generally provides better results.
set.seed(2348)
cv.10.folds <- createMultiFolds(rf.label, k = 10, times = 10)

# Check the stratification
table(rf.label)
342/549

table(rf.label[cv.10.folds[[33]]])
308/494

# Set up caret's trainControl object per above
ctrl.1 <- trainControl(method = "repeatedcv", number = 10, repeats = 10, index = cv.10.folds)

# Set up doSNOW package for multi-core training. 
# This is helpful as we're going to be training a lot of trees.
# NOTE - This works on Windows and Mac, unlike doMC
cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

# Set seed for reproducibility and train
set.seed(32324)
rf.5.cv.1 <- train(x = rf.train.5, y = rf.label, method = "rf", tuneLength = 3, 
                   ntree = 1000, trControl = ctrl.1)

# Shutdown cluster
stopCluster(cl)

# Check out results
rf.5.cv.1

# The above is only slightly more pessimistic than the rf.5 OOB prediction, but not passimistic enough.
# Let's try 5-fold CV repeated 10 times.
set.seed(5983)
cv.5.folds <- createMultiFolds(rf.label, k = 5, times = 10)

ctrl.2 <- trainControl(method = "repeatedcv", number = 5, repeats = 10, index = cv.5.folds)

cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

set.seed(98472)
rf.5.cv.2 <- train(x = rf.train.5, y = rf.label, method = "rf", tuneLength = 3,
                   ntree = 1000, trControl = ctrl.2)

# Shut down cluster
stopCluster(cl)

# Check out results
rf.5.cv.2
# Accuracy went up a tiny bit

# Now, let's try 3-fold CV repeated 10 times
set.seed(37596)
cv.3.folds <- createMultiFolds(rf.label, k = 3, times = 10)

ctrl.3 <- trainControl(method = "repeatedcv", number = 3, repeats = 10, index = cv.3.folds)

cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

set.seed(94622)
rf.5.cv.3 <- train(x = rf.train.5, y = rf.label, method = "rf", tuneLength = 3,
                   ntree = 64, trControl = ctrl.3)

# Shut down cluster
stopCluster(cl)

# Check out results
rf.5.cv.3



#-------------------------
# Exploratory Modeling 2
#-------------------------

# Let's use a single decision tree to better understand what's going on with our features.
# Obviously, Random Forests are far more powerful than single trees, but 
# single trees have the advantage of being easier to understand

# Install and load packages
library(rpart)
library(rpart.plot)

# Per last part, let's use 3-fold CV repeated 10 times

# Create utility function
rpart.cv <- function(seed, training, labels, ctrl){
  cl <- makeCluster(6, type = "SOCK")
  registerDoSNOW(cl)
  
  set.seed(seed)
  # Leverage formula interface for training
  rpart.cv <- train(x = training, y = labels, method = "rpart", tuneLength = 30, trControl = ctrl)
  
  # Shut down cluster
  stopCluster(cl)
  
  return(rpart.cv)
}

# Grab features
features <- c("Pclass", "Title", "Family.size")
rpart.train.1 <- data.combined[1:891, features]

# Run CV and check out results
rpart.1.cv.1 <- rpart.cv(94622, rpart.train.1, rf.label, ctrl.3)
rpart.1.cv.1
# Best result is 0.8170595 (3rd from top under Accuracy) (cp tells you what the best result is)

# Plot
prp(rpart.1.cv.1$finalModel, type = 0, extra = 1, under = TRUE)

# The plot brings out some interesting lines of investigation. Namely:
# 1. Titles of 'Mr.' and 'Other' are predicted to perish at an overall accuracy rate of 83.2%
# 2. Titles of 'Master.', 'Miss.', and 'Mrs.' in 1st and 2nd class are predicted
#    to survive at an overall accuracy rate of 94.9%
# 3. Titles of 'Master.', 'Miss.', and 'Mrs.' in 3rd class with family size equal to 
#    5, 6, 8, and 11 are predicted to perish with 100% accuracy
# 4. Titles of 'Master.', 'Miss.', and 'Mrs.' in 3rd class with family size NOT equal to 
#    5, 6, 8, and 11 are predicted to survive with 59.6% accuracy

# Point 3 and 4 is an example of 'Overfitting' (too specific to the training set )

# Both rpart and rf confirm that 'Title' is important. Let's investigate further
table(data.combined$Title)

# Parse out last name and title
data.combined[1:25, "Name"]

name.splits <- str_split(data.combined$Name, ",")
name.splits[1]
last.names <- sapply(name.splits, "[", 1)
last.names[1:10]

# Add last names to dataframe in case we find it useful later
data.combined$Last.name <- last.names

# Now for titles
name.splits <- str_split(sapply(name.splits, "[", 2), " ")
titles <- sapply(name.splits, "[", 2)
unique(titles)

# What's up with a title of 'the'?
data.combined[which(titles == "the"),]

# Re-map titles to be more exact
titles[titles %in% c("Dona.", "the")] <- "Lady."
titles[titles %in% c("Ms.", "Mlle.")] <- "Miss."
titles[titles == "Mme."] <- "Mrs."
titles[titles %in% c("Jonkheer.", "Don.")] <- "Sir."
titles[titles %in% c("Col.", "Capt.", "Major.")] <- "Officer"
table(titles)

# Make titles a factor
data.combined$new.Title <- as.factor(titles)

# Visualize new version of titles
ggplot(data.combined[1:891,], aes(x = new.Title, fill = Survived)) +
  geom_bar() +
  facet_wrap(~Pclass) +
  ggtitle("Survival Rates for new.Title by Pclass")
  
# Collapse titles based on visual analysis, to avoid overfitting
indexes <- which(data.combined$new.Title == "Lady.")
data.combined$new.Title[indexes] <- "Mrs."

indexes <- which(data.combined$new.Title == "Dr." |
                 data.combined$new.Title == "Rev." |
                 data.combined$new.Title == "Sir." |
                 data.combined$new.Title == "Officer")
data.combined$new.Title[indexes] <- "Mr."

# Visualize
ggplot(data.combined[1:891,], aes(x = new.Title, fill = Survived)) +
  geom_bar() +
  facet_wrap(~Pclass) +
  ggtitle("Survival Rates for new.Title by Pclass")

# Grab features
features <- c("Pclass", "new.Title", "Family.size")
rpart.train.2 <- data.combined[1:891, features]

# Run CV and check out results
rpart.2.cv.1 <- rpart.cv(94622, rpart.train.2, rf.label, ctrl.3)
rpart.2.cv.1
# Best result is 0.8280584 (3rd from top under Accuracy) (cp tells you what the best result is)

# Plot
prp(rpart.2.cv.1$finalModel, type = 0, extra = 1, under = TRUE)

# Dive in on 1st class "Mr."
indexes.first.mr <- which(data.combined$new.Title == "Mr." & data.combined$Pclass == 1)
first.mr.df <- data.combined[indexes.first.mr,]
summary(first.mr.df)

# We have one female. That's not good. (Made an incorrect assumption)
first.mr.df[first.mr.df$Sex == "female",]

# Update new.Title feature
indexes <- which(data.combined$new.Title == "Mr." &
                 data.combined$Sex == "female")
data.combined$new.Title[indexes] <- "Mrs."

# Any other gender slip-ups?
length(which(data.combined$Sex == "female" &
             (data.combined$new.Title == "Master." |
              data.combined$new.Title == "Mr.")))

# Refresh data frame
indexes.first.mr <- which(data.combined$new.Title == "Mr." & data.combined$Pclass == 1)
first.mr.df <- data.combined[indexes.first.mr,]

# Let's look at surviving 1st class "Mr."
summary(first.mr.df[first.mr.df$Survived == "1",])
View(first.mr.df[first.mr.df$Survived == "1",])

# Take a look at some of the high fares
indexes <- which(data.combined$Ticket == "PC 17755" |
                 data.combined$Ticket == "PC 17611" |
                 data.combined$Ticket == "113760")
View(data.combined[indexes,])

# Visualize survival rates for 1st class "Mr." by 'Fare'
ggplot(first.mr.df, aes(x = Fare, fill = Survived)) +
  geom_density(alpha = 0.5) +
  ggtitle("1st Class 'Mr.' Survival Rate by Fare")

# Engineer featres based on all the passengers with the same ticket
ticket.party.size <- rep(0, nrow(data.combined))
avg.fare <- rep(0.0, nrow(data.combined))
tickets <- unique(data.combined$Ticket)

for(i in 1:length(tickets)){
  current.ticket <- tickets[i]
  party.indexes <- which(data.combined$Ticket == current.ticket)
  current.avg.fare <- data.combined[party.indexes[1], "Fare"]/length(party.indexes)
  
  for(k in 1:length(party.indexes)){
    ticket.party.size[party.indexes[k]] <- length(party.indexes)
    avg.fare[party.indexes[k]] <- current.avg.fare
  }
}

data.combined$Ticket.party.size <- ticket.party.size
data.combined$Avg.fare <- avg.fare

# Refresh 1st class "Mr." dataframe
first.mr.df <- data.combined[indexes.first.mr,]
summary(first.mr.df)

# Visualize new features
ggplot(first.mr.df[first.mr.df$Survived != "None",], aes(x = Ticket.party.size, fill = Survived)) +
  geom_density(alpha = 0.5) +
  ggtitle("Survival Rates 1st Class 'Mr.' by Ticket.party.size")

ggplot(first.mr.df[first.mr.df$Survived != "None",], aes(x = Avg.fare, fill = Survived)) +
  geom_density(alpha = 0.5) +
  ggtitle("Survival Rates 1st Class 'Mr.' by Avg.fare")

# Hypothesis: Ticket.party.size is highly correlated with Avg.fare
summary(data.combined$Avg.fare)

# There's one value missing. Let's take a look
data.combined[is.na(data.combined$Avg.fare),]

# Important thought process!!!!!! (Making NA equal to median or mean)
# Note: if there is too many NA, then you need to use different technique
#------------------------------------------------------------------------
# Get records for similar passengers and summarize Avg.fare
indexes <- with(data.combined, which(Pclass == "3" & Title == "Mr." & Family.size == "1" &
                                       Ticket != "3701"))
similar.na.passengers <- data.combined[indexes,]
summary(similar.na.passengers$Avg.fare)

# Use median since close to mean and a little higher than mean
data.combined[is.na(avg.fare), "Avg.fare"] <- 7.840
#------------------------------------------------------------------------

# Leverage caret's preProcess function to normalize data
preproc.data.combined <- data.combined[, c("Ticket.party.size", "Avg.fare")]
preProc <- preProcess(preproc.data.combined, method = c("center", "scale"))

postproc.data.combined <- predict(preProc, preproc.data.combined)

# Hypothesis refuted for all data
cor(postproc.data.combined$Ticket.party.size, postproc.data.combined$Avg.fare)
# We get 0.09428625, so highly uncorrelated.
# Which is good, b/c we potentially have 2 new features to extract predictive power out of our data set

# How about for just 1st class all-up?
indexes <- which(data.combined$Pclass == "1")
cor(postproc.data.combined$Ticket.party.size[indexes],
    postproc.data.combined$Avg.fare[indexes])
# We get 0.2576249, so still uncorrelated. (Hypothesis refuted)

# OK, let's see if our feature engineering has made any difference
features <- c("Pclass", "new.Title", "Family.size", "Ticket.party.size", "Avg.fare")
rpart.train.3 <- data.combined[1:891, features]

# Run CV and check out results
# Note: rpart models = individual decision trees
rpart.3.cv.1 <- rpart.cv(94622, rpart.train.3, rf.label, ctrl.3)
rpart.3.cv.1
# Accuracy 0.8315376 (check cp)

prp(rpart.3.cv.1$finalModel, type = 0, extra = 1, under = TRUE)



#--------------------------------
# Final thoughts
#--------------------------------

# Let's see how rpart scores on Kaggle
# Subset our test scores and features
test.submit.df <- data.combined[892:1309, features]

# Make predictions
rpart.3.preds <- predict(rpart.3.cv.1$finalModel, test.submit.df, type = "class")
table(rpart.3.preds)

# Write out a CSV file for submission to Kaggle
submit.df <- data.frame(PassengerId = rep(892:1309), Survived = rpart.3.preds)

write.csv(submit.df, file = "RPART_SUB_20190113_1.csv", row.names = FALSE)
# On Kaggle, rpart scores 0.80383

# Let's see how Random Forest scores on Kaggle
# Subset our test scores and features
features <- c("Pclass", "new.Title", "Ticket.party.size", "Avg.fare")
rf.train.temp <- data.combined[1:891, features]

set.seed(1234)
rf.temp <- randomForest(x = rf.train.temp, y = rf.label, ntree = 1000)
rf.temp

test.sumbit.df <- data.combined[892:1309, features]

# Make predictions (?predict.randomForest) 
# Note: bit different to predict in rpart trees. rf has type = "response" (0 and 1)
rf.preds <- predict(rf.temp, test.submit.df)
table(rf.preds)

# Write out a CSV file for submission to Kaggle
submit.df <- data.frame(PassengerId = rep(892:1309), Survived = rf.preds)

write.csv(submit.df, file = "RF_SUB_20190113_2.csv", row.names = FALSE)
# On Kaggle Random Forest scores 0.80861


# If we want to improve our model, a good place to start is focusing on where it gets things wrong.
# YouTube 'features engineering' for more techniques

# First, let's explore our collection of features using mutual information to gain some additional insight.
# Our intuition is that the plot of our tree should align well to the definition of mutual information.
# defn: Mutual information is a mathematical way of understanding if features that we engineered
#       are useful, in terms of prediction. (related to entropy)
# For more on mutual information, go to nlp.stanford.edu/IR-book/html/htmledition/mutual-information-1.html
library(infotheo)

mutinformation(rf.label, data.combined$Pclass[1:891])
mutinformation(rf.label, data.combined$Sex[1:891])
mutinformation(rf.label, data.combined$SibSp[1:891])
mutinformation(rf.label, data.combined$Parch[1:891])
mutinformation(rf.label, discretize(data.combined$Fare[1:891]))
mutinformation(rf.label, data.combined$Embarked[1:891])
mutinformation(rf.label, data.combined$Title[1:891])
mutinformation(rf.label, data.combined$Family.size[1:891])
mutinformation(rf.label, data.combined$Ticket.first.char[1:891])
mutinformation(rf.label, data.combined$Cabin.multiple[1:891])
mutinformation(rf.label, data.combined$new.Title[1:891])
mutinformation(rf.label, data.combined$Ticket.party.size[1:891])
mutinformation(rf.label, discretize(data.combined$Avg.fare[1:891]))

# OK, now let's leverage the tsne algorithm to create a 2-D representation of our data suitable for
# visualization starting with folks our model gets right very often - females and boys
# Rtsne = used for dimensionality reduction
library(Rtsne)
most.correct <- data.combined[data.combined$new.Title != "Mr.",]
indexes <- which(most.correct$Survived != "None")

tsne.1 <- Rtsne(most.correct[, features], check_duplicates = FALSE)
ggplot(NULL, aes(x = tsne.1$Y[indexes, 1], y = tsne.1$Y[indexes, 2],
                 color = most.correct$Survived[indexes])) +
  geom_point() +
  labs(color = "Survived") +
  ggtitle("tsne 2D Visualization of Features for Females and Boys")
# Pay attention to the clusters

# Mutual information: 1 for 1, got a predictor variable called A, and a potential feature called B.
#                    "If I know B, how much uplift does that give me in terms of my understanding of A.
#                    "How predictive is B for A".
# Conditional mutual information: use A,B, and C. "If I know B, and you give me C, does C give me even
#                                more uplift regarding predicting the values of A".

# To get a baseline, let's use conditional mutual information on the tsne X and Y features for
# females and boys in 1st and 2nd class. The intuition here is that the combination of these features
# should be higher than any indivual feature we looked at above
condinformation(most.correct$Survived[indexes], discretize(tsne.1$Y[indexes,]))
# Mathematical way of representing how cleanly the turquois color is separated from the orange

# As one more comparison, we can leverage conditional mutual information using the top two features
# used in our tree plot - new.Title and Pclass
condinformation(rf.label, data.combined[1:891, c("new.Title", "Pclass")])

# OK, now let's take a look at adult males since our model has the biggest potential upside for improving.
# (i.e. the tree predicts incorrectly for 86 adult males). Let's visualize with tsne
misters <- data.combined[data.combined$new.Title == "Mr.",]
indexes <- which(misters$Survived != "None")

tsne.2 <- Rtsne(misters[, features], check_duplicates = FALSE)
ggplot(NULL, aes(x = tsne.2$Y[indexes, 1], y = tsne.2$Y[indexes, 2],
                 color = misters$Survived[indexes])) +
  geom_point() +
  labs(color = "Survived") +
  ggtitle("tsne 2D Visualization of Features for new.Title of 'Mr.'")

# Now conditional mutal information for tsne features for adult males
condinformation(misters$Survived[indexes], discretize(tsne.2$Y[indexes,]))

# Idea - How about creating tsne features for all of the training data and using them in our model?
tsne.3 <- Rtsne(data.combined[, features], check_duplicates = FALSE)
ggplot(NULL, aes(x = tsne.3$Y[1:891, 1], y = tsne.3$Y[1:891, 2],
                 color = data.combined$Survived[1:891])) +
  geom_point() +
  labs(color = "Survived") +
  ggtitle("tsne 2D Visualization of Features for all Training Data")

# Now conditional mutal information for tsne features for all training data
condinformation(data.combined$Survived[1:891], discretize(tsne.3$Y[1:891,]))

# Add the tsne features to our data frame for use in model building
data.combined$tsne.x <- tsne.3$Y[,1]
data.combined$tsne.y <- tsne.3$Y[,2]
