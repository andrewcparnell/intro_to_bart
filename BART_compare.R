# A simple script to compare BART against other methods on basic data sets

library(mlr)
library(mlbench)
library(tidyverse)

# Get the data
data(BostonHousing, package = "mlbench")

# Set the seed
set.seed(123)

# Set the task
regr.task = makeRegrTask(id = "bh", data = BostonHousing, target = "medv")
regr.task

# Choose learners
lrns = list(makeLearner("regr.lm"), 
            makeLearner("regr.randomForest"),
            makeLearner("regr.svm"),
            makeLearner("regr.h2o.deeplearning"),
            makeLearner("regr.bartMachine"))

# resample description
rdesc = makeResampleDesc("CV", iters = 10)

# Conduct the benchmark experiment
bmr = benchmark(lrns, regr.task, rdesc, measures = list(rmse))

# Compare
p = plotBMRBoxplots(bmr)

# Do the plot properly
plot_dat = p$data
levels(plot_dat$learner.id) = c("Linear regression", 
                                "Random forests", 
                                "Support vector regression", 
                                "Deep learning", 
                                "BART")
# ggplot(plot_dat, aes(x = learner.id, y = rmse, group = learner.id, fill = learner.id)) + 
#   geom_boxplot() + 
#   theme_bw() + 
#   labs(title = 'Out of sample 10-fold cross validation root mean square error',
#        x = '',
#        y = 'Root mean square error') + 
#   coord_flip() + 
#   theme(legend.position = 'None')
# ggsave('Boston_housing_benchmark.pdf', width = 8, height = 6)


# Show uncertainty in predictions -----------------------------------------

# Re-fit the RF and BART models
set.seed(123)
n = nrow(BostonHousing)
train = sort(sample(1:nrow(BostonHousing), size = 19*n/20))
test = (1:n)[-train]
rf_fit = randomForest::randomForest(x = BostonHousing[train,-ncol(BostonHousing)], 
                      y = BostonHousing[train,'medv'],
                      xtest = BostonHousing[test,-ncol(BostonHousing)],
                      ytest = BostonHousing[test,'medv'])
bart_fit = bartMachine::bartMachine(X = BostonHousing[train,-ncol(BostonHousing)], 
                                    y = BostonHousing[train,'medv'])

true_y = BostonHousing[test,'medv']
pred_rf = rf_fit$test$predicted
pred_bart_mean = predict(bart_fit, new_data = BostonHousing[test,-ncol(BostonHousing)])
pred_bart_interval = bartMachine::calc_prediction_intervals(bart_fit,
                                                            pi_conf = 0.5,
                                               new_data = BostonHousing[test,-ncol(BostonHousing)])$interval

df = data.frame(y = true_y,
                rf = pred_rf,
                bart_mean = pred_bart_mean)
                
interval = data.frame(y = true_y,
                      fit = pred_bart_mean,
                      bart_low = pred_bart_interval[,1],
                      bart_high = pred_bart_interval[,2],
                      Method = 'BART')

df2 = df %>% gather(-y,
                    key = method,
                    value = fit)

df2$Method = as.factor(df2$method)
levels(df2$Method) = c("BART", "Random forest") 

ggplot(df2, aes(x = y, y = fit, colour = Method)) +
  geom_point() +
  theme_bw() +
  geom_abline(intercept = 0, slope = 1) +
  labs(title = 'Out of sample performance: 50% uncertainty interval',
       x = 'True values',
       y = 'Predicted values') +
  geom_errorbar(data = interval, 
                aes(ymin = bart_low, ymax = bart_high))
ggsave('BART_prediction_intervals.pdf', width = 8, height = 6)

# Friedman example --------------------------------------------------------

source('https://raw.githubusercontent.com/andrewcparnell/rBART/master/rBART.R')

# Now do it on the Friedman example
set.seed(123)
dat = sim_friedman(500)
dat2 = data.frame(dat$X, dat$y)

# Create new task
regr2.task = makeRegrTask(id = "fr", data = dat2, target = "dat.y")

# Choose learners
lrns = list(makeLearner("regr.lm"), 
            makeLearner("regr.randomForest"),
            makeLearner("regr.svm"),
            makeLearner("regr.h2o.deeplearning"),
            makeLearner("regr.bartMachine"))

# resample description
rdesc = makeResampleDesc("CV", iters = 10)

# Conduct the benchmark experiment
bmr2 = benchmark(lrns, regr2.task, rdesc, measures = list(rmse))

# Compare
p2 = plotBMRBoxplots(bmr2)

# Do the plot properly
plot_dat2 = p2$data
levels(plot_dat2$learner.id) = c("Linear regression", 
                                "Random forests", 
                                "Support vector regression", 
                                "Deep learning", 
                                "BART")
# ggplot(plot_dat2, aes(x = learner.id, y = rmse, group = learner.id, fill = learner.id)) + 
#   geom_boxplot() + 
#   theme_bw() + 
#   labs(title = 'Out of sample 10-fold cross validation root mean square error',
#        x = '',
#        y = 'Root mean square error') + 
#   coord_flip() + 
#   theme(legend.position = 'None')
# ggsave('Friedman_benchmark.pdf', width = 8, height = 6)
