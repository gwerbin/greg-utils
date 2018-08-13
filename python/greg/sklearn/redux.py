from sklearn.utils.extmath import safe_sparse_dot

# from sklearn.naive_bayes import *


# c = class
# a = attribute
# m = method
# ma = abstract method
# <- = calls
# * = mutates (attrs only)

## Naive Bayes
# c BaseNB
#   a classes_
#   ma _joint_log_likelihood
#   m predict
#     <- _joint_log_likelihood
#     <- classes_
#   m predict_log_proba
#     <- _joint_log_likelihood
#   m predict_proba
#     <- _joint_log_likelihood
#   m predict_proba
#     <- predict_log_proba
# c BaseDiscreteNB
#   a class_count_
#   a class_log_prior_
#   a feature_count_
#   m _update_class_log_prior
#     <- *class_log_prior_
#     <- classes_
#     <- class_count_
#   m partial_fit
#     <- *class_count_
#     <- *feature_count_
#     <- classes_
#     <- class_prior
#     <- _update_feature_log_prob
#     <- _update_class_log_prior
# c BernoulliNB
# 
