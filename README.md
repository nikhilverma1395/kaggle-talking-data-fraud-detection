# kaggle-talking-data-fraud-detection
https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection


A few tries over the Kaggle Fraud Detection:

Training Data is comparatively large, so that is a hiccup, had to pickle whatever transformations I did over the notebooks.

Used embedding layers for all the input features, though I might have overdone the features generation part, minute level aggregations don't help I guess.

Training several different models wasn't feasible, a single epoch took 5hrs for 90% of the training data.

I would be trying a lightgbm model in the next few iterations.
