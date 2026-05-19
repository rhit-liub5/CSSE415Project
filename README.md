# Data Source
Our data is collected at [this](https://www.kaggle.com/datasets/ektarr/counter-strike-pro-matches) Kaggle page. We originally imported the cs2_tier1_games.csv, but later engineered our actual datasets to predict on since this is historical data which is not immediately ML-ready.

# Structure
All datasets should be placed at `./data`. All model and analysis code is collected under the `./project` directory with additional information specified below.

## Project Subdirectories
There are 4 subdirectories inside of `./project`, the role of each is described subsequently.

### engineered_set
This collects notebooks on the player-level rolling features dataset. It includes decision trees, gradient boosting, KNN, logistic regression base and forward feature selection, and random forest.

### engineering
This collects notebooks used to create the engineered rolling feature sets for both the player-level dataset and the team-level dataset. It also include a baseline analysis and data distribution analysis.

### initial_data
This collects notebooks that briefly work on and clean the original feature set, but this was quickly abandoned as we realized the post-game nature of the features. It includes a data distribution analysis and KNN and logistic regression models.

### team_set
Analogous to the engineered_set, this collects notebooks for work done on the team-level rolling feature set which includes gradient boosted trees, random forest, various lasso regressions, logistic regression, KNN, SVM, and PCA.

# Dependencies
See `./src/requirements.txt`

# Running the Code
All notebooks have been ran with output included, and we do not recommend rerunning them as some hyperparameter tuning takes upwards of an hour. You can simply open any notebook and read the output.