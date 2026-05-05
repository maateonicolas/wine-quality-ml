# EDA Notes - Wine Quality

## Dataset overview

The dataset contains physicochemical measurements of wine samples and a target variable called `quality`.

## Target distribution

The target variable is not uniformly distributed. Most observations are concentrated in medium quality scores, while very low and very high quality wines are less frequent. This is relevant because models may perform better on common quality levels and worse on rare scores.

## Key variables

### Alcohol

Alcohol appears to be one of the most relevant variables. Higher-quality wines tend to show higher alcohol values on average.

### Volatile acidity

Volatile acidity tends to be lower for higher-quality wines. This suggests a negative relationship with perceived wine quality.

### Sulphates

Sulphates may show some positive association with quality, although the relationship is not perfectly linear.

### pH

pH shows less obvious separation across quality levels compared with alcohol or volatile acidity.

## Correlation and multicollinearity

The correlation matrix helps identify relationships among predictors. This is especially relevant for Ridge, Lasso, and Elastic Net. Lasso may select only one variable among correlated predictors, while Elastic Net can be more stable when predictors are correlated.

## Modeling implication

Because `quality` is ordinal and numeric, the project can initially treat it as a regression task. However, the imbalance in quality levels should be discussed because extreme scores are underrepresented.