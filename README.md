# Clickbait-Detection

Machine Learning based Clickbait Detector

Clickbait is basically some content on the Internet whose main purpose is to attract attention and encourage visitors to click on a link to a particular web page.

Although clickbaits can be in the form of texts or thumbnails these model detect text based Clickbaits

## Data Sources

* [Kaggle: clickbait-dataset](https://www.kaggle.com/amananandrai/clickbait-dataset)
* [Kaggle: news-clickbait](https://www.kaggle.com/micdsouz/news-clickbait)
* [Webis-clickbait-16](https://webis.de/data/webis-clickbait-16.html)
* [RVCJ](https://www.rvcj.com/) `crawled`
* [Viral Nova](https://viralnova.com/) `crawled`
* [Viral Stories](http://viralstories.in/) `crawled`
* [Inshorts](https://inshorts.com/en/read) `crawled`

Crawler : [Lord Varys](https://github.com/Ritvik19/Lord-Varys)

## Model Performances

Model | Accuracy | AUROC
---|---:|---:
Logisitic Regression | 0.8745  | 0.9431
Multinomimal Naive Bayes + Oversampling | 0.8561 | 0.9342
Random Forest | 0.8448 | 0.9060
Bagging Logistic Regression | 0.8760  | 0.9443 
