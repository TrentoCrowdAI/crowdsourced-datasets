# Weather Sentiment - AMT Dataset

Link to the original source: https://eprints.soton.ac.uk/376543/ 

This dataset contains the ground truths and individual judgments of 110 workers for the sentiment of 300 tweets. The classification task is performed in the following categories: negative (0), neutral (1), positive (2), tweet not related to weather (3) and can't tell (4). From the source given above, `WeatherSentiment_amt.csv` file should be copied into the `data-raw` folder.

**Cite this work as:**
```
@inproceedings{soton376365,
       booktitle = {International Joint Conference on Artificial Intelligence (IJCAI-15) (31/07/15)},
           month = {July},
           title = {Bayesian modelling of community-based multidimensional trust in participatory sensing under data sparsity},
          author = {Matteo Venanzi and W.T.L. Teacy and Alex Rogers and Nicholas R. Jennings},
            year = {2015},
           pages = {717--724},
             url = {https://eprints.soton.ac.uk/376365/},
        abstract = {We propose a new Bayesian model for reliable aggregation of crowdsourced estimates of real-valued quantities in participatory sensing applications. Existing approaches focus on probabilistic modelling of user?s reliability as the key to accurate aggregation. However, these are either limited to estimating discrete quantities, or require a significant number of reports from each user to accurately model their reliability. To mitigate these issues, we adopt a community-based approach, which reduces the data required to reliably aggregate real-valued estimates, by leveraging correlations between the re- porting behaviour of users belonging to different communities. As a result, our method is up to 16.6\% more accurate than existing state-of-the-art methods and is up to 49\% more effective under data sparsity when used to estimate Wi-Fi hotspot locations in a real-world crowdsourcing application.}
}
```
