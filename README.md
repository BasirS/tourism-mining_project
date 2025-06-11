# Istanbul Shopping and Tourism Analysis (in-progress)

## Project Overview
This project analyzes Istanbul's retail and tourism sectors using machine learning and data analytics techniques. The analysis focuses on understanding customer behaviors, mall operations, and tourism trends to provide actionable insights for business optimization.

## Dataset Description
The analysis utilizes a comprehensive dataset of customer shopping data from Istanbul (2021-2023), containing:
- Customer demographics (age, gender)
- Transaction details (invoice ID, date, payment method)
- Product information (category, quantity, price)
- Location data (shopping mall name)

## Analysis Methodology

### 1. Data Preprocessing
The raw data underwent several preprocessing steps:
- Conversion of temporal features (year, month, day, day of week)
- Calculation of total transaction amounts
- Handling of missing values and data type standardization

### 2. Customer Segmentation Analysis
Using K-means clustering, customers were segmented into four distinct groups based on:
- Total spending
- Visit frequency
- Average transaction value
- Shopping patterns

### 3. Purchase Prediction Modeling
A Random Forest Regressor was implemented to predict customer spending patterns, considering:
- Product categories
- Payment methods
- Mall locations
- Temporal features

Model Performance Metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared Score

## Key Visualizations and Insights

### Demographic Analysis: Who Are the Shoppers?
![Demographic Distribution](plots/demographic_distribution_20250516_193003.png)

To get a baseline for the project, I first looked at the customer demographics. The `plot_demographic_distribution` function in your `visualization/plot_generator.py` script generates these plots, which give us the first essential clues about our dataset before we dive into the more complex models.

* **A Female-Driven Market**
    The first thing that jumps out is the gender split. The pie chart shows the customer base is mostly female, making up about **60%** of the total. This means for every two male shoppers, there are three female shoppers. This fact directly impacts our later analysis. The `customer_segmentation.py` script uses 'gender' as a key feature for its K-Modes clustering. Because this 60/40 split is so clear, the 'gender' feature will almost certainly help define the customer segments that the model uncovers.

* **All Ages are Welcome**
    The second chart, which details the age of shoppers, was pretty surprising. Instead of showing a spike for one generation, the distribution is almost completely flat from age 18 all the way to 70. This tells us the malls successfully attract a very wide range of people. This visual also justifies a key preparation step inside the `customer_segmentation.py` script. Because there are no natural peaks at individual ages, the code makes a smart feature engineering choice by creating broader 'age_group' bins. This helps our clustering models find patterns across generations (like the '26-35' group) instead of getting lost in the noise of individual ages.

#### Putting It All Together

So, when we combine these two findings, we get a much clearer picture of the "typical" shopper. While a specific age doesn't define her, her gender likely does: she is probably female. This fact is a big deal for the project because it validates the decision to use 'gender' as a core feature and explains the need to engineer the 'age' feature into broader groups. This demographic snapshot provides the foundational "why" for the technical steps our scripts take later on.

### Spending Patterns: What People Buy and How They Pay
![Spending Patterns](plots/spending_patterns_20250516_193003.png)

After looking at who the shoppers are, the next logical step was to dig into what they're actually buying and how they're paying for it all. These plots give us a great overview of the main product categories and payment habits, which are key features our models in `customer_segmentation.py` and `purchase_prediction.py` rely on to find patterns.

* **Clothing and Shoes Dominate Sales**
    The bar chart on the left makes it super clear where the money is coming from. **Clothing** is the undisputed king, bringing in the most revenue by a long shot, with **Shoes** as a strong second. Technology also holds its own as a major category. This is really valuable information because 'category' is one of the main features we're feeding into the K-Modes clustering algorithm in `customer_segmentation.py`. Seeing this dominance suggests that the algorithm will likely identify a large and valuable customer segment that is purely focused on fashion. It also gives us context for our `purchase_prediction.py` script, which will probably learn that shopping carts containing 'Clothing' or 'Technology' are a strong indicator of a higher total spend.

* **Cards are Preferred, but Cash is Still Key**
    The pie chart on the right, which breaks down how people pay, shows that plastic is the favorite. **Credit Cards** are the most common method at about 45%, and when you add in Debit Cards, card payments account for roughly two-thirds of all transactions. At the same time, we can't ignore **Cash**, which still makes up a huge chunk of purchases. This `payment_method` column is another critical feature that both our segmentation and prediction scripts use. This breakdown makes me wonder if the models will find a link between payment choice and spending habits—for example, do the big spenders on 'Technology' prefer to use credit cards? Our models will help us answer that.

#### Putting It All Together

So, these charts give us the "what" and "how" of customer behavior. We now know that the market is heavily driven by fashion sales and that customers prefer paying by card. This isn't just trivia; it provides meaning to the categorical features we are using in our Python scripts. It helps us form better hypotheses about the customer segments we expect to find and gives us a solid foundation for understanding the patterns our Random Forest model will eventually predict.

### Mall Performance: Where the Action Happens
![Mall Analysis](plots/mall_analysis_20250516_193003.png)

Now that I have a good idea of who the customers are and what they're buying, I wanted to check out *where* they do their shopping. These charts look at how the different malls perform, which is super important because the `shopping_mall` column acts as a key categorical feature that both our `customer_segmentation.py` and `purchase_prediction.py` scripts use to find patterns.

* **A Clear Hierarchy in Sales**
    The bar chart on the left immediately shows that the malls don't perform equally. The **Mall of Istanbul** clearly leads the pack, pulling in the most revenue, with **Kanyon** following as a strong second. The other malls grab their share, but a definite pecking order exists. This insight provides fantastic information for our prediction model. Since our `purchase_prediction.py` script uses the mall's name to help predict spending, this chart confirms that the `shopping_mall` feature offers powerful predictive value. The model will likely learn that a transaction at the Mall of Istanbul strongly points to higher revenue.

* **Customer Footprint Mirrors Sales**
    The pie chart on the right tells a very similar story by showing where customers physically go. The **Mall of Istanbul** and **Kanyon** attract the most shoppers, taking the biggest slices of the pie. This makes sense, as more customers generally drive more sales. This provides really useful information for our `customer_segmentation.py` script. The K-Modes algorithm, which groups customers by their habits, will probably find that certain customer segments show strong loyalty to specific malls. For example, the fashion-focused shoppers we identified earlier probably flock to these top-performing malls.

#### Putting It All Together

Basically, these two charts prove the old saying "location, location, location" holds true here. The specific mall where a purchase happens contributes more than just random noise; it acts as a powerful variable that connects directly to both sales volume and customer traffic. This validates our choice to use `shopping_mall` as a feature in both the segmentation and prediction models and gives us a great real-world context for the patterns our Python scripts will uncover.

### Temporal Analysis: When People Are Shopping
![Temporal Trends](plots/temporal_trends_20250516_193003.png)

After figuring out the who, what, and where of the shopping story, the final piece of the puzzle was the *when*. This is where the time-based features that the `preprocess_data` function creates in both `main.py` and `customer_segmentation.py` really come to life. These charts visualize the patterns that those engineered features, like 'Month' and 'DayOfWeek', capture from the raw invoice dates.

* **The Holiday Shopping Rush is Real**
    The top line chart, which maps out sales over the entire 2021-2023 period, shows a ton of daily ups and downs, but three massive spikes are impossible to miss. These peaks happen consistently around the end-of-year holiday season. This visual evidence is great because it confirms that our decision to engineer `Year` and `Month` as separate features was a good one. For the `purchase_prediction.py` script, this is gold. The Random Forest model will be able to use these temporal features to learn this seasonal pattern and predict higher spending during the holiday rush.

* **The Weekly Shopping Rhythm**
    The bottom chart gives us a closer look at the data by breaking down sales by the day of the week. There’s a clear and predictable rhythm here: sales start lower on Monday and build up through the week, peaking on Friday and Saturday before dipping again on Sunday. This chart validates the usefulness of the `DayOfWeek` feature we created during preprocessing. It's not just a random number; it captures a consistent human behavior. This weekly pattern is another strong signal that our prediction model can use to understand the flow of sales.

#### Putting It All Together

So, these charts make it obvious that time is a super important factor in this dataset. Shopping behavior isn't random; it follows clear seasonal and weekly cycles. This completely justifies the work in our preprocessing scripts to extract `Month` and `DayOfWeek` from the invoice dates. These temporal features give our machine learning models the context they need about *when* a purchase is happening, which allows them to make much smarter and more accurate predictions about customer spending.

### Correlation Analysis: How the Numbers Relate
![Correlation Heatmap](plots/correlation_analysis_20250516_193003.png)

For the final piece of the exploratory analysis, I wanted to see how the different numerical features in the dataset relate to each other. This heatmap, which the `plot_correlation_analysis` function generates, gives us a quick visual summary of the linear relationships. In this chart, brighter squares mean two variables move up or down together, while darker squares mean they don't have a strong straight-line relationship.

* **The Strongest Links: A Good Sanity Check**
    The first thing to notice are the really bright yellow squares. There's a very strong positive correlation between **TotalAmount** and both **quantity** and **price**. This makes perfect sense and acts as a great sanity check for our data. The `preprocess_data` function in `main.py` calculates `TotalAmount` by multiplying `quantity` and `price`, so we expect them to be tightly linked. Seeing this confirms our feature engineering worked just as we designed it to.

* **What the Weak Links Tell Us**
    What's even more interesting, though, are the darker squares. For example, **age** shows a very weak linear correlation with **TotalAmount**. This doesn't mean age is a useless feature. It just tells us that the relationship isn't a simple "as age goes up, spending goes up" kind of thing. This totally backs up the approach in `customer_segmentation.py` to create 'age_group' bins, which helps capture more complex, non-linear patterns.

    Similarly, the temporal features like **Month** and **DayOfWeek** also show weak linear correlations with `TotalAmount` here. We already know from the previous line charts that these features hold clear seasonal and weekly patterns. The heatmap's weak result here just proves that those patterns aren't simple straight lines. This is the single best justification for our choice of model in `purchase_prediction.py`. A simple linear regression would fail to see these patterns, but a **Random Forest** is great at capturing exactly these kinds of complex, non-linear relationships, which makes it the right tool for the job.

## Key Findings
1. Customer Segments:
   - High-value frequent shoppers
   - Moderate-value regular customers
   - Occasional shoppers
   - Budget-conscious customers

2. Purchase Patterns:
   - Peak shopping periods
   - Preferred product categories
   - Payment method preferences

3. Mall Performance:
   - Revenue distribution
   - Customer traffic patterns
   - Product category success

## Business Implications
1. Inventory Management:
   - Optimize stock levels based on segment preferences
   - Align product offerings with customer demographics

2. Marketing Strategies:
   - Target specific customer segments
   - Personalize promotions based on shopping patterns

3. Mall Operations:
   - Optimize store layouts
   - Enhance customer experience
   - Improve revenue distribution

## Technical Implementation
The analysis was implemented using:
- Python 3.8+
- Key libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
- Custom modules for segmentation, prediction, and visualization

## Project Structure
```
tourism-mining_project/
├── README.md
├── requirements.txt
├── src/
│   ├── main.py
│   ├── analysis/
│   │   └── customer_segmentation.py
│   ├── models/
│   │   └── purchase_prediction.py
│   └── visualization/
│       └── plot_generator.py
├── data/
│   └── istanbul_shopping_data.csv
├── plots/
└── results/
```

## Future Work
1. Enhanced segmentation using additional features
2. Real-time prediction system implementation
3. Integration with business intelligence tools
4. Development of automated reporting system

## References
- Istanbul Shopping Dataset (2021-2023)
