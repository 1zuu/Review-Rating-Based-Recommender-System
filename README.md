#Review-Rating-Based-Recommender-System

In here I suggest a method to build a recommender system based on both reviews and rating of a item(cloth).
First feed the reviews into sentiment analysis and predict the sentiment score for the prediction.sentiment score for item m,

            sentiment score(m) = positive_count(m) / (positive_count(m) + negative_count(m))

Next predict the ratings for users, based on Item-Item Collaborative filtering amd using both these scores get the weighted average for final prediction.final score for item m,

            final_score(m) = alpha * sentiment_score(m) + (1 - alpha) * recommended_score(m)
