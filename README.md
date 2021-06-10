# BGG-fixed-ratings
[BGG](https://boardgamegeek.com/) is the biggest board game (and not only) encyclopedia. If you want information about a tabletop game that is the right place. 

Thousands of users all over the world comment and review games so that other users can evaluate a game before buying it... BUT hundreds of comments are not rated by users and this makes the general review of a game not properly correct. 

In this notebook we tried to give a rate to all those unrated comments and assigned to the hottest (top 50) games, a more accurate rating

## BoardGameGeek: ratings and comments 
In each page on https://boardgamegeek.com/, a detailed list of information is returned for each game, such as: 
-	Number of players, 
-	Playing time, 
-	Designer, 
-	…


![Image1](https://github.com/informatica92/BGG-fixed-ratings/blob/main/static/images/1.png)

Also, users can add their own reviews in order to share their thought about the game in subject or simple comments that do not have a rating but in some way are another instruments for the site’s users to let others know what they think about the game.

![Image2](https://github.com/informatica92/BGG-fixed-ratings/blob/main/static/images/2.png)

![Image3](https://github.com/informatica92/BGG-fixed-ratings/blob/main/static/images/3.png)

Our idea is to assign a hypothetical score to these comments in order to better understand the users’ preferences.

## Assign a score to comments
What we thought about is to use all the ratings available on the platform (with text and score) in order to fine-tune a pre-trained model and use it to give a score to the un-rated comments

<p align="center">
  <img src="https://github.com/informatica92/BGG-fixed-ratings/blob/main/static/images/4.png" alt="Image4"/>
</p>

In this way learning on top of the ratings: 

![Image5](https://github.com/informatica92/BGG-fixed-ratings/blob/main/static/images/5.png)
 
We can assign a score to simple comments like this: 

![Image6](https://github.com/informatica92/BGG-fixed-ratings/blob/main/static/images/6.png)
![Image7](https://github.com/informatica92/BGG-fixed-ratings/blob/main/static/images/7.png)

## Code key points
### Data acquisition
Let's start collecting the hottest games in order to get some titles to get reviews and comments from with: 
```python
hot_array = get_hot_data()
hot_array[:2]
```
and then let's use this list to get corresponding reviews and comment using another utility function: 
```python
comments_df = get_comments(hot_array, verbose=10)  # verbose=10 means print a row each 10 iterations
```
### Data cleaning
Remove URLs from ratings/comments
```python
comments_df['value'] = [re.sub(r"http\S+", "", v) for v in comments_df.value.values]
```
Remove comments under a specific length
```python
comments_df = remove_short_comments(comments_df, MIN_COMMENT_LEN)
```
### Datasets creation
Let's split rated and not-rated comments:
```python
# get rated comments only
rated_comments = comments_df.query('rating != "N/A"')

# get non rated comments only
not_rated_comments = comments_df.query('rating == "N/A"').reset_index(drop=True)
```
### Classifier training
We decided to use a scikit-learn wrapper in order to have access to the GridSearchCV method that performs a traning based on Cross Validation check, in this way we can be sure that the performances we get are not influenced by the training/validation split  
```python
def build_classifier():
    return build_model(hub_layer=None, pre_trained_model_name=MODEL_NAME, model_type='classifier', verbose=0)

estimator = KerasClassifier(build_fn=build_classifier, epochs=100, batch_size=1024, verbose=2, validation_split=VAL_FRACTION)
x_train_clf = np.array(list(rated_comments.value))
y_train_clf = np.array(list((rated_comments.rating.astype(float)>=GOOD_REVIEW_THRESHOLD).astype(int)))

clf = GridSearchCV(
    estimator, 
    cv=3, 
    param_grid={}
)
clf.fit(x_train_clf, y_train_clf, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.001)])
```
The resulting model returned the following training charts: 

<table>
  <tr style="border-collapse: collapse; border: none;">
    <td>
      <img src="https://github.com/informatica92/BGG-fixed-ratings/blob/main/static/images/8.png"/>
    </td>
    <td>
      <img src="https://github.com/informatica92/BGG-fixed-ratings/blob/main/static/images/9.png"/> 
    </td>
  </tr>
</table>

and we can be quite confident saying it is a good model.

### Regressor training
Let's now try to train a classifier instead using a very similar approach:
```python
def build_regressor():
    return build_model(hub_layer, pre_trained_model_name=MODEL_NAME, model_type='regressor', verbose=0)


estimator = KerasRegressor(build_fn=build_regressor, epochs=100, batch_size=512, verbose=0, validation_split=VAL_FRACTION)
x_train_reg = np.array(list(rated_comments.value))
y_train_reg = np.array(list(rated_comments.rating.astype(float)))

clf = GridSearchCV(
    estimator, 
    cv=3, 
    param_grid={}
)
clf.fit(x_train_reg, y_train_reg, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=5, min_delta=0.001)])
```
that returns the following training chart (here loss and metric - mean squared error - match):

![Image10](https://github.com/informatica92/BGG-fixed-ratings/blob/main/static/images/10.png)

### Model comparison
```python
not_rated_comments = not_rated_comments.sample(frac=1)
inputs = list(not_rated_comments.value.astype(str))[:10]

clf_results = classifier.predict(inputs, verbose=0)
reg_results = regressor.predict(inputs, verbose=0)
for i in range(len(inputs)):
    print(f"""\"{inputs[i]}\"
    reg score: {reg_results[i]:.2f}
    clf score: {clf_results[i][0]}
""")
```
looking at some comments and evaluating the scores assigned by the two models we can easily notice that regressor is a bit more accurate and the scores assigned are more reasonable. For this reasin we decied to continue the study with the **regressor**

## TODO:
 * ~~exclude non-english comments/reviews~~
 * ~~exclude very short comments/reviews~~
 * ~~added regressor~~
 * ~~compare regressor vs classifier~~
 * ~~clip regressor results~~
 * LSTM in build_model
 * find a better (faster) way to get all comments for hottest games
