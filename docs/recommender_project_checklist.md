# Recommender-System Project Checklist

Use this checklist when planning, building, evaluating and documenting a recommender-system project.

---

## 1. Define the recommendation problem

- [ ] Who will receive recommendations?
- [ ] What items will be recommended?
- [ ] Are we recommending unseen items, repeat purchases or both?
- [ ] What counts as a successful recommendation?
- [ ] Are we predicting ratings, clicks, purchases or the next interaction?
- [ ] How many recommendations should be displayed?
- [ ] Which products must be excluded?
- [ ] What should happen for a new user?
- [ ] What should happen for a new product?

### Why this matters

The recommendation objective determines how the data should be processed, how the model should be trained and how success should be measured.

---

## 2. Understand the datasets

- [ ] Identify every data file.
- [ ] Record the source of each file.
- [ ] Count rows and columns.
- [ ] Inspect column names.
- [ ] Inspect data types.
- [ ] Count unique users.
- [ ] Count unique products.
- [ ] Count total interactions.
- [ ] Check missing values.
- [ ] Check blank strings.
- [ ] Check whitespace-only strings.
- [ ] Check empty lists and dictionaries.
- [ ] Check completely duplicated rows.
- [ ] Check repeated user–product interactions.
- [ ] Validate rating ranges.
- [ ] Validate timestamp units.
- [ ] Convert timestamps into readable dates.
- [ ] Check the earliest and latest timestamps.
- [ ] Check tied timestamps.
- [ ] Match interaction products to metadata.
- [ ] Measure metadata coverage.
- [ ] Measure user–product matrix sparsity.
- [ ] Identify possible cold-start users and products.

---

## 3. Define the feedback

- [ ] Determine whether feedback is explicit or implicit.
- [ ] Define what counts as a positive interaction.
- [ ] Decide how lower ratings will be used.
- [ ] Decide how missing interactions will be interpreted.
- [ ] Decide how repeated interactions will be handled.
- [ ] Document all feedback decisions.

### Important definitions

**Explicit feedback** is information deliberately supplied by a user, such as a rating from 1 to 5.

**Implicit feedback** is behaviour such as a click, view, purchase, save or watch event.

**Unobserved interaction** means the dataset contains no interaction between that user and product.

Unobserved does not automatically mean disliked.

### Example provisional policy

- Ratings 4 and 5: positive interactions
- Ratings 1 to 3: non-positive or contextual evidence
- Missing user–product interaction: unknown preference

A product is not permanently positive or negative. A particular user has a positive or non-positive interaction with that product.

---

## 4. Perform meaningful exploratory data analysis

- [ ] Plot the rating distribution.
- [ ] Plot interactions per user.
- [ ] Plot interactions per product.
- [ ] Measure product-popularity concentration.
- [ ] Plot interaction volume over time.
- [ ] Plot positive-interaction share over time.
- [ ] Plot useful metadata coverage.
- [ ] Inspect category distributions.
- [ ] Inspect meaningful numeric features.
- [ ] Identify outliers.
- [ ] Document unusual records.
- [ ] Explain every chart in plain language.
- [ ] State what every chart does not prove.

### EDA rule

Every chart must answer a useful question.

Use graphs to show meaningful patterns. Use exact calculations and tables for rare or unusual cases.

---

## 5. Design evaluation before training

- [ ] Define the training data.
- [ ] Define the validation targets.
- [ ] Define the test targets.
- [ ] Respect chronological order when appropriate.
- [ ] Decide whether the split is user-level chronological or global-time-based.
- [ ] Prevent future information from entering training features.
- [ ] Check for tied target timestamps.
- [ ] Define how ambiguous targets will be handled.
- [ ] Measure how many users remain eligible.
- [ ] Document evaluation coverage.
- [ ] Keep the test set untouched during model selection.
- [ ] Fit preprocessing transformations using training data only.

### Target definitions

A **validation target** is a hidden correct answer used to compare models and choose settings.

A **test target** is a hidden final answer used after the model and settings have been selected.

### User-level chronological example

For one user:

1. Earlier positive interactions become training history.
2. The second-latest positive interaction becomes the validation target.
3. The latest positive interaction becomes the test target.

### Important limitation

A user-level chronological split respects time within each user, but it is not identical to a global deployment simulation.

A global-time split chooses one calendar cutoff and trains only on information recorded before that cutoff.

---

## 6. Establish simple baselines

- [ ] Random baseline for a basic sanity check
- [ ] Most-popular-products baseline
- [ ] Category-popularity baseline
- [ ] Basic content-similarity baseline
- [ ] Basic collaborative-filtering baseline

### Why baselines matter

A complicated model should demonstrate that it performs better than a simple method.

If a sophisticated model cannot beat a popularity baseline, its additional complexity may not be justified.

---

## 7. Build models progressively

Recommended progression:

1. Popularity recommender
2. Content-based recommender
3. Collaborative filtering
4. Matrix factorization
5. Hybrid recommender
6. Sequential or neural recommender, if justified

For every model:

- [ ] State what information it uses.
- [ ] State what it predicts.
- [ ] Explain how it produces scores.
- [ ] Record its configuration.
- [ ] Compare it with the same evaluation protocol.
- [ ] Record strengths and weaknesses.
- [ ] Avoid unnecessary complexity.

---

## 8. Construct valid recommendation candidates

- [ ] Decide which products are eligible for recommendation.
- [ ] Exclude already-seen products when recommending unseen products.
- [ ] Exclude unavailable or prohibited products.
- [ ] Handle products with missing optional metadata.
- [ ] Decide how new products are represented.
- [ ] Decide how new users receive recommendations.
- [ ] Score candidates consistently.
- [ ] Rank candidates from strongest to weakest score.
- [ ] Return only the required Top-K products.

### Important principle

Candidate generation and ranking are different stages.

Candidate generation finds a manageable collection of possible products.

Ranking orders those candidates according to their predicted relevance for the user.

---

## 9. Select suitable evaluation metrics

### Ranking-quality metrics

- [ ] Hit Rate@K
- [ ] Recall@K
- [ ] Precision@K, when appropriate
- [ ] Mean Reciprocal Rank
- [ ] NDCG@K

### Recommendation-behaviour metrics

- [ ] Catalog coverage
- [ ] Popularity bias
- [ ] Novelty
- [ ] Diversity
- [ ] Personalization

### Segment evaluation

Compare performance for:

- [ ] Users with short histories
- [ ] Users with long histories
- [ ] Popular products
- [ ] Less-popular products
- [ ] Products with rich metadata
- [ ] Products with limited metadata

### Metric principle

Do not report only one metric.

A model can have good ranking accuracy while recommending the same popular products to everyone.

---

## 10. Protect against data leakage

- [ ] Do not use future interactions to construct historical features.
- [ ] Do not use the test set to choose models.
- [ ] Do not use the test set to tune hyperparameters.
- [ ] Do not calculate preprocessing statistics from the complete dataset.
- [ ] Check whether metadata fields are historical or current snapshots.
- [ ] Fit encoders, scalers and vectorizers using training data only.
- [ ] Document every field that may contain future information.

### Temporal-leakage example

If a recommendation is being evaluated as though it occurred in April, it may use information available in January and March.

It must not use information recorded in June.

---

## 11. Perform error analysis

- [ ] Find users for whom the model performs poorly.
- [ ] Inspect missed validation and test targets.
- [ ] Inspect incorrect high-ranked products.
- [ ] Check whether recommendations are excessively popular.
- [ ] Check whether recommendation lists are repetitive.
- [ ] Check whether users receive nearly identical lists.
- [ ] Inspect cold-start behaviour.
- [ ] Inspect products with incomplete metadata.
- [ ] Compare failure patterns across models.
- [ ] Preserve representative failure examples.

### Why this matters

Metrics tell us how much the model succeeded.

Error analysis helps explain where and why it failed.

---

## 12. Explain recommendations

- [ ] Provide a reason for each recommendation.
- [ ] Use evidence from the user’s history.
- [ ] Use relevant product similarities.
- [ ] Show model contributions when appropriate.
- [ ] Use plain language.
- [ ] Avoid causal claims.
- [ ] Provide fallbacks when metadata is missing.
- [ ] Distinguish global model importance from local recommendation explanations.

### Example explanation

Recommended because the product shares platform and adventure-related features with games the user previously rated highly.

This is an association-based explanation. It does not prove that a particular feature caused the user’s preference.

---

## 13. Plan the user interface

- [ ] Show the recommended product.
- [ ] Show a recommendation score carefully.
- [ ] Do not present an uncalibrated score as a probability.
- [ ] Show a local explanation for each recommendation.
- [ ] Show relevant metadata.
- [ ] Provide fallbacks for missing images or descriptions.
- [ ] Clearly label experimental features.
- [ ] Avoid overwhelming the user with technical details.
- [ ] Provide an optional educational section for deeper explanations.

---

## 14. Make the project reproducible

- [ ] Preserve raw data separately.
- [ ] Save processed data deterministically.
- [ ] Record preprocessing rules.
- [ ] Fix random seeds where randomness is necessary.
- [ ] Record package versions.
- [ ] Save model configurations.
- [ ] Save trained models when appropriate.
- [ ] Save evaluation outputs.
- [ ] Save representative predictions.
- [ ] Document how to rerun each stage.
- [ ] Keep notebooks in a clear execution order.
- [ ] Move reusable code into source files.
- [ ] Add automated tests for reusable functions.

---

## 15. Document evidence and limitations

- [ ] Record what was implemented.
- [ ] Record what was tested.
- [ ] Record what was formally evaluated.
- [ ] Record the metrics and results.
- [ ] Record representative failures.
- [ ] Record data limitations.
- [ ] Record model limitations.
- [ ] Record unresolved decisions.
- [ ] Distinguish supported conclusions from hypotheses.
- [ ] Avoid unsupported claims such as “production-ready.”

### Evidence levels

**Implementation:** The feature or model runs.

**Testing:** Selected expected and failure cases were checked.

**Evaluation:** Performance was systematically measured using defined data and metrics.

**Production validation:** Performance was observed under real operating conditions.

These are different levels of evidence.

---

## 16. Production considerations

- [ ] Measure prediction latency.
- [ ] Measure memory and storage requirements.
- [ ] Plan model updates.
- [ ] Monitor interaction drift.
- [ ] Monitor product-catalog changes.
- [ ] Monitor recommendation coverage.
- [ ] Monitor popularity concentration.
- [ ] Monitor failures and missing metadata.
- [ ] Define fallback recommendations.
- [ ] Consider privacy and data retention.
- [ ] Consider feedback loops.
- [ ] Plan online evaluation when appropriate.

Offline success does not guarantee real-world success. A deployed recommender may require controlled online experiments and monitoring.

---

# Common mistakes to avoid

- [ ] Do not treat unobserved interactions as confirmed dislikes.
- [ ] Do not randomly split time-dependent data without justification.
- [ ] Do not use future information in historical features.
- [ ] Do not tune models using test results.
- [ ] Do not evaluate only with classification accuracy.
- [ ] Do not silently remove inconvenient records.
- [ ] Do not average repeated ratings without considering the objective.
- [ ] Do not assume a zero price definitely means free.
- [ ] Do not assume metadata snapshot values existed historically.
- [ ] Do not assume the most complicated model is the best.
- [ ] Do not present recommendation scores as probabilities unless calibrated.
- [ ] Do not claim feature importance proves causation.
- [ ] Do not claim offline evaluation proves production performance.

---

# Recommended learning order

1. Python fundamentals
2. Python data structures
3. pandas and NumPy
4. Data visualization
5. Basic statistics
6. Machine-learning fundamentals
7. Vectors, matrices and similarity
8. Content-based recommendation
9. Collaborative filtering
10. Matrix factorization and embeddings
11. Ranking evaluation
12. Data leakage and temporal evaluation
13. Hybrid recommendation
14. Error analysis and explainability
15. Deployment and monitoring

---

# Questions I should be able to answer

At the end of a recommender project, I should be able to explain:

1. What problem does the system solve?
2. Who are the users and what are the products?
3. What does one interaction row represent?
4. What counts as positive feedback?
5. How are missing interactions interpreted?
6. How sparse is the user–product matrix?
7. How were training, validation and test data created?
8. How was temporal leakage prevented?
9. Which baseline models were used?
10. Which recommendation models were compared?
11. Which metrics were selected and why?
12. Did the model beat the baselines?
13. Where did the model fail?
14. Is the model biased toward popular products?
15. How are new users and products handled?
16. How is each recommendation explained?
17. Which claims are supported by evidence?
18. What limitations remain?
19. What would be improved next?
20. What would still be required before production deployment?

---

# Useful learning resources

- Google Recommendation Systems course:
  https://developers.google.com/machine-learning/recommendation

- Microsoft Recommenders:
  https://github.com/recommenders-team/recommenders

- TensorFlow Recommenders:
  https://www.tensorflow.org/recommenders

- Microsoft Research: Evaluating Recommendation Systems:
  https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/EvaluationMetrics.TR_.pdf

- NVIDIA Merlin:
  https://developer.nvidia.com/merlin