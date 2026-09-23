# Data Validation Report

## Dataset

Amazon Reviews 2023, Video Games category.

Files:

- `Video_Games.csv.gz`: 5-core user-product interactions
- `meta_Video_Games.jsonl.gz`: product metadata

## Interaction-data validation

- Rows: 814,586
- Missing values: 0
- Completely duplicated rows: 0
- Ratings observed: 1, 2, 3, 4 and 5
- Unique users: 94,762
- Unique products: 25,612

## Positive-interaction definition

The provisional positive-interaction definition is a rating of 4 or 5.

- Positive interactions: 654,867
- Users with at least one positive interaction: 93,949
- Users with at least three positive interactions: 88,137
- Eligible users as a percentage of all users: 93.01%

This threshold is suitable for the proposed temporal evaluation because it
retains most users while distinguishing clearly positive feedback.

## Decisions not yet finalized

- How repeated user-product interactions will be handled
- Whether ratings 1–3 will be retained as contextual signals
- Negative-sampling strategy
- Final temporal-splitting implementation
- Treatment of products without metadata

## Repeated user-product interactions

- Unique user-product pairs: 814,586
- Repeated user-product pairs: 0
- Rows belonging to repeated pairs: 0

No user-product aggregation is required because the interaction data has
already been de-duplicated.

### Worked example: one history, different objectives

Imagine the following user history:

| User | Product | Rating | Date |
|---|---|---:|---|
| A | Game X | 5 | January |
| A | Game X | 3 | March |
| A | Game X | 1 | June |

The same three records require different treatment depending on what the
recommendation system is designed to predict.

#### Unseen-product recommendation

The objective is:

> Recommend products the user has not interacted with before.

Once User A interacts with Game X, it becomes a seen product. The system
normally excludes it from future recommendation candidates, regardless of
whether the user rated it once or three times.

The model may still need one representation of the user's relationship with
Game X. Possible choices include:

- Most recent rating: 1
- Average rating: `(5 + 3 + 1) / 3 = 3`
- Maximum rating: 5
- A time-weighted combination

The latest rating may be the most appropriate representation of the user's
current opinion. In this example, the preference changed from strongly
positive to strongly negative.

The average produces a rating of 3, which makes the relationship appear
neutral and hides the direction of the preference change.

For an unseen-product recommender, a reasonable approach would be:

1. Exclude Game X from future recommendation candidates.
2. Use the latest rating as the current preference signal.
3. Retain earlier ratings as historical features if useful.

#### Repeat-purchase recommendation

Suppose the records represented purchases of a replenishable item rather than
ratings of a video game.

Repeated interactions could reveal:

- How often the product is purchased
- The average interval between purchases
- Whether the purchase frequency is changing
- When the user may need the product again

In this situation, reducing the history to one row would destroy useful
temporal information.

#### Explicit rating prediction

If the task is to predict the rating User A will give Game X, the target must
be defined carefully.

Possible targets include:

- The user's most recent rating
- The user's next rating
- The user's average historical rating
- Whether the user currently likes or dislikes the product

These are different prediction problems. Choosing one changes the meaning of
the model and its evaluation.

#### Information that can be preserved

The history can be summarized using:

| Feature | Value |
|---|---:|
| Interaction count | 3 |
| First rating | 5 |
| Latest rating | 1 |
| Average rating | 3 |
| Rating change | -4 |

The `rating_change` is calculated as:

```text
latest rating - first rating = 1 - 5 = -4

#### Temporal-leakage example

Suppose the system generates a recommendation in April.

At that moment, it may use:

- January rating: 5
- March rating: 3

It must not use:

- June rating: 1

The June rating had not happened yet. Using it to generate or evaluate an
April recommendation would give the model information from the future and
cause temporal leakage.

The correct "latest rating" in April is therefore 3, not 1.

> **Note: This is a hypothetical educational example.**
>
> The actual Amazon Video Games 5-core dataset used in this project contains
> no repeated user-product pairs. The example is included to demonstrate why
> preprocessing decisions must follow the system's prediction objective rather
> than applying one generic rule to every recommender system.

## Metadata-record coverage

Every product in the interaction file was matched to a metadata record using
the shared `parent_asin` identifier.

- Products in interaction data: 25,612
- Products matched to metadata: 25,612
- Interaction products without metadata: 0
- Product metadata coverage: 100.00%
- Rating rows involving products without metadata: 0
- Users affected by missing product metadata: 0

This confirms complete record-level metadata coverage. It does not imply that
every field inside every metadata record is populated.

## Metadata field completeness

For the 25,612 metadata records corresponding to interaction products:

| Field | Missing or unusable | Percentage |
|---|---:|---:|
| Price | 8,477 | 33.10% |
| Main category | 514 | 2.01% |
| Blank title | 1 | <0.01% |
| Empty description | 6,816 | 26.61% |
| Unusable feature text | 3,305 | 12.90% |
| Empty categories | 869 | 3.39% |
| Empty images | 1 | <0.01% |
| Neither description nor features | 1,590 | 6.21% |
| No usable textual metadata | 0 | 0.00% |

Every interaction product has at least one usable text source among title,
description, features and categories. Content-based methods can therefore use
a combined text representation with field-level fallbacks.

Price will not be treated as a mandatory feature because it is unavailable
for approximately one-third of products. Missing price must not be replaced
with zero because zero represents a free product rather than an unknown price.

## Planned exploratory-data-analysis visualizations

The project will create visualizations only when they answer a defined
analytical question.

Planned charts include:

- Rating distribution
- Interactions per user
- Interactions per product
- Cumulative product-popularity concentration
- Interaction activity over time
- Metadata-field completeness
- Exact numeric price distribution
- Metadata average-rating distribution
- Product-category distribution
- User eligibility under different history thresholds

Highly skewed user, product and price distributions should use log-scaled or
outlier-aware visualizations where appropriate.

Very small data-quality cases, such as the three lower-bound price strings,
will be documented in text or tables rather than given misleading standalone
charts.

## Numerical metadata validation

- Average ratings outside the 1-to-5 range: 0
- Negative product-page rating counts: 0
- Exact numerical prices available: 17,131
- Missing or non-exact numerical prices: 8,481
- Exact zero prices: 6
- Negative prices: 0
- Median exact price: $29.91
- Mean exact price: $52.91
- Maximum exact price: $2,189.75

The six zero-price products are downloadable PC or online titles, making a
free-product interpretation plausible. They were retained as valid numerical
prices.

Three `from` prices were not converted into exact prices because they represent
lower bounds. One dash value was also treated as unavailable. The original
price values remain preserved for audit and display.

Metadata-snapshot average ratings and rating counts will not automatically be
used as temporal model features because they may contain future information.
Leakage-safe product statistics must be calculated from the appropriate
historical training interactions.