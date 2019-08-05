# KellyCriterionRL

From [Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion):

The Kelly Criterion is to bet a predetermined fraction of assets and can be counterintuitive. In one study,<sup>[[5]](https://en.wikipedia.org/wiki/Kelly_criterion#cite_note-5)[[6]](https://en.wikipedia.org/wiki/Kelly_criterion#cite_note-6)</sup> each participant was given $25 and asked to bet on a coin that would land heads 60% of the time. Participants had 30 minutes to play, so could place about 300 bets, and the prizes were capped at $250. Behavior was far from optimal. "Remarkably, 28% of the participants went bust, and the average payout was just $91. Only 21% of the participants reached the maximum. 18 of the 61 participants bet everything on one toss, while two-thirds gambled on tails at some stage in the experiment." Using the Kelly criterion and based on the odds in the experiment, the right approach would be to bet 20% of the pot on each throw (see first example below). If losing, the size of the bet gets cut; if winning, the stake increases.

## What's in this repo?

The above paragraph described a bankroll management problem, gave metrics to how most humans perform at it, and presented the optimal solution.

I wondered if a reinforcement learning agent could be trained to perform better at this game than the participants of this study; or if it would work out the optimal strategy (Kelly Criterion) for itself, without being explicitly programmed.

## Training runs

### 8th August 2019

After a training run of 30 million episodes, with randomly sampled starts

![Rewards during training](.README_images/2019-08-05-training-rewards.png)

![State-action pairs evaluated during training](.README_images/2019-08-05-training-n-values.png)

![Learned policy](.README_images/2019-08-05-learned-policy.png)

![Learned Q values](.README_images/2019-08-05-learned-q-values.png)


### 2nd August 2019

After an overnight training run of 2 million episodes

![Rewards during training](.README_images/2019-08-02-training-rewards.png)

![Learned policy](.README_images/2019-08-02-learned-policy.png)

![Learned Q values](.README_images/2019-08-02-learned-q-values.png)
