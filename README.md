# Fintech Causal Growth Engine

A project that answers one question: does a fintech incentive campaign actually work, or does it just look like it works?

## The problem

Growth teams run incentive campaigns all the time, cashback, referral bonuses, signup credits. The obvious way to check if they work is to compare users who got the incentive against users who didn't. The problem is that users who opt into promotions are usually already the most active, highest balance users. A simple before and after comparison ends up crediting the incentive for growth that would have happened anyway.

This project builds a small pipeline that corrects for that, measures how confident it should be, and checks its own assumptions before trusting the result.

## What it does

1. Generates a synthetic fintech dataset where incentive assignment is deliberately biased toward active, high balance users
2. Corrects the biased estimate using propensity score weighting
3. Estimates how the effect changes across user segments using a T-learner
4. Wraps conformal prediction intervals around individual estimates, so predictions come with a confidence range instead of a single number
5. Validates the entire pipeline against a held out synthetic randomized experiment
6. Checks where the model's assumptions start to break down, using overlap diagnostics and a confounding sensitivity analysis
7. Runs a cost sensitivity analysis to find when targeting beats giving the incentive to everyone
8. Serves live predictions through a FastAPI endpoint

## Key findings

A naive comparison overstates the incentive's effect by 34 percent, 16.75 versus a true effect of 12.48. Correcting for the selection bias with propensity weighting brings the estimate to 12.62, within 1 percent of the true value.

The effect also isn't the same for everyone. It scales with user activity, from around 5 for low activity users up to over 20 for the most active ones. A T-learner recovers this pattern closely, matching the true effect within a few percent across every activity bucket.

The confidence intervals around these estimates are checked, not assumed. They cover the true value 97 percent of the time against a 90 percent target, on the conservative side, but honest.

To make sure the whole approach isn't just self-consistent nonsense, it's tested against a held out group where treatment was assigned completely at random, a synthetic randomized experiment. The model's prediction on this group, 12.72, lands within 0.01 of the actual gold standard result, 12.71.

The pipeline also checks its own limits. Overlap between treated and untreated users gets thin above 70 percent activity, so estimates in that range carry more uncertainty. Restricting the analysis to users with reasonable overlap barely changes the answer, 12.44 versus 12.62, which suggests the main result isn't an artifact of that thin region. A separate test simulating an unmeasured confounder shows the estimate stays stable under mild hidden bias but drifts by about 2.4 points under strong hidden bias, a limit worth stating rather than ignoring.

Finally, the cost of the incentive matters. Below about 3.50 dollars per user, giving it to everyone works about as well as being selective. Above that, targeting based on the model clearly wins.

## Repository layout

    fintech_causal_engine.ipynb   full analysis, phase by phase
    main.py                       FastAPI endpoint for live scoring
    model_treated.pkl             trained model for treated users
    model_control.pkl             trained model for control users
    q_combined.pkl                saved conformal margin used by the API

## Running it

Open the notebook and run the cells top to bottom. Each phase is marked with a heading.

To run the API, from the same folder, "run uvicorn main:app --reload", then visit "http://127.0.0.1:8000/docs" to send a test request to "/score_uplift".

## Limitations

The data is synthetic. A real deployment would need to check for confounders that aren't in this dataset at all, not just simulate one. The conformal intervals are calibrated a bit wider than necessary, 97 percent instead of 90 percent, and could be tightened with more calibration data. Estimates for the top activity decile are less reliable due to a smaller sample size there. The API itself is a working prototype, not production infrastructure, it has no authentication, logging, or error handling, all of which a real deployment would need.
