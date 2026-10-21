xG estimates the probability that a shot becomes a goal, based on what we can measure about it, 
like distance and angle to goal, by learning from how often similar shots went in historically. 
It doesn’t predict individual shots. It measures the quality of chances, and goals will still be noisy around it.

We train on some shots to learn the general relationship between position and scoring, 
then test on shots it hasn’t seen to check that the relationship holds up and wasn’t just memorized.