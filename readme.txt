Assignment:

In this assignment you are supposed to use the methodology
of reinforcement learning as we taught it during the class.
There are more advanced RL-algorithms and libraries, 
but we do not want you to use them.
Implement the RL algorithm by yourself. 
Game: Please use a 7x7 board and use the 8 different L's as winning shape.
(All rotations and reflections) The L's consist of 5 positions.
In this assignment you have two milestones to reach.

Groups: You can submit in a group of up to 3 students.
In case that there is someone who has not setup a neural network,
make sure this student also reaches the learning goal!
 

1. Milestone: (5 points)
Generate Game Data, by using self-play with the agents that
you created in the previous assignments (random/bandit/score/mcts).
Then train a neural network using your own data. 
At last test your neural network against four different agents:
random/bandit/mcts(200 iterations)/score

 

2. Milestone: (5 points)
Initialize a neural network with random weights and biases.
Create an agent that plays moves according to the neural network.
There are many possibilities on how to use the neural network.
(value-function/Q-function/policy)
Specifically, combine the neural network with mcts.
Train iteratively the neural network from the games that you generate
and then use the updated neural network to generate more games.
At last test your neural network against four different agents:
random/bandit/mcts(200 iterations)/score.

 

Submission:
The submission should consist of a report and the code that you used.
In the report you describe what you did and what your outcomes were. (4 pages max.)
The code should be testable by your grader.
In other words, the grader should be able to test your agent.

 
Deadline: 7th of November. 23:59. (Will be added to feedback fruits.)
 

Grading Criteria:
The learning goal of this assignment is to learn about some basic 
strategies to use neural networks in order to create strong agents. 
Therefore the main criteria is the correct usage of those methods. (70%)
However, it is important that your agent learned at least something.
Therefore the playing strength of your agent is also a criteria. (30%)
We would expect that both agents win consistently against a random agent.
A strong nn-agent should also win against mcts and bandit agents.
The evaluation with the score agent is problematic as the score agent is 
deterministic. 

 

As the grader, you should read the report and test the agent.
It should be clear to you which choices where made.
And if the performance that you test is consistent with the
performance in the report.