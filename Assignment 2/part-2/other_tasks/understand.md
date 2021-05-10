Notes => 
    - 3 arrows (at max). Indiana can at max carry only 2 materials.

Hyperparameters => 
    1. Gamma = 0.999
    2. Bellmon error = 10^-3 


State => [Class] 
    0. Position of Indiana => [character]
    1. Health of MM => [integer]
    2. Number of arrows [He has bow and blade too.] => integer
    3. Number of materials
    4. State of MM.

Actions =>
    every state will have some actions
    1. Centre Square
        a. Up, Down, Right, Left, Stay => P(Success) = 0.85 otherwise goto East Sqaure
        b. Shoot => P(success) = 0.5, if success damage = 25. [Conditional]
        c. Blade => P(success) = 0.1, if success damage = 50.
    
    2. North Sqaure
        a. Down, Stay => P(Success) = 0.85 otherwise goto East Sqaure
        b. Craft arrows [need atleast 1 material], P(+1) = 0.5, P(+2) = 0.35, P(+3) = 0.15

    3. South Sqaure
        a. Up, Stay => P(Success) = 0.85 otherwise goto East Sqaure
        b. Gather Material , P(+1) = 0.5, P(+2) = 0.35, P(+3) = 0.15

    4. East Square
        a. Left, Stay => P(success) = 1.0
        b. shoot, P(hit) = 0.9, damage = 25
        c. Blade, P(hit) = 0.2, damage = 50
    
    5. West Square
        a. Right, Stay => P(success) = 1.0
        b. Shoot, P(hit) = 0.25, damage = 25

Rewards =>
    1. For actions => -10 / Y = -10
    2. For states => 

Finish State => 
    when MM dies(i.e. 0 Health)



## Optimal
1. 2 arrows at C
    - move to E and hit => -20 + -20 + 0.9*0.9*25 
    - move to N and collect


## Simulate
- V[states]
- best_actions => select any with random probability.
- next state based on best action.