## CLaSP 410 Final Project Plan
Final project will aim to expand of the existing Lab 4 N-Layer atmosphere lab by introducing variable emissivity values at each atmospheric layer. The lab will have 3 questions, and assume that the first part of the lab has already been completed.

### Question 1 - Adding functionality
* Verify that your existing lab code works by writing a comprehensive unit test suite against a large set of reference solutions. Ensure your code passes these unit tests before proceeding.
* Add the functionality to have variable emissivity in each layer. This should have the ability to be passed in as a single digit, an array, or as a function.
* Add the following unit tests for a 5 layer Earth:
    1. A prescribed set of epsilons
    2. A linear decrease in epsilon
    3. An exponential decrease in epsilon

### Question 2 - Exploring Flexible Epsilon
* Set up an idealised Earth with 6 layers (representing the Earth as it is). Lets explore the following scenarios:
    1. Inject sulfates into stratosphere to increase its' emissivity
    2. Lots of WV in the troposphere to increase its' emissivity
* Compare with nuclear winter - what do we learn about the climate system?

### Question 3 - Adding Functionality
* Now lets flip the problem : instead of solving for temperature, 

