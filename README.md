# MNIST Overkill

## Using MNIST to try out MLOps and deployment approaches

Trying out iterative development of a ML service.
MNIST is chosen because of its small scale to keep costs
and compute requirements low. Rather than doing
groundbreaking ML research, the goal is to try out deployment
best practces.

## Roadmap

- [ ] Create a very basic ML model for a restricted subset
of the MNIST dataset in a notebook, to emulate the initial
research phase

- [ ] Create a Gradio web app to serve the model

- [ ] Use DVC to incorporate the full MNIST dataset, to 
emulate acquisition of new data

- [ ] Dockerize the web app

- [ ] Use Github Actions to automate testing of new models

- [ ] We'll see...
