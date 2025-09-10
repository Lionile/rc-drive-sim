# Fixes/addons
- add car bounding box which rotates with the car. The bounding box is the same size as the car sprite
- map generator already gives track edge segments as a series of edges. use those edges to detect car collisions, and sensor distance
- new controllers folder. Base controllers, other controllers inherit interface.
- change main file to a run file that takes a wanted controller as an argument, and loads the correct one
- make rendering optional, allow headless training. Make rendering toggleable during training for checking car dynamics/progress
- remove close function from environment, main code should be the one closing
- in env step, add distinction between episode termination and truncation (time limit hit)
- make the side sensor rays start form the actual sensors, not the middle (move them about 1.5 cm forward in the direction of each respective ray)
- make all of the occurences of the update frequency (60Hz) be controlled by the main code, so I dont forget to change the frequency in one place.