Codename: TeaStain
==================
This is a toy project. The idea is to develop a genetic algorithm for creating
image filters.

This will be a user-driven genetic algorithm. The user will provide an image to
filter, and the system will display a number of filtered images. The user will
then choose their favorites (ranked in order from most liked to least liked),
and the system will mix the underlying filters with some randomness applied to
generate candidates for the next iteration.

My initial implementation of image filters will be a simple convolution matrix.
This is limited, sure, but it's also very easy to mix several matricies using
basic matrix ops, so it'll do for now. I'll assume that each image has 3
channels (rgb), so convolution matricies will be of size nxnx3 where each
"slice" of the matrix is applied seperately to each image channel.

I'll implement the interface as a web-app with a python backend. I could
probably do everything in the client, but I like python better than javascript.
Maybe I'll move to all-javascript later.
