# symmpy
Symmetry operations module for crystallography

# Functions

*clean*: Used for translating a vector outside of the unit cell back into the unit cell fractional space.

*translate*: Simple translation of a vector.

*rotate*: Rotation of a vector around an arbitrary axis.

*reflect*: Reflection through an arbitrary plane.

*invert*: Inversion through an inversion centre located at an arbitrary position.

*rotoinvert*: (Improper Rotation) Convenient combination of *rotate* and *invert* functions.

*screw*: (Screw-axis) Convenient combination of *rotate* and *translate* functions.

*glide*: (Glide-plane) Convenient combintation of *relfect* and *translate* functions.

# Requirements

Written and tested with
> Python 3.6
> Numpy 1.14
