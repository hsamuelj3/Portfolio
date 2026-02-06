# Controls Applications

As described in the portfolio readme.md this folder contains projects to implement various control theories in python. The projects are listed below:

## Blockbeam

The "blockbeam", or "block on a beam," system is another version of the ball on a beam system, where a block slides on a beam which is connected at one end to a motor (I choose to ignore sliding friction). This motor can apply a torque input to the system. The only known values or the value $\theta$ and the position of the block, often referred to as $z$ or $x$ (I chose to use $z$ in my project).

The control theories determine what force to apply so that the position ($z$) of the block tracks a certain reference value.

## InversePendulum

Similar to the blockbeam system the inverse pendulum system has one input: the force pushing on the side of the block or cart; and two outputs: the position $z$ of the cart on its track, and angle $\theta$ of the rod. As the name implies, the pendulum should be inverted and remain upright.

The implemented control theories will determine what force pushes the cart so that the cart position ($z$) tracks a reference value - all the while the rod remains upright.
