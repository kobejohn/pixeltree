pixeltree
=========

*pixeltree* converts each region of an image into a tree, for example, a tracing of a hand into connected branches between fingers and wrist.

Backstory
---------

This was motivated by a StackOverflow question [Building tree/graph from image points][question], I just couldn't let it go until I had something working.

[question]: http://stackoverflow.com/questions/20730166/building-tree-graph-from-image-points

Current status
--------------

It's more of a demo and not setup to work as in CLI or even as a library right now. It wouldn't take much work to get it there though.

Here are some example images:

![Pre-reduced hand](/demo_results/hand.png)
![Image with large areas](/demo_results/fat.png)
![Image with various test sections](/demo_results/all.png)

Dependency
----------

* Python
* NumPy
* OpenCV

License
-------

The MIT License, see [`LICENSE`](LICENSE).
