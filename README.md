# rayn

This is a relatively simple path-tracing based renderer. It was originally based on the book "Ray tracing in one weekend" by Peter Shirley, which I heartily recommend, though it has now evolved into a structure of my own design, taking hints from both that and `pbrt`, which is also an excellent reference.

# Features

* Physical light transport algorithm
* Importance sampling (soon multiple importance sampling)
* Multiple-bounce indirect lighting/global illumination
* Signed distance field rendering through leveraging [`sdfu`](https://github.com/termhn/sdfu/)

### Demo image

![demo](/render.png?raw=true)