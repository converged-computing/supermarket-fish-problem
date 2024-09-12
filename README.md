# The Supermarket Fish Problem

This is part of the [Performance study](https://github.com/converged-computing/performance-study) and the single-node-benchmark analysis. The analyses afforded generation of a lot of intermediate data and a web interface, and were moved here for better organization.

[![DOI](https://zenodo.org/badge/837429553.svg)](https://zenodo.org/doi/10.5281/zenodo.13738495)

Have you ever been to the supermarket and ordered white fish? You may be getting tilapia, flounder, branzino, catfish, cod, haddock, hake, halibut, pollock, sea bass, sole, or whiting. The same is true for cloud CPU architectures. You may know that you are getting some flavor of Intel, but it's unclear if it's Skylake, Icelake, Sandy Bridge, or some other flavor. We did a large [performance study](https://github.com/converged-computing/performance-study) in August 2024 that looked across many different environments, clouds, and instance types, and can now reflect on what we found. In the case of finding a potpourri of architectures, we call this the supermarket fish problem.

 - 🐠🐟 [View the Fish!](https://converged-computing.org/supermarket-fish-problem/web/machines/)

**Under development** data processing is underway - a table will be added to each view!

## Generate

Make some pngs (they render better in react):

```bash
for filename in $(find . -name machine.svg)
  do
    echo $filename
    directory=$(dirname $filename)
    outpng="$directory/machine.png"
    echo inkscape $filename -o $outpng
    inkscape $filename -o $outpng
done
```

To generate data for the gallery:

```bash
python 1-generate-gallery.py
```
## License

HPCIC DevTools is distributed under the terms of the MIT license.
All new contributions must be made under this license.

See [LICENSE](https://github.com/converged-computing/cloud-select/blob/main/LICENSE),
[COPYRIGHT](https://github.com/converged-computing/cloud-select/blob/main/COPYRIGHT), and
[NOTICE](https://github.com/converged-computing/cloud-select/blob/main/NOTICE) for details.

SPDX-License-Identifier: (MIT)

LLNL-CODE- 842614
