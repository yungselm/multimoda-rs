We thank @richardkoehler for the detailed evaluation of our manuscript and the appreciation of our work. These were some very helpful suggestions which resulted in a great improvement of the codebase and manuscript.

We changed the following:

## Performance
- [x] There is a performance claim in the paper on lines 102-104. There should be code, and ideally sample data to reproduce this claim: Empirical performance on a 16-core CPU: an OCT pullback with 280 frames and a rotation search range of ±3° (final accuracy 0.01°) saw alignment time reduced from 150𝑠 to 18𝑠 with the optimized multiscale search.

*Thank you for this very helpful suggestion. I have now added a complete benchmark suite testing the algorithmic improvements and additionally the parallelization improvements. I additionally added a section in the docs highlighting these results. This focuses only on the alignment algorithms, since I didn't have the ideas/time to improve the labeling algorithms so far. Therefore I also highlighted this in the paper:*

- Changed title from **Performance and parallelization** to **Alignment algorithm performance and parallelization**

## General
- [x] Scipy is missing as dependency for mm.label_geometry(), I think it needs to be added to ``pyproject.toml``

*Added scipy to ``pyproject.toml``*

- [x] I would suggest to use dependency-groups instead of optional-dependencies in pyproject.toml, since optional-dependencies are shipped with the package when you publish it, but dependency-groups are not. In your case, you have additional dependencies that do not need to be shipped, like docs and test. An example for the use of optional-dependencies would for example be if you wanted to install support for excel with pandas, in which case you would do: pip install "pandas[excel]". See here: https://packaging.python.org/en/latest/specifications/dependency-groups/

- [x] Using dependency-groups you can also solve the problem of listing the dependencies that are necessary to run the example notebook, since common package and environment managers like uv use the dependency-group dev for these types of dependencies. The dev dependencies are installed by default when developing, but are like all dependency-groups not distributed with the package when publishing. So I would suggest listing the dependencies necessary for running the notebook (trimesh plotly scipy nbformat and ipykernel) in the group dev.

*These two comments were very helpful. I was not yet familiar with the differences especially regarding shipping. I have now completely revised the ``pyproject.toml`` file leaving only pyglet and pymeshlab as optional dependencies. pyglet is used for debug plotting, while pymeshlab is only used for some improved mesh fixing. Since pymeshlab has a GPL license I wanted to leave this option to the user.*

## Tutorials
- [x] In general, it is not possible to follow the tutorials step-by-step, since for example not all the code necessary is included and there are sometimes changes in variable names. I therefore entirely agree with this comment by crnh:

*Thank you very much for highlighting this. I completely agree that the notebooks caused the biggest issue so far. I really liked the suggestion by @crnh and therefore now provide the prerun notebooks directly on the documentation and added a CI pipeline to check that they run without error. I updated the complete documentation accordingly.*

- [x] Remove tuple brackets from (0.0, 1.0, 2.0) here since this results in an error: contour_trsl = contour_rot.translate((0.0, 1.0, 2.0))

*Removed the tuple brackets*

## The examples zip folder
- [x] Is there a reason the examples folder has to be downloaded separately and is not included as is in the repository? That would be very helpful.

*I agree that this was a suboptimal design choice. The amount of data is not extensive enough to bloat the repository, therefore I now directly add it to the repo, and removed the pinned example.zip.*

- [x]  By including the examples folder in the repository you can also write a unit test for the example notebook and make sure that it still runs through when you make any code changes to your library.

*Again, I completely agree that this is a much more solid approach, not resulting in all these problems of not being able to run the examples. I have no added a CI test, that ensures the notebooks run without problems.*

- [x]  The notebook ``ivus_to_centerline.ipynb`` yields the following error, since the folders output and unprocessed do not exist. I would suggest including the empty folders in the examples folder, or creating them on-the-fly with os.makedirs or similar: ValueError: string is not a file: 'output/unprocessed/dia_lumen.obj'

*Thank you very much for this suggestion. I have now uploaded a new version for the ivus and the ccta notebooks to docs/notebooks, which are ensured to run by the CI pipeline.*

## Human and animal research
- [x] I think I didn't find any information on this but is the sample data (in the repo and the zip folder) real human data or synthetic data? Maybe I also overlooked that you elaborated on this somewhere, but of course data should follow the JOSS policies, so if it is not synthetic data I would recommend adding a small statement on the conformity of the data to your repository (and potentially the paper): https://joss.readthedocs.io/en/latest/policies.html#joss-policies.

*Thank you for the thorough review of the policies. The example data originates from the NARCO study and has been fully anonymized. The study was conducted in accordance with the Declaration of Helsinki, with ethics approval from the Kantonale Ethikkommission Bern (KEK 2020-00841). All participants provided written informed consent, and the study is registered with ClinicalTrials.gov (NCT04475289). We have added a corresponding statement to the manuscript.*