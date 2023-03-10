Introduction
==============

**Lyse** is a data analysis system which gets *your code* running on experimental data as it is acquired. It is fundamenally based around the ideas of experimental *shots* and analysis *routines*. A shot is one trial of an experiment, and a routine is a ``Python`` script, written by you, that does something with the measurement data from one or more shots.

Analysis routines can be either *single-shot* or *multi-shot*. This determines what data and functions are available to your code when it runs. A single-shot routine has access to the data from only one shot, and functions available for saving results only to the hdf5 file for that shot. A a multi-shot routine has access to the entire dataset from all the runs that are currently loaded into **lyse**, and has functions available for saving results to an hdf5 file which does not belong to any of the shots---it's a file that exists only to save the "meta results".

Actually things are far less magical than that. The only enforced difference between a single shot routine and a multi-shot routine is a single variable provided to your code when **lyse** runs it. Your code runs in a perfectly clean ``Python`` environment with this one exception: a variable in the global namespace called ``path``, which is a path to an hdf5 file. If you have told **lyse** that your routine is a singleshot one, then this path will point to the hdf5 file for the current shot being analysed. On the other hand, if you've told **lyse** that your routine is a multishot one, then it will be the path to an h5 file that has been selected in **lyse** for saving results to.

The other differences listed above are conventions only (though **lyse**'s design is based around the assumption that you'll follow these conventions most of the time), and pertain to how you use the API that **lyse** provides, which will be different depending on what sort of analysis you're doing.

The **lyse** API
~~~~~~~~~~~~~~~~~

So grea, you've got a single filepath. What data analysis could you possibly do with that? It might seem like you have to still do the same amount of work that you would without an analysis system! Whilst that's not quite true, it's intentionally been designed that way so that you can run your code outside **lyse** with very little modification. Another motivating factor is to minimise the amount of magic black box behaviour, such that an analysis routine is actually just an ordinary ``Python`` script which makes use of an API designed for our purposes. **lyse** is both a program which executes your code, and an API that your code can call on.

To use the API in an analysis routine, begine your code with:

.. code-block:: python

	from lyse import *

The details of the API are found in the :doc:`API reference<api/_autosummary/lyse>`.

**lyse** GUI
~~~~~~~~~~~~~~~

The **lyse** GUI uses the API to apply single and multi-shot routines to collections of shot files, added either manually by the user or automatically by BLACS after shot completion.

Here's a screenshot of **lyse**:

.. _fig-gui:

.. figure:: /img/gui.svg
	
	Screenshot of the Lyse GUI

#. Here's where single shot routines can be added and removed, with the plus and minus buttons. They will be executed in order on each shot (more on how that works shortly). They can be reordered, or enabled/disabled with the checkboxes on the left. The checkboxes to the right, underneath the plot icons don't currently do anything, but they are intended to provide control over how plots generated by the analysis routines are displayed and updated.

#. Here is where multi-shot routines can be added or removed. The file selection button at the top allows you to select what hdf5 file multi-shot routines will get given (to which they will save their results).

#. Allows pausing of analysis. **lyse** by default will run all single-shot routines on a shot when it arrives (either via the HTTP server or having been manually added). After all the shots have been processed, only then will the multi-shot routines be executed. So if you load ten shots in quickly, the multi-shot routines won't run until they've all been processed by the single-shot routines. However most of the time there will be sufficient delay in between shots arriving that multi-shot routines will be executed pretty much every time a new shot arrives.

#. If you want to re-run single-shot analyses on some shots, select them and click this button. They'll then be processed in order.

#. This will rerun all the multi-shot analyses.

#. Here is where shots appear, either having arrived over HTTP of having been added manually via the file browser (by clicking the plus button). Many columns will populate this part of the screen, one for each global and each of the results (as saved by single-shot routines) present in the shots. A high-priority planned feature is to be able to choose exactly which globals and results are displayed. Otherwise this display is overwhelming to the point of uselessness. The data displayed here represents the entirety of what is available to multi-shot routines via the API provided by **lyse**.

#. This is where the output of routines is displayed, errors in red. If you're putting ``print`` statements in your analysis code, here is where to look to see them. Likewise if there's an exception and analysis stops, look here to see why.

.. sectionauthor:: Chris Billington